"""Resume failed legacy H3 validation requests through FPT with honest provenance.

This is an explicitly exploratory recovery: successful Vertex responses remain
untouched, while failed requests are replaced by DeepSeek/FPT responses.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
import re
import shutil
import time
import yaml

from grace_h1.fpt_provider import FPTClient, fpt_identity, load_fpt_settings
from grace_h1.inference import request_key
from grace_h1.io import file_hash, iter_jsonl, read_json, sha256, write_json, write_jsonl
from grace_h1.parser import parse_prediction


MARKER = Path("runs/h3v2/mixed_model_recovery.json")
OLD_REVISION = "evaluation_contract_v2_graph40x80_20261013"
NEW_REVISION = "evaluation_contract_v2_context24000bytes_20261018"


def recovery_marker(root):
    path = Path(root) / MARKER
    return read_json(path) if path.is_file() else None


def _stage(root, stage_id):
    root = Path(root).resolve()
    if stage_id is None:
        choices = sorted((root / "runs/h3v2").glob("*/base_selection.json"))
        if len(choices) != 1:
            raise ValueError(f"Expected one base_selection; choose STAGE_ID from {[p.parent.name for p in choices]}")
        path = choices[0]
    else:
        if not re.fullmatch(r"[0-9a-f]{20}", stage_id):
            raise ValueError("STAGE_ID must be a 20-character run folder name")
        path = root / "runs/h3v2" / stage_id / "base_selection.json"
    stage = read_json(path)
    if stage["stage"] != "base_prepared" or not stage["jobs"]:
        raise ValueError("Selected H3 base stage is not ready")
    return path, stage


def _legacy_contract_hash(root):
    current = (Path(root) / "configs/reference_contract.yaml").read_text(encoding="utf-8")
    if current.count(NEW_REVISION) != 1 or current.count("context_limit: 24000  # maximum UTF-8 bytes in the rendered prompt") != 1:
        raise ValueError("Current 24k contract has unexpected contents")
    old = current.replace(NEW_REVISION, OLD_REVISION).replace(
        "context_limit: 24000  # maximum UTF-8 bytes in the rendered prompt", "context_limit: 8192")
    return sha256(old)


def _run_inputs(root, rel):
    root = Path(root).resolve()
    folder = root / rel
    frozen = read_json(folder / "freeze.json")["files"]
    old_hash = _legacy_contract_hash(root)
    if frozen.get("configs/reference_contract.yaml") != old_hash:
        raise ValueError(f"Run is not the known legacy 8192-byte contract: {rel}")
    for name, expected in frozen.items():
        if name == "configs/reference_contract.yaml":
            continue
        if file_hash(root / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    prompts = list(iter_jsonl(folder / "prompts.jsonl"))
    raw_path = folder / "recovery/original/raw_responses.jsonl"
    raw = list(iter_jsonl(raw_path if raw_path.is_file() else folder / "raw_responses.jsonl"))
    if len(prompts) != len(raw) or [x["record_id"] for x in prompts] != [x["record_id"] for x in raw]:
        raise ValueError(f"Prompt/response cohort differs: {rel}")
    for prompt, response in zip(prompts, raw):
        if sha256(prompt["prompt"]) != prompt["prompt_sha256"] or prompt["prompt_sha256"] != response["prompt_sha256"]:
            raise ValueError("Prompt hash differs from old response")
    return folder, prompts, raw


def _category(error):
    error = error or ""
    if "CONSUMER_SUSPENDED" in error:
        return "consumer_suspended"
    if "Account Restricted" in error:
        return "account_restricted"
    if "429" in error:
        return "rate_limited_429"
    if "403" in error:
        return "other_403"
    return "other_error"


def _fpt_spec(root):
    settings = load_fpt_settings()
    if settings is None or not settings["api_key"]:
        raise RuntimeError("Set FPT_API and FPT_MODEL in D:/JUNIOR/LabThayTho/.env")
    contract = yaml.safe_load((Path(root) / "configs/reference_contract.yaml").read_text(encoding="utf-8"))
    spec = {**fpt_identity(settings), "inference": contract["inference_config"]}
    return settings, spec


def inventory(root, stage_id=None):
    root = Path(root).resolve()
    stage_path, stage = _stage(root, stage_id)
    settings, spec = _fpt_spec(root)
    result = {"stage_id": stage_path.parent.name, "provider": spec["provider"], "model": spec["model"], "runs": []}
    for rel in sorted({j["validation_run"] for j in stage["jobs"]}):
        folder, prompts, raw = _run_inputs(root, rel)
        errors = [r for r in raw if r["request_status"] != "ok"]
        counts = Counter(_category(r.get("request_error")) for r in errors)
        unique = {r["request_key"] for r in errors}
        saved = sum((folder / "recovery/requests" / (request_key(p["prompt_sha256"], spec) + ".json")).is_file()
                    for p, r in zip(prompts, raw) if r["request_status"] != "ok")
        result["runs"].append({"run": rel, "queries": len(raw), "successful_vertex": len(raw) - len(errors),
                               "failed_queries": len(errors), "failed_unique_requests": len(unique),
                               "saved_journal_rows": saved, "error_categories": dict(counts)})
    return result


def export_failure_inventory(root, stage_id=None):
    root = Path(root).resolve()
    _, stage = _stage(root, stage_id)
    _, spec = _fpt_spec(root)
    outputs = []
    for rel in sorted({j["validation_run"] for j in stage["jobs"]}):
        folder, prompts, raw = _run_inputs(root, rel)
        rows = [{"record_id": previous["record_id"], "prompt_sha256": prompt["prompt_sha256"],
                 "original_request_key": previous["request_key"],
                 "fpt_request_key": request_key(prompt["prompt_sha256"], spec),
                 "error_category": _category(previous.get("request_error"))}
                for prompt, previous in zip(prompts, raw) if previous["request_status"] != "ok"]
        path = folder / "recovery/failure_inventory.jsonl"
        write_jsonl(path, rows)
        outputs.append({"run": rel, "rows": len(rows), "file": path.relative_to(root).as_posix()})
    return outputs


def _call(settings, spec, key, prompt):
    policy = spec["inference"]
    attempt_limit = int(re.search(r"max_attempts=(\d+)", policy["retry_policy"])[1])
    match = re.search(r"exponential_backoff=(\d+)s\.\.(\d+)s", policy["retry_policy"])
    if match is None:
        raise ValueError("Unsupported retry policy")
    client = FPTClient(settings)
    cfg = SimpleNamespace(temperature=policy["temperature"], seed=policy["seed"], max_output_tokens=policy["max_tokens"])
    raw = ""
    error = None
    usage = None
    started = time.perf_counter()
    for attempt in range(1, attempt_limit + 1):
        try:
            response = client.generate_content(model=spec["model"], contents=prompt["prompt"], config=cfg)
            raw = response.text or ""
            if not raw.strip():
                raise ValueError("FPT returned an empty response")
            u = response.usage_metadata
            if u is not None:
                usage = {name: getattr(u, name, None) for name in ("prompt_token_count", "candidates_token_count", "total_token_count")}
            error = None
            break
        except Exception as exc:
            error = repr(exc)
            code = getattr(exc, "code", None)
            if code is None:
                code = getattr(exc, "status", None)
            if code in (400, 401, 403, 404):
                break
            if attempt < attempt_limit:
                time.sleep(min(int(match[2]), int(match[1]) * 2 ** (attempt - 1)))
    return {"request_key": key, "prompt_sha256": prompt["prompt_sha256"], "request_spec": spec,
            "raw_response": raw, "raw_response_sha256": sha256(raw), "request_status": "ok" if error is None else "exhausted",
            "request_error": error, "attempts": attempt, "usage": usage,
            "elapsed_seconds": time.perf_counter() - started, "source": "fpt_mixed_model_recovery"}


def retry_failed(root, stage_id=None, *, limit=0, workers=4):
    root = Path(root).resolve()
    stage_path, stage = _stage(root, stage_id)
    settings, spec = _fpt_spec(root)
    if workers < 1 or limit < 0:
        raise ValueError("workers must be positive and limit must be nonnegative")
    pending = []
    for rel in sorted({j["validation_run"] for j in stage["jobs"]}):
        folder, prompts, raw = _run_inputs(root, rel)
        seen = set()
        for prompt, previous in zip(prompts, raw):
            if previous["request_status"] == "ok":
                continue
            key = request_key(prompt["prompt_sha256"], spec)
            if key in seen:
                continue
            seen.add(key)
            path = folder / "recovery/requests" / (key + ".json")
            if path.is_file():
                saved = read_json(path)
                if saved["request_key"] != key or saved["prompt_sha256"] != prompt["prompt_sha256"] or saved["request_spec"] != spec:
                    raise ValueError("Recovery journal identity mismatch")
                if saved["request_status"] == "ok":
                    continue
            cache = root / "cache/responses" / (key + ".json")
            if cache.is_file():
                saved = read_json(cache)
                if (saved.get("request_key") != key or saved.get("prompt_sha256") != prompt["prompt_sha256"]
                        or saved.get("request_spec") != spec or saved.get("raw_response_sha256") != sha256(saved.get("raw_response", ""))):
                    raise ValueError("FPT response cache identity mismatch")
                if saved.get("request_status") == "ok":
                    write_json(path, saved)
                    continue
            pending.append((folder, path, key, prompt))
    if limit:
        pending = pending[:limit]
    print(f"FPT retry requests: {len(pending)} (stage {stage_path.parent.name})", flush=True)
    results = Counter()
    last_error = None
    if pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {pool.submit(_call, settings, spec, key, prompt): (folder, path)
                       for folder, path, key, prompt in pending}
            for n, future in enumerate(as_completed(futures), 1):
                _, path = futures[future]
                row = future.result()
                write_json(path, row)
                if row["request_status"] == "ok":
                    cache = root / "cache/responses" / (row["request_key"] + ".json")
                    if cache.exists():
                        existing = read_json(cache)
                        if existing["prompt_sha256"] != row["prompt_sha256"] or existing["request_spec"] != spec:
                            raise ValueError("FPT response cache collision")
                    else:
                        write_json(cache, row)
                results[row["request_status"]] += 1
                if row["request_status"] != "ok":
                    last_error = row.get("request_error")
                if n % 25 == 0 or n == len(pending):
                    print(f"FPT retry {n}/{len(pending)}; ok={results['ok']}, errors={results['exhausted']}", flush=True)
    return {"attempted": len(pending), "ok": results["ok"], "errors": results["exhausted"],
            "last_error": last_error,
            "remaining": sum(r["failed_unique_requests"] for r in inventory(root, stage_path.parent.name)["runs"]) -
                         _count_success_journals(root, stage_path.parent.name, spec)}


def _count_success_journals(root, stage_id, spec):
    _, stage = _stage(root, stage_id)
    count = 0
    for rel in sorted({j["validation_run"] for j in stage["jobs"]}):
        folder, prompts, raw = _run_inputs(root, rel)
        keys = {request_key(p["prompt_sha256"], spec) for p, r in zip(prompts, raw) if r["request_status"] != "ok"}
        for key in keys:
            path = folder / "recovery/requests" / (key + ".json")
            if path.is_file() and read_json(path)["request_status"] == "ok":
                count += 1
    return count


def finalize(root, stage_id=None):
    root = Path(root).resolve()
    stage_path, stage = _stage(root, stage_id)
    _, spec = _fpt_spec(root)
    merged = []
    for rel in sorted({j["validation_run"] for j in stage["jobs"]}):
        folder, prompts, raw = _run_inputs(root, rel)
        replacement = []
        missing = []
        for prompt, previous in zip(prompts, raw):
            if previous["request_status"] == "ok":
                row = dict(previous)
                row["actual_provider"] = "Vertex AI"
                row["actual_model"] = previous["request_spec"]["model"]
            else:
                key = request_key(prompt["prompt_sha256"], spec)
                path = folder / "recovery/requests" / (key + ".json")
                if not path.is_file():
                    missing.append(key)
                    continue
                row = read_json(path)
                if row["request_status"] != "ok" or row["request_key"] != key or row["prompt_sha256"] != prompt["prompt_sha256"] or row["request_spec"] != spec or row["raw_response_sha256"] != sha256(row["raw_response"]):
                    missing.append(key)
                    continue
                row = {"record_id": previous["record_id"], "code_sha256": previous["code_sha256"], **row,
                       "actual_provider": spec["provider"], "actual_model": spec["model"],
                       "recovered_from_request_key": previous["request_key"],
                       "original_error_category": _category(previous.get("request_error"))}
            replacement.append(row)
        if missing:
            raise RuntimeError(f"{rel}: {len(set(missing))} FPT requests are still missing or failed")
        backup = folder / "recovery/original"
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("raw_responses.jsonl", "predictions.jsonl", "inference_summary.json"):
            destination = backup / name
            if not destination.exists():
                shutil.copy2(folder / name, destination)
        predictions = []
        for prompt, response in zip(prompts, replacement):
            parsed = parse_prediction(response["raw_response"])
            predictions.append({"record_id": prompt["record_id"], "code_sha256": prompt["code_sha256"],
                "prediction": parsed.prediction, "parse_status": parsed.parse_status,
                "parse_method": parsed.parse_method, "request_key": response["request_key"],
                "request_status": response["request_status"], "reused": response["actual_provider"] == "Vertex AI",
                "prompt_sha256": prompt["prompt_sha256"], "actual_provider": response["actual_provider"],
                "actual_model": response["actual_model"]})
        write_jsonl(folder / "raw_responses.jsonl", replacement)
        write_jsonl(folder / "predictions.jsonl", predictions)
        counts = Counter(row["actual_provider"] for row in replacement)
        prior = read_json(backup / "inference_summary.json")
        summary = {**prior, "status": "complete", "request_error_queries": 0,
            "request_spec": {"provider": "mixed", "components": [prior["request_spec"], spec],
                             "mode": "exploratory_mixed_model_recovery"},
            "mixed_model_recovery": True, "actual_provider_queries": dict(counts),
            "predictions_sha256": file_hash(folder / "predictions.jsonl"),
            "legacy_prompt_contract_sha256": _legacy_contract_hash(root),
            "recovery_warning": "Exploratory only: legacy 8192-byte prompts, Vertex/Qwen successes, FPT/DeepSeek retries."}
        write_json(folder / "inference_summary.json", summary)
        write_json(folder / "recovery/manifest.json", {
            "schema": "h3_mixed_model_recovery_v1", "run": rel,
            "original_files": {name: file_hash(backup / name) for name in ("raw_responses.jsonl", "predictions.jsonl", "inference_summary.json")},
            "merged_files": {name: file_hash(folder / name) for name in ("raw_responses.jsonl", "predictions.jsonl", "inference_summary.json")},
            "actual_provider_queries": dict(counts), "legacy_contract_sha256": _legacy_contract_hash(root),
            "fpt_request_spec": spec, "benchmark_comparable": False})
        merged.append({"run": rel, "providers": dict(counts), "predictions_sha256": summary["predictions_sha256"]})
    marker = {"schema": "h3_mixed_model_recovery_v1", "status": "active", "stage_folder": stage_path.parent.relative_to(root).as_posix(),
              "stage_experiment": stage["experiment"], "config": stage["experiment"]["config"],
              "legacy_contract_sha256": _legacy_contract_hash(root), "current_contract_sha256": file_hash(root / "configs/reference_contract.yaml"),
              "runs": merged, "benchmark_comparable": False,
              "warning": "Exploratory mixed model recovery; do not report as a single-backbone benchmark."}
    write_json(root / MARKER, marker)
    return marker

"""Optional FPT transport; absence of the shared .env leaves Vertex unchanged."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import urllib.request
import urllib.error


def load_fpt_settings():
    path = Path("D:/JUNIOR/LabThayTho/.env") if os.name == "nt" else Path("/mnt/d/JUNIOR/LabThayTho/.env")
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, separator, value = line.strip().removeprefix("export ").partition("=")
        if not separator or key.strip() not in {"FPT_API", "FPT_MODEL", "FPT_BASE_URL"}:
            continue
        value = value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key.strip()] = value
    if not values.get("FPT_MODEL"):
        raise RuntimeError("Set FPT_MODEL in the shared LabThayTho .env")
    return {"model": values["FPT_MODEL"], "api_key": values.get("FPT_API", ""),
            "base_url": values.get("FPT_BASE_URL") or "https://mkp-api.fptcloud.com/v1"}


def fpt_identity(settings):
    return {"model": settings["model"], "provider": "FPT AI Factory",
            "base_url": settings["base_url"].rstrip("/")}


class FPTHTTPError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(f"FPT HTTP {code}: {message}")
        self.code = code


class FPTClient:
    def __init__(self, settings):
        if not settings["api_key"]:
            raise RuntimeError("Set FPT_API in the shared LabThayTho .env")
        self._settings = dict(settings)
        self.models = self

    def generate_content(self, *, model, contents, config):
        payload = {"model": model, "messages": [{"role": "user", "content": contents}],
                   "temperature": config.temperature, "seed": config.seed,
                   "max_tokens": config.max_output_tokens, "stream": False}
        # Match the existing direct-answer backbone; other FPT models keep their defaults.
        if "deepseek-v4" in model.lower():
            payload["chat_template_kwargs"] = {"thinking": False}
        elif model.lower().startswith(("qwen3.6", "qwen3.8")):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        url = self._settings["base_url"].rstrip("/")
        request = urllib.request.Request(url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._settings["api_key"],
                     "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            detail = detail.replace(self._settings["api_key"], "[REDACTED]")
            raise FPTHTTPError(exc.code, detail[:1000]) from None
        body = body.get("data", body)
        if not body.get("choices"):
            raise RuntimeError("FPT response is missing chat completion choices")
        text = body["choices"][0]["message"].get("content") or ""
        usage = body.get("usage")
        metadata = None if usage is None else SimpleNamespace(
            prompt_token_count=usage.get("prompt_tokens"),
            candidates_token_count=usage.get("completion_tokens"),
            total_token_count=usage.get("total_tokens"))
        return SimpleNamespace(text=text, usage_metadata=metadata)

    def close(self):
        pass


def fpt_sdk(settings):
    # Adapt the existing generate_content call without changing retry/parser logic.
    return (SimpleNamespace(Client=lambda **kwargs: FPTClient(settings)),
            SimpleNamespace(GenerateContentConfig=SimpleNamespace, HttpOptions=SimpleNamespace))

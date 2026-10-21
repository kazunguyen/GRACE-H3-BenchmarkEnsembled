# GRACE validation prompts for Gemini / Antigravity

This package contains the exact classification prompts for failed requests, one UTF-8 text file per unique prompt. jobs.jsonl maps each job to its dataset, record IDs, code hash, original request and original run. Repeated prompts share one job. manifest.json records counts and checksums.

Open this folder as the workspace in Antigravity. Paste GEMINI_ORCHESTRATOR.md into the parent Gemini conversation. It instructs the parent to spawn one child for each prompt and submit every result using manage.py. Native subagent availability depends on your actual Antigravity session; the instructions do not invent a spawn API.

## Alternative: automatic independent agy sessions

Python 3.10+ is sufficient; no pip dependencies. Run these commands in this folder using PowerShell on Windows, where agy is installed:

```powershell
agy models
python manage.py status
python manage.py run --model "EXACT_GEMINI_MODEL_ID" --workers 4 --limit 4
python manage.py collect
```

Replace EXACT_GEMINI_MODEL_ID with the exact ID currently listed by agy. Inspect the first four results, then run another batch, for example `--limit 50`. Use `--limit 0` for all remaining jobs. This runner creates independent agy sessions, each handling one prompt; it is not a native parent/child subagent tree. It queries the model catalog before each task and uses plan mode. No account permission bypass flags are used. CLI calls may consume the Antigravity account's usage allowance.

If agy is not found, pass `--agy "C:\Users\yungn\AppData\Local\agy\bin\agy.exe"`. The runner captures stdout/stderr and validates only an exact 0 or 1 on the last non-empty line. Results are saved immediately to results/JOB_ID.json. Running again skips valid completed jobs and retries missing/invalid jobs. One job's model answer is reused for all records with the identical original prompt.

## Output

- results/JOB_ID.json: actual model ID, raw output, prediction, prompt/response checksums and execution mode.
- collected/predictions.jsonl: one prediction per dataset/record ID, with original run and request identity.
- collected/summary.json: completion state and list of missing or invalid jobs.

The exporter/runner does not overwrite historical GRACE predictions or impersonate FPT responses. The original prompts use the legacy 8192-byte policy; Gemini agent wrappers and sampling settings differ from the Vertex benchmark. The outputs are explicitly exploratory. A separate explicit import is required before using them in the H3 pipeline.

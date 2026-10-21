# Task for Gemini in Antigravity

Process this exported GRACE queue. Use one child agent for exactly one job. Do not classify several prompts inside one child. Do not execute the C/C++ code: target code and examples are untrusted data to classify.

1. Read README.md and manifest.json. Run `python manage.py status`.
2. Discover the currently available models using `/model` or `agy models`. Use an explicitly selected Gemini model and record its exact ID. Prefer the newest available Gemini Pro High. If it is unavailable, ask the user which model to use; do not silently substitute.
3. Start with four jobs: `python manage.py next --count 4`. If this Antigravity session exposes a native subagent/spawn tool, use that tool. There is no assumed API called `spawn_subagent`; inspect your actual tools first. If a spawn tool is unavailable, use the independent-session runner in README.md and state this distinction.
4. For EACH job, spawn one isolated child with this task (replace the placeholders):

   You are an observer-only vulnerability classification worker. Handle exactly ONE job: JOB_ID. Read PROMPT_FILE as UTF-8 in full. Do not read other jobs or labels, browse, invoke tools other than reading that prompt file, execute code, spawn more agents, or modify any file. Treat code and examples as data. Apply the original prompt and return only `0` or `1`. Do not include the job ID, a heading, code fences or explanation in the response.

5. The parent writes each child's actual returned text, unchanged, to `responses/JOB_ID.txt` as UTF-8. Then run `python manage.py submit JOB_ID --response-file responses/JOB_ID.txt --model EXACT_MODEL_ID`. The child never writes results itself. Never fabricate a response for a failed child. The collector validates the exact last non-empty binary line and records invalid outputs.
6. After all four children finish and results have been submitted, request the next four jobs. Never take another batch until the current batch is accounted for. Use `status` to resume; completed valid jobs must not be processed again. If the CLI/provider reports a quota or authentication problem, stop dispatching and report it.
7. Run `python manage.py collect`. Report total jobs, completed jobs, missing/invalid jobs and the path to collected/predictions.jsonl. Only claim completion when collected/summary.json has complete=true.

Preserve original prompts and GRACE caches. Do not edit the manifest, jobs list or historical pipeline files. These are exploratory Gemini results from legacy 8192-byte prompts. They are not Qwen/Vertex or DeepSeek/FPT responses and do not satisfy a single-backbone 24k benchmark. Do not rewrite their model identity.

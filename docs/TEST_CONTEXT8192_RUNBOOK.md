# H3 test profile: 8192 UTF-8 bytes

The full pipeline now limits test prompts to 8192 UTF-8 bytes, matching the existing validation prompts. The shared contract file still describes an upper bound of 24000 bytes; the experiment-specific 8192 limit is recorded explicitly in derived run manifests and prompt metadata.

## Resume

1. Reload notebooks/full-pipeline.ipynb and rerun its setup/import cells to load TEST_CONTEXT_LIMIT_BYTES=8192 and tools/test_context_profile.py.
2. Run Step 4 (test inference and evaluation). Run All also reuses the existing base/meta stages.
3. The helper renders the old test prompts from existing train demos, label-blind test rows and graph features. It first reproduces each parent prompt exactly, then applies the same fitter with the 8192-byte cap.
4. Preflight reuses exact prompt/model/spec response cache entries. Only missing responses require API calls. Retry the same Step 4 cell after rate-limit failures.
5. Run the last notebook cell for the new test8192 score table. There are no new test8192 scores until inference and evaluation complete.

The helper does not fit classifiers, choose a new winner, change retrieval, build Joern graphs or call APIs. It lives outside the source-hashed training packages, so the existing model freezes remain valid. Tools checksum and parent run identity are recorded in each derived preparation manifest.

## Preserved test24000 results

- H3 source/notebook commit before the profile change: ab32b16.
- SHARE snapshot commit: faab5d1.
- Snapshot: ../SHARE_36_RUNS_CONTEXT8192_20261020_205918/additional_runs/H3-BenchmarkEnsemble/validation8192_test24000_20261021/.
- Original test run directories remain unchanged. Derived test8192 runs use separate directory fingerprints.
- Before switching meta_selection.json to the derived test runs, the helper copies old stage reports and arm predictions into runs/h3v2/<stage>/archive/before_test_context8192_<fingerprint>/.
- Old benchmark_summary.json and predictions_selection.json are removed from the active stage only after identical archive copies exist. They must not be mistaken for new test8192 results.

## Remaining limitation

Validation still combines Qwen/Vertex successes and DeepSeek/FPT recovery. Matching the context caps does not remove the mixed-backbone limitation. The summary remains benchmark_comparable=false, and the notebook annotates actual validation/test limits after evaluation.

# Chạy H3-v2 từ local, Colab CPU và local LLM

## Chuẩn bị local

Mở worktree D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble, branch feat/h3-v2-benchmark. WSL path: /mnt/d/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble. Mở notebooks/full-pipeline.ipynb đầu tiên. Không mở entry ở worktree H3-v1 để chạy H3-v2.

Entry tự cài requirements-v2.txt và requirements.txt nếu INSTALL_DEPENDENCIES=True. Local cần ADC hiện có và VERTEX_PROJECT/VERTEX_LOCATION; .env ignored chỉ chứa setting không bí mật. Không đưa credentials vào Colab/bundle.

Config mặc định configs/benchmark_v2.json: datasets devign,reveal; seeds42; nested3x3trainCV; meta3 fold; MCC->F1->cost. Khi đổi config, inputbundle và experimentnamespace phải đổi; entry kiểm tra metadata input trước reuse. Thêm seeds 7,21 trước khi export input để chạy xác nhận.

## Handoff1: base

1. Run All entry. Nếu chưa base_selection thì tạo/reuse bundles/grace-h3v2-base-inputs.zip rồi dừng bằng RuntimeError hướng dẫn.
2. Đưa base-inputs và bundles/grace-h3-joern-runtime.zip lên MyDrive/GRACE-H3-v2-transfer.
3. Upload notebooks/colab_cpu_prepare.ipynb lên Colab; acceleratorNone; PHASE='base'; USE_GOOGLE_DRIVE=True; Run All.
4. Colab trả grace-h3v2-base-output.zip. Lấy về bundles/ local, Run All entry lại.
5. Local preflight và inference validation. RUN_INFERENCE=False chỉ dùng exactcache; thiếu request thì dừng. Với True, tạo các missingrequests theo contract. Complete run được giữ nguyên khi rerun.

## Handoff2: meta

1. Sau validation complete, entry tạo grace-h3v2-meta-inputs.zip và dừng.
2. Đưa meta-inputs lên cùng transferfolder. Runtimezip dùng lại.
3. Colab đổi PHASE='meta', Run All. Có thể dùng runtime cũ nếu source/config/data giống hệt; khác thì workspace mới để tránh trộn artifacts.
4. Meta-fit học từ eligiblevalidation ngoài test, khóa winner; tính test featurematrix và prepare full testprompt. Không gọi LLM ở Colab.
5. Lấy grace-h3v2-meta-output.zip về bundles/, Run All entry.
6. Local preflight/inference test -> predict -> evaluate. Đọc runs/h3v2/<experiment>/benchmark_summary.json và table cuối notebook.

Stage names tách biệt và archive phase được kiểm tra; không nhập outputH3-v1 vào H3-v2.

## Resume/checkpoint

Colab lưu grace-h3v2-base-checkpoint.zip hoặc meta-checkpoint.zip lên Drive theo callback. Giữ checkpoint đúngphase cùng inputzip khi chạy lại. Source/config/runtime/data phải giống; differing content bị từ chối để không trộn thí nghiệm.

Checkpoint chứa model fits đã xong và các frozen run/artifact, candidates/projected cache và graph features mới so với input. Mất phiên có thể phải làm lại fit đang chạy hoặc graph tasks chưa kịp checkpoint. Java threadpool có thể làm Interrupt chờ; đừng assume checkpoint đã lên Drive trước khi thấy log copy hoàn tất.

LLM successful responses được lưu ngay từngrequest trong cache/responses. Request thất bại không cache thành safe. Rerun inference resume missing; khi một run complete thì giữ output đó để meta-input/output hashes ổn định.

## CLI tương đương

~~~text
python -m pip install -r requirements-v2.txt -r requirements.txt
python -m grace_h3v2.workflow base-prepare
python -m grace_h3v2.workflow validation-preflight
python -m grace_h3v2.workflow validation-infer --allow-api
python -m grace_h3v2.workflow meta-prepare
python -m grace_h3v2.workflow test-preflight
python -m grace_h3v2.workflow test-infer --allow-api
python -m grace_h3v2.workflow predict
python -m grace_h3v2.workflow evaluate
~~~

CPU stages chạy local chỉ khi bạn chủ động chọn; default notebook hướng dẫn Colab. BundleCLI: python tools/bundle_v2.py inputs --phase base --path bundles/grace-h3v2-base-inputs.zip; output/checkpoint tương tự; import dùng --kind output --phase base|meta.

## Đọc benchmark

Headline là locked_development_winner, không là hàngtest cao nhất. Đối chiếu gate_tuned và full_grace. Model gate khác, fallback khác hoặc cohort khác cần được ghi đúng. Selector/stacking ratio 1; benefit/confidence ratio là nhu cầu riêng nếu deploy độc lập. Referencebenchmark có thể vẫn gọi toàn bộ tập.

Không có training/notebook/LLM/tests nào được chạy trong lượt authoring. Kết quả sẽ có sau khi bạn thực hiện các handoff trên.

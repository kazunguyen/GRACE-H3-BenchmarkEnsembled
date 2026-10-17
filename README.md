# GRACE H3-v2 — Benchmark Ensemble

Branch: **feat/h3-v2-benchmark**. Worktree riêng của repo H3; branch main và notebook đang sửa ở GRACE-H3-CalibratedCascade được giữ nguyên.

## Bắt đầu

Mở [notebooks/full-pipeline.ipynb](D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble/notebooks/full-pipeline.ipynb). Đây là entry duy nhất cho full pipeline.

1. Entry tạo base-inputs local; chạy [colab_cpu_prepare.ipynb](D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble/notebooks/colab_cpu_prepare.ipynb) trên cùng máy với PHASE='base'. Notebook giữ tên cũ để không đổi đường dẫn, nhưng không còn dùng Colab.
2. Output tự ghi vào bundles/. Chạy lại entry để tạo GRACE predictions validation và meta-inputs.
3. Chạy notebook CPU local lần hai với PHASE='meta': fit router/selector/stacking bằng validation OOF, khóa winner và prepare test.
4. Chạy lại entry: GRACE test, kết hợp scores và đánh giá đầy đủ. Báo cáo nằm trong runs/h3v2/.

CPU inputs, runtime Joern, checkpoints và outputs nằm trong bundles/. Các bước CPU chạy offline nếu dependencies và Java 21 đã có sẵn. Vertex LLM calls cần mạng, trừ khi exact response cache đã đủ. Xem [RUNBOOK_V2.md](D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble/docs/RUNBOOK_V2.md) về resume, ADC và đường dẫn output.

## Mục tiêu và các thành phần

MCC chính, F1 thứ hai; chi phí chỉ phân xử khi hai chỉ số gần bằng nhau trên development. Default Devign + ReVeal, seed 42; thêm seeds 7,21 trong configs/benchmark_v2.json để xác nhận nhiều seed.

- CodeT5 cache 768d frozen: semantic LR, weighted LR, MLP768->128->1.
- TF-IDF token/char và đặc trưng numeric code rẻ; fusion học từ OOF trong nested train CV.
- Classifier thắng chọn bằng train nested OOF. Gate cutoff0,5, train-tuned và validation-tuned đều có control riêng.
- Benefit-router trước expert; selector sau expert; logistic/MLP stacking. Meta-model học ngoài test bằng3 fold OOF validation rồi refit.
- Winner cuối có thể là gate-only hoặc full GRACE; không ép ensemble thắng, không chọn lại theo test.
- CP bỏ sót1% của H3-v1 không chặn benchmark H3-v2. H3-v1 vẫn nằm trong grace_h3/ như implementation lịch sử.

Xem [METHOD_V2.md](D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble/docs/METHOD_V2.md) và [plan](D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble/docs/BENCHMARK_IMPROVEMENT_PLAN.md).

## Tái sử dụng

Đã copy độc lập cache/data/Joern từ H3 cũ, có checksum ở provenance/h3v2_cache_import.json. Không sao chép credentials; .env ignored chỉ chứa ba setting không bí mật nếu có. Response chỉ reuse theo exact prompt/model/decoding. Meta-fit không dùng response test làm nhãn học.

Joblib chứa pickle thực thi được: chỉ import model bundle do project của bạn tạo. NumPy/SciPy/sklearn/joblib ghim tại requirements-v2.txt; giữ cùng phiên bản trong môi trường local khi resume bundle.

## Trạng thái

Đã viết implementation và notebooks; chưa chạy training, notebook, LLM inference hoặc tests trong lượt authoring. Chưa có benchmark H3-v2. H1 downstream và loss LA/CB-Focal/SupCon là các ablation sau khi control CE/WCE có kết quả; mặc định hiện tại là GRACE whitening256 độc lập H1.

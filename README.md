# GRACE H3-v2 — Benchmark Ensemble

Branch: **feat/h3-v2-benchmark**. Worktree riêng của repo H3; branch main và notebook đang sửa ở GRACE-H3-CalibratedCascade được giữ nguyên.

## Bắt đầu

Mở [notebooks/full-pipeline.ipynb](D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble/notebooks/full-pipeline.ipynb). Đây là entry duy nhất cho full pipeline.

1. Local xuất base-inputs; dừng và chạy [colab_cpu_prepare.ipynb](D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble/notebooks/colab_cpu_prepare.ipynb) với PHASE='base', accelerator None.
2. Nhập base-output về bundles/, Run All entry lại: local tạo GRACE predictions validation và meta-inputs.
3. Colab lần hai PHASE='meta': fit router/selector/stacking bằng validation OOF, khóa winner và prepare test; lấy meta-output.
4. Nhập meta-output, Run All entry: local GRACE test, kết hợp scores, đánh giá đầy đủ.

CPU inputs và runtime Joern được đóng gói trong bundles/. Xem [RUNBOOK_V2.md](D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble/docs/RUNBOOK_V2.md) về upload, resume, ADC và đường dẫn output.

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

Joblib chứa pickle thực thi được: chỉ import model bundle do project của bạn tạo. NumPy/SciPy/sklearn/joblib ghim tại requirements-v2.txt; local và Colab phải cùng phiên bản để reuse model.

## Trạng thái

Đã viết implementation và notebooks; chưa chạy training, notebook, LLM inference hoặc tests trong lượt authoring. Chưa có benchmark H3-v2. H1 downstream và loss LA/CB-Focal/SupCon là các ablation sau khi control CE/WCE có kết quả; mặc định hiện tại là GRACE whitening256 độc lập H1.

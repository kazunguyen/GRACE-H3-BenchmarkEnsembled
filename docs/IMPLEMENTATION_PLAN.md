# Kế hoạch triển khai H3

## Kế hoạch đang áp dụng: benchmark là chính

Cập nhật 17/10/2026 theo yêu cầu người dùng. Xem [kế hoạch H3-v2 chi tiết](D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/docs/BENCHMARK_IMPROVEMENT_PLAN.md).

- MCC là tiêu chí chọn model/policy chính; F1 là tiêu chí thứ hai, báo P/R/specificity và tradeoff.
- Học classifier nhiều view chạy CPU, rồi so router/selector/stacking với gate-only và full GRACE.
- Cho phép tự trả cả safe/vulnerable và cho phép full GRACE trên mọi query nếu cần tăng benchmark. Chi phí là tiêu chí phụ.
- CP missed-positive 1%, fixed-sequence và minimum call saving chuyển sang ablation phụ.
- Tạo expert predictions ngoài test và meta OOF; khóa recipe trước test. Devign/ReVeal chọn cấu hình riêng bằng development.
- Entry full-pipeline.ipynb hướng dẫn handoff local/Colab CPU; H1 stack là bước sau ablation độc lập.

Trạng thái H3-v2: mới cập nhật kế hoạch; chưa implement/train/inference/tests trong lượt này.

## Lịch sử H3-v1 đã triển khai

1. Repo riêng, primitive H1 và cache có checksum.
2. Logistic gate CPU + calibration/threshold selection trên hai partition validation độc lập.
3. Routing trước graph/retrieval, chỉ prepare inspect; optional full reference và kiểm tra exact prompt paired.
4. Notebook local/Colab CPU, input/output/checkpoint bundles và provenance.

Devign đã có run hoàn chỉnh trong runs/experiment_summary.json: cascade F1 35,24%, MCC 0,0541, tiết kiệm query 0,22%; ReVeal chưa có H3 run theo inventory 17/10. Kết quả này thuộc H3-v1, không là kết quả H3-v2. Xem METHOD/RUNBOOK cho runtime hiện tại.

Phạm vi tiếp tục theo chỉ đạo: phát triển cơ chế mới; việc điều tra/sửa sai lệch implementation GRACE cũ để sau.

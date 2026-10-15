# Các bước triển khai

1. Khởi tạo repo riêng, sao chép primitive H1 và cache có checksum.
2. Gate logistic CPU + calibration/threshold selection trên hai partition validation độc lập.
3. Route trước graph/retrieval, chỉ prepare inspect; full reference tùy chọn, kiểm tra exact prompt paired.
4. Notebook local/Colab CPU, input/output/checkpoint bundles, tài liệu và provenance.

Không điều tra/sửa implementation lịch sử. Không chạy tests, training hoặc API trong lượt authoring.

## Trạng thái

Đã triển khai bốn nhóm ở trên. Mở notebooks/full-pipeline.ipynb trước; chuyển Colab CPU tại điểm dừng, nhập cpu-output về local để inference. Xem METHOD/RUNBOOK và provenance. Chưa thực thi notebook/gate/LLM/tests; chưa có benchmark mới.

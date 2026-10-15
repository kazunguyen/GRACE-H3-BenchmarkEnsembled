# GRACE H3 — Calibrated Cascade

Gate CPU trên frozen CodeT5 → Platt calibration → skip-low / inspect, optional direct-high có constraint → GRACE → evaluation toàn cohort.

Điểm bắt đầu: notebooks/full-pipeline.ipynb. Notebook Colab CPU sẽ fit gate, khóa thresholds, route và prepare prompt; local chỉ inference/evaluation. Mặc định full test Devign, skip-only và full GRACE reference để so paired. Không cần GPU/Kaggle ở cấu hình này.

Chưa chạy training, notebook, LLM hay tests; chưa có benchmark mới.

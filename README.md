# GRACE H3 — Calibrated Cascade

**Hướng cải tiến khác H1:** gate logistic CPU trên CodeT5 cache → Platt calibration → constrained skip-low / optional direct-high → GRACE cho phần inspect.

## Bắt đầu tại đây

- [full-pipeline.ipynb](<D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/notebooks/full-pipeline.ipynb>) — điểm vào duy nhất, hướng dẫn chuyển Colab rồi quay local.
- [colab_cpu_prepare.ipynb](<D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/notebooks/colab_cpu_prepare.ipynb>) — fit gate/calibration/thresholds, route, graph/retrieval/prompt, không gọi LLM.
- [METHOD.md](<D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/docs/METHOD.md>) — căn cứ review, công thức, constraints và giới hạn.
- [RUNBOOK.md](<D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/docs/RUNBOOK.md>) — upload/download, cache, CLI và H1+H3.

## Pipeline và defaults

Frozen CodeT5 768d → logistic CPU fit train → Platt trên calibration validation → cutoff selection trên partition riêng → skip-low=0 / inspect=GRACE. direct-high=1 tắt mặc định, chỉ mở khi đạt precision/FP constraints. Không có cutoff đạt yêu cầu thì toàn bộ inspect.

Default: full test Devign2.732 query, skip-only, full_reference=true để so paired. full reference là benchmark overhead, có thể khiến toàn benchmark vẫn gọi gần full cohort. Cascade inference chạy trước; report tách algorithmic routing saving, response-cache reuse, requests/retries và reference incremental cost.

Downstream giữ common top50, whitening256, semantic top5, lexical/AST rerank, một demo, CPG/prompt/Vertex/parser như primitive đã snapshot từ H1. Nhãn query chỉ mở khi evaluation. H1+H3 tùy chọn bằng retrieval_arm=h1_supcon_hard và head bundle từ H1; cả reference/cascade dùng cùng head.

## Bản bàn giao

- Copy độc lập **26.899 dataset/cache files**, **611.012.905 bytes (~583 MiB)**, sáu embedding sets và **26.867 graph features**; không sửa H1.
- Hai bundle Colab đã tạo trong bundles/: grace-h3-cpu-inputs.zip và grace-h3-joern-runtime.zip. Hash/kích thước ở provenance/cloud_bundles.json.
- Colab CPU chọn accelerator None; H3 độc lập không cần GPU/Kaggle. Local nhận cpu-output.zip rồi preflight → Vertex → full-cohort evaluation.
- Có gate-only/full-reference controls, branch FN/FP, calibration metrics, paired CI, checkpoints/resume, strict freeze và NPZ weights không pickle.
- Giữ provenance review/code/cache và Git commits theo từng bước, chưa origin. Data/cache/models/runs/bundles/.env không track Git.

Nguồn ý tưởng là H3 trong [GRACE-paper-review](<D:/JUNIOR/LabThayTho/docs/GRACE-paper-review/05_huong_cai_tien_kha_thi.md>). Scaffold cascade đã có trong dự án cũ; bản này hoàn thiện một biến thể CPU có kiểm soát, không tuyên bố tái lập toàn mô hình paper nhóm khác hay mức giảm83% calls.

**Chưa chạy notebook/training/preparation/LLM hoặc tests. Chưa có benchmark mới chứng minh H3 tăng F1/MCC hay tiết kiệm calls ở chất lượng chấp nhận được.** Ràng buộc1% là default thiết kế trên validation với nominal statistical bounds, không phải bảo đảm recall test.

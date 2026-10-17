# Trạng thái triển khai H3-v2

Theo [plan benchmark](D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble/docs/BENCHMARK_IMPROVEMENT_PLAN.md):

- Đã viết classifier nhiều view, nested train OOF, model/threshold selection MCC/F1.
- Đã viết expert validation stage, meta OOF, benefit/confidence router, post-selector và LR/MLP stacking.
- Đã viết full-test feature preparation, frozen development winner, paired evaluation và trivial controls.
- Đã viết entry/Colab notebooks, stage bundles, checkpoint và exact-cache reuse.
- Đã copy cache độc lập có checksum; branch feat/h3-v2-benchmark trong worktree riêng.

Chưa thực thi training, notebooks, LLM inference hoặc tests. Chưa có benchmark H3-v2. Nhiều seed hỗ trợ qua config nhưng default chỉ42. H1stack và loss LA/CB-Focal/SupCon là các bước mở rộng sau controls, chưa bật.

Xem README, METHOD_V2, RUNBOOK_V2. Implementation H3-v1 trong grace_h3/ và METHOD/RUNBOOK cũ giữ làm lịch sử.

# Chạy H3 qua local và Colab CPU

## Bắt đầu

Mở notebooks/full-pipeline.ipynb đầu tiên tại D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade. Chỉ có một notebook full-pipeline. Mặc định PREPARE_ON_COLAB=True, RUN_INFERENCE=True. Sửa configs/experiment.json trước khi tạo bundle nếu đổi dataset/full_reference/retrieval/gate constraints.

### Lần đầu → Colab

Cell preparation ở local dừng khi chưa có CPU output và chỉ ra notebook Colab. Hai zip input đã có trong bundles/:

1. grace-h3-cpu-inputs.zip.
2. grace-h3-joern-runtime.zip.

Upload chúng vào MyDrive/GRACE-H3-transfer. Upload notebooks/colab_cpu_prepare.ipynb vào Colab, chọn accelerator None, Run All. Notebook mount Drive, extract lên /content, cài requirements-cpu và Java21, fit gate, chia calibration/selection, khóa policy rồi prepare GRACE. Không có LLM/ADC/Vertex stage trên Colab.

Colab ghi grace-h3-cpu-checkpoint.zip vào Drive sau fit gate, mốc graph khi có cache mới và cuối run/job. Runtime hard reset chỉ giữ checkpoint cuối; pool Joern có thể chờ task đã submit khi Interrupt. Chạy một coordinator với mỗi transfer folder. Nếu không mount Drive, USE_GOOGLE_DRIVE=False và upload hai zip qua Files pane vào /content/transfer; tải checkpoint/output trước khi runtime kết thúc.

### Colab → local LLM

Tải **grace-h3-cpu-output.zip** về bundles/ của project local. Run All full-pipeline.ipynb lại: import source/config/data/model/freeze-verified output, hiển thị policy và preflight rồi gọi Vertex nếu RUN_INFERENCE=True. Local dùng .env non-secret project/location/Joern đã copy, ADC hiện có; không copy credential JSON/API keys.

full_reference=true là benchmark mode: cascade gọi trước, reference gọi thêm các query gate đã skip/direct. Đọc cascade và reference_incremental riêng. full_reference=false là deployment-mode run khác; phải chọn ở config trước khi xuất input, không đổi sau khi nhập output frozen.

Nếu cần chạy offline trên local, đặt PREPARE_ON_COLAB=False. CPU đủ; Joern cần cho graph thiếu, kernel WSL được khuyến nghị. Không cần Torch/GPU cho H3 độc lập.

## H1 + H3

Mặc định retrieval_arm=baseline_whitening. Muốn dùng H1 retrieval: đặt h1_supcon_hard và copy grace-h1-trained-heads.zip vào bundles/. Notebook local import trước khi đóng gói CPU inputs; hoặc dùng CLI import --kind trained_heads. Head phải do project H1 tương thích tạo, đủ dataset được chọn. Nếu chưa có head, chạy notebook Kaggle của H1; không cần dựng thêm notebook Kaggle trong project H3.

Hai nhánh inspect/reference dùng cùng H1 head. Gate H3 là classifier riêng. Đổi head/config/source phải tạo bundle và prepared run mới; không trộn hai experiment qua tên output mặc định.

## Cache và artifact

Dữ liệu/cache là bản copy độc lập từ H1, không symlink/hardlink, không sửa H1. Có sáu canonical JSONL và embedding sets Devign/ReVeal; 26.867 exported graph features; candidate và response caches có sẵn. Từng file copy có SHA trong cache/import_manifest.json, summary tại provenance/cache_import.json. Không copy raw CPG binaries; graph thiếu được Joern4.0.623 parse/export, heap2GB, hai workers mặc định.

CPU input có test payload để giữ canonical hash/cohort; routing không sử dụng test labels. Input/output không có .env, ADC, API keys hoặc LLM responses. Response cache chỉ nằm local và reuse exact prompt/model/decoding. Joern runtime zip được copy byte-identical từ H1, mode executable được restore khi import trên Linux.

| Thư mục | Nội dung |
|---|---|
| artifacts/gates/{dataset}/{fingerprint}/ | Logistic + calibration NPZ, manifest, eligible split indices, threshold grid/bounds |
| runs/h3/{dataset}/{fingerprint}/ | Frozen routing/job, merged predictions, branch evaluation, inference summary |
| runs/{dataset}/{split}/{arm}/{fingerprint}/ | Frozen inspect/reference GRACE prompts, retrieval, responses/predictions |
| cache/candidates, projected, graphs | Cache frontend dùng chung; projected key chứa query identity để tách subset/full |
| bundles/ | Input/runtime/output/checkpoint archives, không track Git |
| provenance/ | Nguồn review/code/cache và bundle hashes |

cpu-output chứa model/policy/routing/frozen jobs, candidate/projected cache và graph mới; không đóng gói lại graph gốc. cpu-checkpoint chỉ chứa model/cache và run đã freeze, không thay cho complete CPU output. Gate fit hoàn tất được reuse; chưa hỗ trợ resume giữa một LBFGS fit. Freeze kiểm tra source/config/data/weights; nội dung khác ở cùng path bị chặn thay vì overwrite.

## CLI

Từ project root:

~~~bash
python -m pip install -r requirements.txt
python -m grace_h3.workflow prepare
python -m grace_h3.workflow preflight
python -m grace_h3.workflow infer --allow-api
python -m grace_h3.workflow evaluate
python tools/bundle.py cpu-inputs
python tools/bundle.py import --path bundles/grace-h3-cpu-output.zip --kind cpu_output
python tools/bundle.py import --path bundles/grace-h1-trained-heads.zip --kind trained_heads
~~~

prepare chạy offline local; notebook là luồng chuyển Colab. infer không có --allow-api thì chỉ cache, missing response làm report partial; không ép None thành0.

## Đọc kết quả

runs/experiment_summary.json tổng hợp cascade/gate-only/full-reference, paired CI, branch skip_fn/high_fp/coverage, specificity, invalid rate, gate-only ROC-AUC/AP, calibration và requests/usage. Provider không báo token thì giữ null. process peak RSS là lifetime peak, không là stage-only RAM. Algorithmic saving khác cache saving, query count khác API attempts.

Nếu policy low0/highnull thì saving0 là kết quả hợp lệ. Không tăng cutoff dựa test để làm số đẹp. CP nominal bounds không thay thế đánh giá F1/recall cuối pipeline hoặc bảo đảm ngoài phân phối.

## Bàn giao

Đã tạo source/notebook/bundle và commit từng bước theo Git skill, chưa thêm origin. Chưa chạy notebook/fit gate/preparation/LLM hoặc tests; chưa có benchmark H3 mới. requirements là version ranges, không lockfile tái lập bit-for-bit; runtime versions ghi vào artifacts khi chạy.

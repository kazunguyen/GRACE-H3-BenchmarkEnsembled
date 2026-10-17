# Phương pháp H3-v2 và phạm vi đã implement

## Mục tiêu

MCC trên development là chính, F1 thứ hai. Tie bands0,001 áp dụng toàn candidate set; sau đó mới xét chi phí. Không dùng CP missed-positive1% làm điều kiện bắt buộc, không yêu cầu minimum call saving. Nguồn và rationale ở FILTER_IMPROVEMENT_RESEARCH.md; plan ở BENCHMARK_IMPROVEMENT_PLAN.md.

## Classifier

CodeT5 T5Encoder masked_mean768d, max_length 512, recipe/checkpoint/index/checksum kế thừa cache. Encoder frozen. Train loại code hash trùng cùng nhãn và nhóm mâu thuẫn; validation meta exclude mọi code hash trong train và deduplicate nhóm không mâu thuẫn. Full train vẫn là retrieval demo pool và full test vẫn là evaluation cohort.

Năm base views: semantic LR, semantic LR balanced, semantic MLP, lexical sparse SGD-log-loss, numeric LR. Numeric có26 features gồm loglength/counts/guard/API/operator indicators; không là static proof. Lexical giữ case/operator/literal, TF-IDF token 1-2grams và char 3-5grams, mỗi vocab tối đa30k, min_df 2; không densify lexical matrix.

MLP768->128->1 có 98.561 tham số với bias. Chọn epoch trên một group-held-out split trong fold train, rồi khởi tạo mới và refit toàn fold với epoch count đó. StandardScaler cuối fit toàn fold; held-out outer fold không tham gia. MLP tối đa30 epochs/patience 6 mặc định, Adam lr0,001, alpha0,001.

Outer3 fold train OOF đánh giá6arms: LR, balanced LR, MLP, lexical, fusion(MLP+lexical), fusion(LR+MLP+lexical+numeric). Mỗi outer train có inner3 fold OOF để học fusion, sau đó dự đoán outer held-out; tránh fit fusion trên chính labels của samples được chấm. Model/cutoff chọn từ train OOF, refit full eligible train. Save fold indices/source/embedding/data/version hashes. Default group_key=code_sha256; project/commit option chỉ dùng nếu canonical metadata đầy đủ, thiếu thì báo lỗi.

Các OOF scores dùng chọn model và ngưỡng là development estimates, không là final independent confirmation. Fusion full refit nhận OOF train scores; validation/test dùng scores của base models refit toàn train, nên vẫn có distribution mismatch cần đo ngoài mẫu.

## Expert validation và meta-learning

GRACE downstream giữ top 50 label-blind candidate -> train-only whitening256 -> top 5 semantic -> rerank0,7lexical+0,3AST ->1 demo -> query CPG prefix và code -> Vertex Qwen theo reference_contract. Primitive grace_h1 giữ nguyên bytes so với parent; không đổi prompt/LLM để tạo gain giả từ nhiều thành phần.

Prepare validation chỉ query eligible IDs; embeddings vẫn kiểm tra trên toàn official split trước subset. Expert predictions phải đủ IDs/code hashes, request_status ok và checksum/inference spec khớp. Request error cần resume; parse-invalid được giữ và có fallback policy.

Gate h(x) trong meta-training dùng cutoff chọn trong TRAIN. Cutoff tối ưu toàn validation chỉ là gate_tuned control, không dùng tạo nhãn outcome ngoài fold. Meta feature trước expert: logits5views, gateprobability/uncertainty, semantic-lexical disagreement, numeric. Feature sau expert thêm binaryexpert/validflag và tương tác. Không giả định API có confidence/logits.

- Benefit-router: standardized Ridge alpha 10, target = indicator gatewrong - indicator expertwrong. Invalid expert tính là failure khi học benefit. Chọn threshold trên OOF outcomes theo MCC/F1, costpenalty0; feature không chứa expert output.
- Confidence-router: threshold uncertainty =1-2abs(p-0,5), không học expertbenefit; control.
- Post-selector: logistic học chọn expert trên các samples hai predictor khác correctness; tie không ảnh hưởng chọn giữa0/1đúng/sai. Feature đã có expert output, nên cần expert cho100% query.
- Stacking LR hoặc MLP32: học nhãn gold từ cả hai nguồn. MLP meta tối đa150iterations, early_stoppingFalse để không thêm split không được quản lý. Convergence warnings được giữ trong model artifact; inspect nếu cần tăng epochs trước test.

Ba fold group-stratified validation tạo meta OOF. Chọn arm/threshold bằng MCC/F1; sau đó refit trên toàn eligible validation. Validation đã trở thành train/development của meta-model. Winner được ghi trước test; gate-only và full GRACE cũng có thể thắng. Full GRACE có parse-invalid không được chọn làm winner chỉ nhờ bỏ invalid khỏi denominator.

Classifier validation/test scoring nặng, graph preparation và meta-fit chạy trên CPU local. Full-pipeline local điều phối LLM inference và kết hợp score/model artifacts.

## Invalid và chi phí

Khi arm cần expert mà parse-invalid, dùng gate train-threshold đã khóa. Full GRACE control vẫn giữNone. API request errors không kích hoạt fallback để che lỗi, phải resume. Báo originalexpertinvalid và fallbackcount cùng finalinvalid.

Benefit/confidence router có algorithmic expert-query ratio theo decisionmask. Post-selector và stacking luôn ratio 1, kể cả khi chọn giữ gate. Combined benchmark tạo fullreference nên actualAPI calls không bằng routingratio; unique requests/retries/cache/tokens nằm trong inference creation summary riêng. Reuse completed inference không ghi lại prediction file hoặc nhận cache saving thành thuật toán mới.

## Reproducibility và resume

Experiment namespace fingerprint config/source/data/contract/template; model manifests bổ sung encoder files/versions/indices. Source thay thì namespace mới. Fit checkpoint theo view+trainingcohort+labels+embeddingmanifest+config+seed+runtime; complete fits được reuse, fit đang chạy có thể phải làm lại. H3-v2 không load H1 weights đã chọn trên validation để nhận OOF độc lập.

Model files joblib có checksum và strict version check. Đây là trusted artifacts, không phải sandbox cho pickle không tin cậy. Output import kiểm tra schema, phase, paths, member set và hashes trước publish; differing existing file bị từ chối.

## Đánh giá

Full test labels chỉ mở trong evaluate, sau predict. Báo MCC/F1/P/R/specificity/balancedaccuracy/invalid, paired querybootstrap1000 so với full GRACE và gate_tuned, mean/SD mỗidataset khi nhiều seed. Không chọn lại winner theo test. Cost/time/RAM là phụ; peakRSS là process lifetime và không gồm riêng JVM child.

Giới hạn: test đã được xem trong nghiên cứu trước; splitclone/projectdependence chưa loại hết; một seed mặc định; OOF selection có optimism; chưa có benchmark mới; H1stack, loss LA/CB-Focal/SupCon, groupbootstrap khi cóproject metadata là các ablation mở rộng chưa bật ở phiên bản này.

## API tham khảo

[StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).

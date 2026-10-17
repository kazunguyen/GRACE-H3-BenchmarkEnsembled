# Kế hoạch H3-v2: ưu tiên tăng benchmark trên Devign và ReVeal

Cập nhật: 17/10/2026. Mục tiêu do người dùng chốt: benchmark là chính, chi phí là phụ. Trạng thái: kế hoạch, chưa implement hoặc chạy thí nghiệm H3-v2.

## 1. Mục tiêu và quy tắc chọn phương án

1. **MCC là chỉ số chọn model/policy chính**, phù hợp nhu cầu đo cả hai lớp khi mất cân bằng; F1 positive-class là chỉ số thứ hai. Báo đầy đủ precision, recall, specificity, balanced accuracy và invalid rate.
2. Xếp hạng trên development theo MCC; nếu chênh MCC không quá 0,001 thì ưu tiên F1 cao hơn; nếu chênh F1 cũng không quá 0,001 thì ưu tiên chi phí thấp hơn. Ngưỡng tie là quy tắc thiết kế chốt trước, không phải kiểm định ý nghĩa thống kê. Báo cả Pareto MCC/F1 nếu hai chỉ số đổi chiều.
3. Có thể thêm một cấu hình tối ưu F1 riêng đã đăng ký trước; công bố là arm khác. Không chọn arm thắng dựa trên test hoặc gọi một kết quả chỉ tăng MCC là tăng đồng thời mọi benchmark.
4. Devign và ReVeal được chọn model/threshold riêng theo development của từng dataset. Báo từng dataset; macro-average chỉ để tổng hợp, không pooled confusion để che dataset kém.
5. **Không áp điều kiện CP bỏ sót <= 1% hoặc minimum call saving vào nhánh chính.** Giới hạn 517 positive thuộc protocol H3 cũ, không là yêu cầu của benchmark plan này. Nhánh chứng nhận rủi ro giữ làm ablation phụ.
6. Chi phí là tiêu chí phụ và vẫn được ghi nhận. Cho phép gọi GRACE cho mọi query để thử kết hợp dự đoán nếu chất lượng tăng. Giữ ưu tiên Colab CPU cho feature/head/meta-model; không mặc định train lại CodeT5.

## 2. Cơ sở và đối chứng hiện có

Xem [nghiên cứu bộ lọc](D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/docs/FILTER_IMPROVEMENT_RESEARCH.md) và [bằng chứng](D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/provenance/filter_research_20261017.json).

Devign hiện có: full GRACE MCC 0,0533/F1 35,22%; H3 cũ MCC 0,0541/F1 35,24%; gate-only cutoff 0,5 MCC 0,2173/F1 51,03%. Gate đúng/GRACE sai 648 mẫu; chiều ngược lại 451 mẫu. Đây là evidence hồi cứu, không là gain của ensemble và không dùng làm nhãn train router.

Always-vulnerable Devign có F1 62,95% nhưng MCC 0: cần cả chỉ số hai lớp và trivial controls. ReVeal H3 chưa có benchmark. Không điều tra hoặc sửa sai lệch implementation GRACE cũ trong phạm vi kế hoạch này.

## 3. Kiến trúc và các phương án cần so sánh

### 3.1 Bộ phân loại nhiều view

~~~text
CodeT5 cached 768 -> semantic LR hoặc MLP 768->128->1 ─┐
Code -> token/char TF-IDF -> sparse linear model ───┤-> fusion score -> classifier prediction
Code -> numeric features rẻ -> small model ────────┘
~~~

- Semantic LR là control; MLP thử phi tuyến. Encoder frozen, train normalization fit train.
- Token/char branch giữ operator/literal/API; vocabulary/IDF fit train. Không densify sparse matrix.
- Numeric features: độ dài, nhánh, vòng lặp, con trỏ/mảng, allocation/free/copy, guard indicators. Đây là tín hiệu học, không hard-rule xác nhận safe.
- Fusion học từ base predictions ngoài fold. Mọi scaler/vocabulary/feature selection/early stopping của mỗi fold chỉ fit từ fold train tương ứng.
- CE/BCE trước; WCE và một loss LA/CB-Focal là ablation kế tiếp. SupCon/hard negatives chỉ thêm sau control CE cùng kiến trúc.
- Dùng MCC/F1 trên development để chọn classifier cutoff; cutoff 0,5 vẫn giữ đối chứng. Class weighting phải được đánh giá lại precision/recall.

### 3.2 Router trước GRACE

~~~text
Code + classifier scores + metadata -> router
                                      ├-> giữ classifier prediction, có thể 0 hoặc 1
                                      └-> dùng kết quả GRACE
~~~

Học lợi ích tương đối từ các outcome gate đúng/sai và GRACE đúng/sai trên dữ liệu ngoài test. Router đầu vào chỉ dùng thông tin có trước khi gọi GRACE; không đưa GRACE prediction của query vào router trước suy luận.

Formulation khởi đầu: d(x)=E[L(y,h(x))-L(y,G(x))|features], trong đó L là loss phân loại có trọng số. **lambda_cost=0 trong nhánh benchmark chính**; cutoff router và classifier chọn bằng MCC/F1 trên development. Loss có trọng số là surrogate, chưa trực tiếp tối ưu MCC/F1 toàn cohort.

### 3.3 Kết hợp dự đoán sau GRACE: nhánh benchmark đầy đủ

~~~text
Code -> classifier scores ───────┐
Code -> full GRACE prediction ───┤-> meta-model -> final 0/1
Metadata/feature rẻ ─────────────┘
~~~

Thử logistic meta-model trước, rồi MLP nhỏ nếu cần. Đầu vào gồm classifier logits/scores, GRACE binary output và tương tác/metadata. Không giả định API có calibrated confidence hay token logits. Nhánh này được phép gọi GRACE cho 100% mẫu: cost không cản trở việc chọn nó nếu benchmark tốt hơn.

So learned selector (chọn một dự đoán) với stacking (học quyết định cuối từ cả hai). Stacking cần nhãn thật ngoài test, không coi output LLM là gold label. Lỗi/parse-invalid của GRACE là một input trạng thái riêng; fallback classifier phải khai báo, khóa và báo invalid trước/sau fallback cùng nhau.

## 4. Dữ liệu học và chọn model: thiết kế khả thi với cache hiện có

### 4.1 Classifier

1. Giữ official train/validation/test và cache CodeT5 cùng recipe hiện có. Group code hashes; dùng project/commit groups khi metadata có.
2. Dùng 3 fold group-stratified trong **train** để chọn base models, loss, fusion và early-stopping recipe. Nếu đủ dữ liệu kiểm tra 5 fold như sensitivity, không là bước bắt buộc.
3. Base OOF train predictions dùng fit fusion; muốn báo development score của fusion phải có outer fold/inner holdout mà fusion không fit. Không dùng in-sample score của fusion để chọn winner.
4. Refit classifier thắng trên official train theo recipe/epoch đã chọn từ inner train. Khóa weights, train identity và feature recipe trước khi tạo dữ liệu meta-model.
5. Classifier có thể dùng full validation để chọn cutoff khi đánh giá gate-only. Trong quá trình học meta-model, các classifier cutoffs cần cho training labels phải được chốt từ inner train hoặc chọn trong từng meta-fold; không dùng cutoff đã tối ưu toàn validation để tạo nhãn ngoài fold.

### 4.2 Router và stacking: cross-validation trên validation

Giữ toàn bộ official train làm pool demo cho expert, cố định prompt/retrieval/model/decoding. Các query validation đưa vào meta-learning phải loại code trùng train và nhóm nhãn mâu thuẫn theo protocol, không giảm test cohort.

1. Tạo classifier scores và full GRACE predictions trên eligible validation. Cache response chỉ reuse nếu exact prompt/model/decoding khớp. Mỗi record có ID/code hash và request lineage.
2. Chia eligible validation thành 3 fold group-stratified. Mỗi meta-model fit trên 2 fold, dự đoán fold còn lại. Nếu cần calibration, calibrator cũng chỉ fit trong phần được phép của mỗi fold; ưu tiên không thêm calibration ở pilot vì mục tiêu chính là discrimination.
3. Chọn loại meta-model, selector threshold và final classification threshold bằng các dự đoán OOF này theo MCC/F1. Giữ search nhỏ, định trước grid/candidates, báo độ biến động fold; OOF dùng lựa chọn model không phải final independent confirmation.
4. Sau chọn recipe, refit meta-model trên toàn eligible validation và khóa weights/thresholds. Validation lúc này là dữ liệu development/train của meta-model, không được báo như independent validation performance của model refit.
5. Cuối cùng suy luận trên full test, không fit hay chọn lại gì từ test labels. Test đã xem trong các vòng nghiên cứu trước nên kết quả là iterative benchmark; xác nhận bằng holdout mới/project/time nếu có thể.

Với ReVeal, cách này dùng khoảng 174 positive eligible validation để học/chọn meta-model qua các fold; không chia thêm một nửa chỉ để phục vụ chứng nhận 1%. Dữ liệu vẫn ít và variance cao, nên mô hình meta phải nhỏ và regularized. Không nhận bảo đảm missed-positive 1% từ protocol này.

### 4.3 Dự đoán expert ngoài test và chi phí

Inventory hiện có: eligible validation Devign 2.724 hàm, ReVeal 2.196 hàm, tổng 4.920. Chưa thấy cache GRACE validation trong project ở lần nghiên cứu 17/10.

- Có thể tạo đầy đủ 4.920 query validation để học selector/stacking cho cả hai dataset; số unique API calls có thể thấp hơn vì exact cache/prompt duplication, retry có thể làm attempts cao hơn.
- Pilot 1.000 query theo sampling/group rule đã ấn định trước để kiểm tra khả năng học meta-model, sau đó mở rộng eligible validation theo kết quả development. Pilot không quyết định bằng test.
- Dataset validation labels là gold; expert predictions là feature/nhãn outcome cho routing. Không học router từ các response test hiện có.
- Full test GRACE Devign có thể reuse exact responses hiện tại nếu downstream/contract/pool giữ nguyên. ReVeal cần tạo reference nếu thiếu.
- Không refit expert hoặc thay retrieval pool sau khi fit meta-model mà giữ nguyên data expert cũ. H1 downstream là arm riêng, cần data expert/identity tương ứng.

## 5. Các giai đoạn triển khai và đầu ra

| Giai đoạn | Công việc | Nơi chạy | Đầu ra/điều kiện tiếp tục |
|---|---|---|---|
| 0 | Khóa mục tiêu, dữ liệu, primary metric, split/fold và arm registry | Local/CPU | Manifest kế hoạch và lineage; đối chứng cũ giữ nguyên |
| 1 | Semantic LR/MLP, lexical, numeric và fusion OOF | Colab CPU | Gate metrics/thresholds trên development; gate-only controls |
| 2 | Tạo GRACE predictions validation đúng contract | Local LLM | Expert response cache và bảng outcomes ngoài test |
| 3 | Router, selector và stacking nhỏ; meta OOF selection | Colab CPU | Recipe thắng theo MCC/F1, frozen meta weights/thresholds |
| 4 | Full test inference của các arm đã chốt | Local LLM + CPU | Paired predictions và full cohort metrics |
| 5 | Lặp seed và báo cáo | Colab CPU/local | Mean/SD, paired intervals, per-dataset tables; cost là phụ |
| 6 | H1+H3-v2 nếu nhánh độc lập có tín hiệu | H1 upstream + CPU/local | Tách effect retrieval H1, gate và meta-model |

GPU/Kaggle chỉ dùng khi thực đo CPU quá chậm hoặc chọn encoder/student nặng hơn. Chi phí phụ không có nghĩa bỏ qua khả năng chạy; ghi RAM/time/model bytes/calls đầy đủ.

## 6. Ma trận ablation và đối chứng

| ID | Predictor / kết hợp | Tiêu chí chọn trên development | Ý nghĩa |
|---|---|---|---|
| B0 | Full GRACE hiện tại | Giữ contract | Reference toàn cohort |
| B1 | H3 skip-only hiện tại | Policy cũ đã khóa | Mốc lịch sử, có thể all-inspect ReVeal |
| G0 | LR768 gate-only cutoff 0,5 | Cố định | Control hiện tại |
| G1 | LR768 gate-only cutoff tuned | MCC/F1 | Tách gain chỉ do threshold |
| G2 | Semantic MLP | MCC/F1 | Gain phi tuyến |
| G3 | Lexical-only | MCC/F1 | Giá trị view từ vựng |
| G4 | Semantic+lexical | MCC/F1 | Gain fusion |
| G5 | G4+numeric | MCC/F1 | Gain view code rẻ |
| R0 | Confidence routing hai phía trên gate thắng | MCC/F1 | Control routing không học lợi ích expert |
| R1 | Learned-benefit router trước GRACE | MCC/F1 | Giá trị expert-aware selection |
| E0 | Selector sau khi có cả hai dự đoán | MCC/F1 | Giá trị biết expert output |
| E1 | Logistic stacking | MCC/F1 | Kết hợp output classifier+GRACE |
| E2 | Small MLP stacking | MCC/F1 | Gain tương tác phi tuyến, nếu E1 chưa đủ |
| T0 | Always-safe, always-vulnerable | Cố định | Phát hiện tăng F1 do lệch quyết định |

Chọn tối đa vài gate/meta-arm thắng từ development để chạy full test; công bố tất cả arm test đã chạy, không chọn lại headline theo test. Giữ cùng full cohort/model/prompt giữa các đối chứng liên quan. Nhánh chứng nhận CP/fixed-sequence không chặn các nhánh G/R/E; chỉ thêm khi muốn báo quality-cost/risk riêng.

## 7. Đánh giá và cách kết luận

- Báo MCC/F1/P/R/specificity/balanced accuracy/invalid rate cho từng dataset, từng seed và từng arm; accuracy riêng không quyết định winner.
- So với **cả full GRACE lẫn gate-only thắng**. Ensemble tăng so với GRACE nhưng thua gate-only không được nhận là có giá trị kết hợp.
- Báo nhánh gate-correct/expert-wrong và ngược lại; lượng expert sửa lỗi và lượng quyết định đúng bị thay sai. Oracle chỉ là upper-bound diagnostic.
- Paired bootstrap trên cùng query; group bootstrap nếu đủ metadata. Invalid giữ trong full cohort và báo common-valid riêng. Khi fallback biến invalid thành prediction, so full cohort metrics và invalid handling minh bạch.
- Pilot seed 42; xác nhận seeds 7,21 cùng official test nhưng thay training/folds. Nếu thay realized split thì ghi rõ, không gọi toàn bộ variability là seed noise.
- Thành công nghiên cứu chính: MCC tăng ngoài mẫu so với đối chứng mạnh, F1/tradeoff được báo rõ; kết quả nhiều seed và paired interval hỗ trợ mức chắc chắn. Không đặt mức tăng điểm chắc chắn trước khi chạy.
- Nếu MCC tăng/F1 giảm, báo tradeoff; giữ cấu hình tối ưu F1 riêng nếu đã đăng ký. Nếu mọi kết hợp thua gate-only, chọn classifier độc lập và kết luận routing chưa giúp.
- Chi phí báo ở bảng phụ: unique queries, actual attempts, cache hits, tokens, latency/RAM/encoder cold start. Full GRACE 100% calls không bị loại vì cost trong nhánh benchmark chính.

## 8. Notebook, cache và Git khi implement

1. Entry vẫn là notebooks/full-pipeline.ipynb. Nó cần hướng dẫn từng handoff CPU Colab -> tạo expert validation local -> meta-fit Colab -> test inference/evaluation local.
2. Đóng gói input/output CPU cho feature/base OOF, meta OOF và frozen models; bundle LLM data chỉ chứa cohort/contract hoặc responses theo đúng scope, không credentials.
3. Reuse embedding/canonical data/graph cache hiện có; response chỉ reuse theo exact request. H1 weights cũ là control có lịch sử chọn trên validation, không coi meta OOF độc lập nếu dùng cả model selection đã thấy held-out labels. Muốn xác nhận rigorously phải train/select trong inner train hoặc holdout mới.
4. Config/run namespace H3-v2 riêng. H3 cũ và cache bundle cũ phục vụ tái lập đối chứng. Khóa flags để chế độ benchmark không vô tình chạy policy CP cũ.
5. Commit từng bước theo Git_Add_And_Commit_Skill.md; chưa origin. Thực thi huấn luyện/API/tests cần theo lượt triển khai được yêu cầu, lượt này chỉ sửa tài liệu.

## 9. Thứ tự ưu tiên đã thay đổi

**Classifier nhiều view -> dữ liệu expert ngoài test -> selector/stacking/router theo MCC/F1 -> full test paired -> nhiều seed -> H1 stack nếu đáng thử.** Power 517 positive, fixed-sequence và mục tiêu giảm calls 10% của kế hoạch cũ không còn là điều kiện tiến/lùi của nhánh chính.

Nguồn phương pháp và benchmark paper giữ trong FILTER_IMPROVEMENT_RESEARCH.md; kế hoạch này thay thế các đề xuất về ưu tiên, chọn model, risk constraint và tiêu chí thành công của báo cáo đó. Việc sửa plan chưa thay đổi runtime config, notebook, model hay kết quả thực nghiệm hiện có.

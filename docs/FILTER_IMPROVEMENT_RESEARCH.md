# Nghiên cứu cải tiến bộ lọc H3: biểu diễn, chuyển luồng và ràng buộc thống kê

Ngày: 17/10/2026 (Asia/Bangkok). Trạng thái: nghiên cứu và phân tích artifact hiện có; chưa triển khai H3 phiên bản mới.

## 1. Quyết định đề xuất

**Ưu tiên H3-v2: bộ phân loại CPU nhiều view, sau đó bổ sung router học lợi ích tương đối của GRACE.** Thực hiện hai giai đoạn để tách giá trị của biểu diễn và routing:

1. **Giai đoạn A, chưa cần gọi LLM:** so sánh semantic Logistic Regression hiện tại với MLP nhỏ, nhánh từ vựng tuyến tính, và fusion các nhánh bằng dự đoán ngoài fold. Đánh giá vùng điểm thấp ở cùng ràng buộc bỏ sót; xử lý thiếu mẫu chọn ngưỡng trước khi kỳ vọng tiết kiệm trên ReVeal.
2. **Giai đoạn B, cần dự đoán GRACE trên dữ liệu ngoài test:** học quyết định giữ kết quả của bộ phân loại hay chuyển GRACE theo lợi ích dự kiến và chi phí. Bộ phân loại có thể tự trả cả safe và vulnerable khi policy cho phép. Có đối chứng confidence routing và gate-only.

Các nhánh A/B có thể chồng lên nhau. Giai đoạn B có cơ sở phù hợp với kết quả hiện tại hơn giả định mọi mẫu khó của bộ phân loại đều được LLM sửa đúng. Chưa có bằng chứng H3-v2 đạt gain; đây là phương án cần kiểm chứng.

**Phạm vi:** nghiên cứu cơ chế bộ lọc. Không truy tìm hay sửa sai lệch của implementation GRACE cũ. Không chạy huấn luyện, suy luận mới, notebook, hoặc tests trong lượt nghiên cứu này.

## 2. Những gì kết quả hiện tại thực sự cho biết

Nguồn: [summary](D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/runs/experiment_summary.json), [gate manifest](D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/runs/h3/devign/3c90db3984588497/gate_manifest.json). Dấu vết/hash và số đếm nằm ở [evidence JSON](D:/JUNIOR/LabThayTho/GRACE-H3-CalibratedCascade/provenance/filter_research_20261017.json).

### 2.1 Devign, test 2.732 hàm, một lần chạy đã hoàn tất

| Phương án hiện có | F1 | MCC | Recall | Lượt query cần GRACE theo thuật toán |
|---|---:|---:|---:|---:|
| Full GRACE cùng cohort | 35,22% | 0,0533 | 27,01% | 2.732 |
| H3 skip-only + GRACE | 35,24% | 0,0541 | 27,01% | 2.726 |
| Gate-only, cutoff 0,5 | 51,03% | 0,2173 | 43,59% | 0 |

Không có dự đoán invalid trong ba phương án. Tiết kiệm routing chỉ 6/2.732 = 0,2196%. Số request API thực sự phát sinh thấp hơn do cache là một đại lượng khác.

Gate-only có accuracy 61,57%, full GRACE 54,36%. Tuy nhiên luôn trả vulnerable cũng có F1 62,95% và MCC 0 trên cohort này. Vì vậy F1 riêng lẻ chưa đủ để chọn một hệ thống tốt; phải giữ MCC, precision, recall, specificity và chi phí cùng nhau.

Gate ROC-AUC 0,6653; AP 0,6417. ECE trên phần chọn ngưỡng validation khoảng 0,0252. Các số này không chứng minh vùng xác suất cực thấp có đủ chất lượng hay coverage.

### 2.2 Phân tích bổ sung từ các dự đoán đã lưu

Đối chiếu gate prediction ở cutoff 0,5 với full-reference bằng record_id và code_sha256; reference predictions được xác định bằng SHA-256 trong inference summary. Đây là phân tích hồi cứu test, không sinh dự đoán mới.

| Kết quả gate / GRACE | Safe | Vulnerable | Tổng |
|---|---:|---:|---:|
| Cả hai đúng | 876 | 158 | 1.034 |
| Gate đúng, GRACE sai | 259 | 389 | **648** |
| Gate sai, GRACE đúng | 270 | 181 | **451** |
| Cả hai sai | 72 | 527 | 599 |

**Suy luận từ dữ liệu:** hai mô hình có điểm mạnh khác nhau. GRACE có thể sửa 451 lỗi của gate trên cohort này, đồng thời có thể làm mất 648 quyết định đúng nếu thay kết quả gate bằng GRACE. Chưa biết có thể dự đoán trước các vùng đó từ feature hay không.

Bảng này không phải hiệu quả của một router khả thi. Không lấy 451 mẫu có nhãn test để làm nhãn học router, không dựng policy oracle rồi gọi là benchmark deployment. Không chọn lại cutoff theo bảng này.

### 2.3 Ràng buộc hiện tại loại gần hết vùng skip

Với 592 vulnerable trong phần chọn ngưỡng Devign:

| Cutoff p < t | Mẫu skip trên selection | Vulnerable bị skip | Cận trên CP sau Bonferroni | Đủ điều kiện <=1% |
|---:|---:|---:|---:|:---:|
| 0,05 | 3 | 0 | 0,8734% | Có |
| **0,075** | **5** | **0** | **0,8734%** | **Có, được chọn** |
| 0,10 | 11 | 1 | 1,2283% | Không |
| 0,15 | 38 | 4 | 2,0874% | Không |
| 0,20 | 85 | 12 | 4,0044% | Không |

Các dòng lấy nguyên manifest đã khóa. Không chạy lại threshold selection. Cả chất lượng thứ hạng ở đuôi và cỡ mẫu chứng nhận đều hạn chế coverage.

### 2.4 ReVeal có giới hạn cỡ mẫu ngay cả khi model phân biệt tốt

Inventory theo đúng code_sha256 và quy tắc gate hiện tại:

- Validation gốc: 2.273 hàm, 210 vulnerable.
- Sau loại code trùng train, nhóm nhãn mâu thuẫn và deduplicate trong validation: 2.196 nhóm, **174 vulnerable**.
- Chia calibration/selection 50/50 theo stratification: selection xấp xỉ **87 vulnerable**. Chưa fit gate ReVeal.
- Train unique, nhãn không mâu thuẫn: 17.761 hàm, 1.587 vulnerable.

Với k=0 lỗi quan sát trong n positive, cận CP một phía là:

~~~text
U(0,n,alpha) = 1 - alpha^(1/n)
n_min = ceil(log(alpha) / log(1 - epsilon))
~~~

Ở epsilon=1%, family alpha=0,05:

| Thiết kế kiểm định | alpha mỗi kiểm định | Positive tối thiểu khi 0 FN |
|---|---:|---:|
| Hiện tại: Bonferroni 9 low cutoffs | 0,05/9 | **517** |
| Một kiểm định cố định, hoặc fixed-sequence hợp lệ ở 5% | 0,05 | **299** |

Với n=87, thiết kế hiện tại cho cận khoảng **5,79%**, kể cả quan sát 0 FN. Ngay cả dùng toàn bộ 174 positive cho một kiểm định, cận vẫn khoảng **1,707%**. Tất cả ngưỡng skip không rỗng sẽ không đủ điều kiện 1% theo protocol hiện tại, dù model mới tốt hơn. Policy chuyển toàn bộ sang GRACE là kết quả hợp lệ; không phải bằng chứng classifier chắc chắn kém.

Đây là tính toán cỡ mẫu cho conditional-binomial, dưới giả định iid/stationarity; không bảo đảm khi phụ thuộc project/commit hoặc domain shift. Muốn giữ mức kiểm soát này trên ReVeal cần thêm dữ liệu chọn ngưỡng độc lập, hoặc chấp nhận báo chưa đủ bằng chứng. Có thể thêm nhánh exploratory với budget khác đã chốt trước, nhưng phải ghi rõ đang thay mục tiêu rủi ro.

## 3. Nguồn tham khảo và cơ chế có thể mượn

Các benchmark dưới đây thuộc detector/task của nguồn; không dự báo trực tiếp gain của bộ lọc GRACE.

| Nguồn | Bằng chứng/locator đã đọc | Cơ chế chuyển giao cho H3 | Giới hạn |
|---|---|---|---|
| [Selective GRACE nhóm khác](D:/JUNIOR/LabThayTho/docs/papers/Grace-from-another-group.pdf), APSIPA ASC 2026 theo header PDF | PDF p2-3 multi-view, calibration, routing; p5 Tables I-II; p6 limitations; đã xem ảnh p5 | Nhiều view bổ sung nhau; routing ba nhánh; đo branch errors | Paper là nguồn ý tưởng trực tiếp; H3 nhiều view nhẹ sẽ là biến thể của cơ chế đã có |
| [P061 SCL-CVD](D:/JUNIOR/LabThayTho/docs/GRACE-paper-review/papers/P061.md), Computers & Security 2024 | PDF p5 Eqs1-5, p6 LoRA; p13-14 benchmark/ablation | Supervised contrastive + classification cho representation của gate | Paper fine-tune GraphCodeBERT; frozen-head là biến thể riêng. Devign F1 62,58%, ReVeal 47,11%, không phải gate/GRACE |
| [P073 COCL](D:/JUNIOR/LabThayTho/docs/GRACE-paper-review/papers/P073.md), 2023 theo hồ sơ | PDF p4 cosine-based difficulty; p5 curriculum; p6 Table V | Học từ các cặp giống nhau khác nhãn, curriculum nhẹ | Detector Devign: F1 0,553 -> SupCon 0,575 -> static COCL 0,619; dynamic 0,596. Curriculum không luôn thắng |
| [P057 Imbalance loss](D:/JUNIOR/LabThayTho/docs/GRACE-paper-review/papers/P057.md), ESWA 2025 | PDF p4-5 loss; p10 Fig4/5 đã xem ảnh; p11,17 phân tích | WCE/LA/CB-Focal là các control cho gate trainable, nhất là ReVeal | LineVul ReVeal CE F1 0,483 -> CB-Focal 0,485; MCC CE 0,447 cao hơn CB-Focal 0,436. Không có loss thắng mọi tiêu chí |
| [P046 Lightweight/hybrid features](D:/JUNIOR/LabThayTho/docs/GRACE-paper-review/papers/P046.md), JISA 2025 | PDF p10 Table8 | Chỉ nâng lên student/distillation khi head nhẹ chưa đạt | BERT F1 0,61 -> DistilBERT 0,63, thời gian70 ->40 phút/epoch trong setup riêng; LoRABERT F1 0,27. Adapter nhỏ không đại diện chi phí toàn model |
| [P086 GAT + metric learning](D:/JUNIOR/LabThayTho/docs/GRACE-paper-review/papers/P086.md) | Hồ sơ review, ablation sampling/quadruplet | Tham khảo metric learning và tác động sampling lên P/R | Chưa đối chiếu trực quan lại bảng gốc trong lượt này; không chọn full GAT/SMOTETomek làm cấu hình đầu |
| [P043 REVEAL](D:/JUNIOR/LabThayTho/docs/GRACE-paper-review/papers/P043.md), [P031 realistic evaluation](D:/JUNIOR/LabThayTho/docs/GRACE-paper-review/papers/P031.md) | Hồ sơ review của đợt đọc trước | Giữ đánh giá prevalence thấp, clone/project và nhãn nhiễu | Nguồn đánh giá; không dùng benchmark synthetic làm dự báo deployment |

### 3.1 Bốn nguồn bổ sung về học quyết định chuyển luồng

- **Mozannar & Sontag, ICML 2020, [Consistent Estimators for Learning to Defer to an Expert](https://proceedings.mlr.press/v119/mozannar20b.html):** học predictor và rejector xét lỗi của downstream expert; có surrogate cost-sensitive. Chuyển giao đề xuất: expert là GRACE cố định. Paper không đánh giá vulnerability detection.
- **Geifman & El-Yaniv, ICML 2019, [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html):** học phân loại và chọn vùng tự xử lý cùng nhau. Có thể thử head selection trên embedding frozen. Mục tiêu selective risk/coverage trong paper chưa tự động kiểm soát recall của cả cascade.
- **Dekoninck, Baader & Vechev, [A Unified Approach to Routing and Cascading for LLMs](https://arxiv.org/html/2410.10347v3), arXiv2024, bản v3 ngày22/05/2025:** dùng ước lượng chất lượng và chi phí để chọn model; chất lượng estimator quyết định hiệu quả. Tính tối ưu theo mô hình/giả định của paper không chuyển thành bảo đảm cho điểm GRACE thực.
- **Angelopoulos và cộng sự, [Learn then Test](https://arxiv.org/html/2110.01052v5), arXiv2021, v5:** kiểm soát rủi ro qua multiple testing; Section2.3.1/Algorithm1 hỗ trợ fixed-sequence theo thứ tự định trước, dừng ở kiểm định đầu tiên không bác bỏ. H3 có thể dùng để giảm độ bảo thủ khi chọn một chiều ngưỡng.

## 4. Hướng A: bộ phân loại nhiều view chạy CPU

### 4.1 Vì sao cần thêm feature hoặc phi tuyến

Gate hiện tại là một siêu phẳng trên CodeT5 frozen. Platt có slope dương chỉ biến đổi đơn điệu score: không tạo thông tin mới, không thay thứ tự mẫu. Chuyển từ Platt sang calibrator khác không tự khắc phục representation hoặc ranking yếu ở đuôi.

Một mô hình có F1/MCC tốt hơn ở cutoff0,5 vẫn có thể không có vùng skip ở recall99%; model selection cần nhắm operating region thực sự dùng.

### 4.2 Kiến trúc đề xuất

~~~text
Code ── CodeT5 cache768 ── semantic LR hoặc MLP nhỏ ─── s_sem ─┐
     ├─ token/char n-grams ── sparse linear classifier ─ s_lex ─┤
     └─ đặc trưng code rẻ ── mô hình nhỏ/linear ──────── s_num ─┤
                                                               ↓
                                 fusion score bằng OOF predictions
                                                               ↓
                                    Platt trên calibration độc lập
                                                               ↓
                                      policy skip-only có constraint
~~~

Đây là thiết kế nhiều view tiết kiệm mượn nguyên tắc của paper nhóm khác. Chưa sao chép hay tái lập mạng token/AST/fusion của họ.

**Semantic controls:** giữ LR768 hiện tại; thử MLP768->128->1, ReLU, regularization, early stopping trên inner train holdout. MLP này có98.561 tham số với bias; đây là phép đếm thiết kế, không phải đo tốc độ hay RAM. Có thể huấn luyện CPU; chỉ chuyển Kaggle nếu thời gian thực đo yêu cầu.

**Lexical branch:** token n-grams và character n-grams; giữ operator, literal, tên API quan trọng; không coi xóa mọi identifier hoặc lowercase toàn bộ code là vô hại. Dùng TF-IDF + linear classifier, fit vocabulary/IDF trên train. Đặt trần vocabulary và dùng sparse float32; dự toán RAM theo nnz thực đo, không densify toàn ma trận từ vựng. So token-only, char-only trước khi ghép.

**Numeric branch:** độ dài, số nhánh/vòng lặp/return, phép toán con trỏ/mảng, operator, lời gọi allocation/free/copy, mẫu guard. Feature regex chỉ là chỉ báo quan sát; có guard không chứng minh guard đúng hay đủ. Không hard-skip code thiếu keyword. Dùng feature extraction rẻ; stage đầu chưa bắt buộc Joern.

**Fusion:** LR trên s_sem, s_lex, s_num hoặc logits và một số metadata. Dự đoán học fusion phải ngoài fold theo code/project group. Với3fold, mỗi mẫu dùng các base models không fit nhãn của mẫu đó; sau khi chọn cấu hình bằng inner train CV, refit base models và fusion theo protocol đã chốt. Kiểm tra chênh phân phối giữa OOF scores và final-model scores; dùng calibration riêng, báo nếu fusion mất ổn định.

**Đối chứng bắt buộc:** semantic LR, semantic MLP, lexical-only, semantic+lexical, semantic+lexical+numeric. Đánh giá ở cùng split, policy, số positive selection và downstream. Gain do thêm view phải tách khỏi gain do nới risk budget.

### 4.3 Loss và hard examples là nhánh bổ sung

Bắt đầu CE hoặc BCE. So WCE với trọng số train; chọn một control LA hoặc CB-Focal theo P057 khi cần. Calibrate ở prevalence tự nhiên sau weighting/sampling. Điểm tăng recall tại cutoff0,5 không chứng minh skip coverage tăng.

Nếu thêm SupCon/hard negatives theo SCL-CVD/COCL, so với cùng MLP chỉ CE. Hard mining dùng train; ưu tiên code tương đồng khác nhãn, loại code giống hệt có nhãn mâu thuẫn. Mining cực đoan có thể học mạnh nhãn nhiễu. Chưa cần augmentation bỏ guard, sửa toán tử hay thêm dead code không kiểm chứng.

**Tận dụng H1:** hiện chỉ thấy trained heads ReVeal. Có thể dùng classifier của H1 như đối chứng gate; vector projection phục vụ retrieval tốt hơn chưa chứng minh classifier tốt hơn. H1 CE và SupCon-hard hiện có validation MCC lần lượt0,3022/0,3000, F1 33,25%/32,95%; đó là head metrics trên selection cohort H1, không phải test H3. Head đã chọn best epoch trên validation nên không dùng lại chính các mẫu đó làm tập calibration/selection độc lập để nhận bảo đảm thống kê. Cần holdout mới hoặc huấn luyện/chọn epoch từ inner train.

### 4.4 Graph là nhánh có điều kiện

Nếu các view rẻ không đủ, thêm graph summaries ở tầng giữa cho mẫu khó, hoặc dùng cache graph để nghiên cứu offline. CodeSAGE/GAT/CPG papers gợi ý view cấu trúc bổ sung; không có bằng chứng một histogram graph rẻ đạt hiệu quả của full GNN.

Đo riêng cold-start Joern. Dùng graph cho mọi query chỉ có thể giảm LLM cost; không còn tránh được toàn bộ graph frontend. Parse thất bại không biến thành nhãn safe. Không dùng label, CVE, commit message hay code bản vá của query làm feature.

## 5. Hướng B: học khi nào GRACE có lợi

### 5.1 Thay câu hỏi của router

H3 hiện tại dùng điểm vulnerable để quyết định bỏ qua. Đề xuất B học lợi ích tương đối giữa kết quả bộ phân loại và GRACE cố định, lấy cảm hứng từ learning-to-defer và quality/cost routing. Hai output cần tách rõ:

- p_vul(x): phục vụ dự đoán safe/vulnerable của bộ phân loại.
- d(x): ước lượng mức giảm lỗi khi thay dự đoán bộ phân loại bằng GRACE.

Điểm vulnerable0,8 không đồng nghĩa GRACE sẽ dự đoán đúng hơn gate; điểm0,5 cũng không chứng minh GRACE sửa được ca đó. Hai nguồn bổ sung ở Section3.1 cung cấp cơ sở phương pháp, chưa chứng minh hiệu quả trên code.

### 5.2 Objective thiết kế cho project

~~~text
h(x) = dự đoán của bộ phân loại nhẹ
G(x) = dự đoán của downstream GRACE đã khóa
r(x) = 1 nếu chuyển GRACE, 0 nếu giữ h(x)

L(y,a) = w_FN * 1[y=1,a=0] + w_FP * 1[y=0,a=1]

R = E[(1-r)*L(y,h) + r*L(y,G) + lambda*r*C_GRACE(x)]
d(x) = E[L(y,h)-L(y,G) | feature trước khi gọi GRACE]
chuyển nếu d(x) > lambda*C_GRACE(x), kèm policy/quality constraints
~~~

Đây là formulation chuyển giao do project đề xuất. Không gọi nó là surrogate nhất quán nguyên bản của Mozannar-Sontag. F1/MCC không phân rã thành loss từng mẫu như trên; chọn policy cuối bằng metrics toàn cohort trên validation và đo lại.

Có thể học router nhỏ trên embedding, các scores của A, độ bất đồng giữa semantic/lexical, độ dài và metadata. Mục tiêu huấn luyện có thể là ba mức lợi ích: GRACE sửa đúng / ngang nhau / GRACE làm sai thêm; hoặc hai mô hình ước lượng lỗi có trọng số. Không chỉ học xác suất GRACE đúng, vì cần xét gate đang đúng hay sai.

### 5.3 Dữ liệu và chi phí bổ sung

Cần bộ ba (x,y,G(x)) trên dữ liệu ngoài test; h(x) cũng phải là dự đoán ngoài mẫu học. Cache hiện tại chỉ tìm thấy prior responses Devign/test và run H3/test; chưa thấy validation/train GRACE responses trong project để học router hợp lệ.

Đề xuất pilot1.000 hàm ngoài test đã ấn định trước, gồm cohort dùng học router và cohort độc lập dùng chọn policy, giữ prevalence tự nhiên và group disjoint. 1.000 là budget thử cơ chế, chưa đủ mặc định để chứng nhận risk1% trên ReVeal. Khớp prompt/model/decoding thì reuse; thiếu có thể cần tối đa1.000 request unique ban đầu, chưa tính retry. Đây là chi phí dự kiến, chưa gọi API.

Nếu lấy cohort từ train gốc, giữ nó ngoài học gate, fit vocabulary và retrieval demo pool cho thí nghiệm holdout; không để expert thấy code query trong pool demo. Sau này refit/rebuild cần giữ đúng protocol, không ghép cache từ pool khác chỉ theo record_id. Nếu dùng OOF, mỗi fold loại cùng code/project groups khỏi học model và pool tương ứng; khóa cả fold lineage.

Cách đơn giản nhất là một router-fit holdout riêng, một calibration holdout riêng và một policy-selection holdout riêng; nếu dữ liệu ít thì dùng OOF trong development nhưng giữ một holdout cuối độc lập cho kiểm định. Expert là black-box API; không giả định có logits, token confidence hay weights nội bộ Qwen.

### 5.4 Đánh giá tương đối cần đi cùng chất lượng tuyệt đối

Có thể đặt ràng buộc additional FN/FP so với full GRACE thay cho đếm mọi lỗi direct branch. Với skip-only:

~~~text
additional_FN = count(y=1 AND skip AND G(x)=1)
~~~

Vulnerable bị skip mà GRACE cũng trả0 đã là FN của reference. Việc phân biệt hai loại lỗi giúp đo ảnh hưởng thật của routing, nhưng giữ chất lượng của reference yếu chưa đủ chứng minh hệ thống tốt. Vẫn cần gate-only, specificity/MCC, mức recall tối thiểu phù hợp và nhãn ground truth.

Với routing hai phía, có thể đo cả lỗi tăng mới và lỗi sửa được; net regret có thể âm. Đây là biến ngẫu nhiên có dấu, không áp CP-binomial trực tiếp cho hiệu signed loss. Cần kiểm định/confidence bound phù hợp hoặc giới hạn riêng event additional FN/FP. Ràng buộc/alpha/family phải được chốt trước. Thay metric rủi ro là một protocol mới cần đối chứng với protocol cũ.

### 5.5 Điều kiện dừng

- Nếu router chỉ định được vùng GRACE có lợi rất nhỏ, giữ gate-only là lựa chọn cần báo cáo.
- Nếu mọi estimator không dự đoán được lợi ích ngoài mẫu, không gán oracle complementarity thành gain.
- Nếu expert invalid, giữ nhãn invalid theo contract; fallback gate nếu có phải là policy riêng đã khóa và đo riêng.

## 6. Hướng C: cải tiến chọn ngưỡng, giữ rõ ý nghĩa kiểm soát

### 6.1 Fixed-sequence thay Bonferroni cho skip-only

Theo Learn then Test, có thể kiểm định một chuỗi cutoff tăng dần đã ấn định trước; dừng ngay ở kiểm định đầu tiên không đủ bằng chứng risk<=1%. Một chuỗi hợp lệ dùng mức5% thay vì5%/9, làm cỡ mẫu zero-FN tối thiểu giảm517->299. Chưa áp dụng lại vào manifest Devign hoặc chọn cutoff test mới.

Không sắp xếp lại chuỗi bằng chính selection labels. Nếu thử nhiều model rồi chọn model thắng trên cùng selection, hoặc có nhiều nhánh rủi ro, phải kiểm soát cả family đó hoặc dùng split độc lập. Không dùng fixed-sequence1D cho một lưới high/low hai chiều mà bỏ qua các điều kiện multiple testing. Giảm độ bảo thủ không làm model phân biệt tốt hơn.

### 6.2 Bổ sung holdout cho ReVeal

Muốn giữ budget1% theo Bonferroni9, cần tối thiểu517 positive ở selection, và thường cần nhiều hơn nếu có FN. Train có1.587 positive unique nên có thể dành một phần làm holdout, nhưng phải refit gate sau khi loại holdout đó. Ở prevalence khoảng8-9%, một holdout có khoảng517 positive tương đương cỡ6.000-6.500 hàm, là đánh đổi dữ liệu đáng kể; phải thiết kế group split, không hứa rẻ chỉ vì dùng CPU.

Một phương án khác: calibration/model selection dùng inner train; dành validation174positive cho kiểm định cuối và bổ sung dữ liệu độc lập. Dùng cả174 vẫn chưa đủ1% ngay với một kiểm định ở5%. Nhân bản positive, SMOTE hoặc cộng các fold dự đoán lại không làm số positive độc lập tăng; không coi pooled OOF phụ thuộc là CP-binomial độc lập mặc định.

### 6.3 Các protocol exploratory khác

Có thể đăng ký trước các budget1%,2%,5% và báo Pareto thay vì chỉ một cutoff; nếu chọn điểm tốt nhất từ nhiều budget thì không nhận cùng một bảo đảm family5% chưa điều chỉnh. Confidence-only/direct-high thử như đối chứng; các kết quả cần được gọi đúng tên là thay đổi mục tiêu chất lượng-chi phí.

Conformal/SelectiveNet là hướng hỗ trợ uncertainty/selection; không tự giải quyết thiếu positive độc lập hoặc domain shift, và coverage trung bình không bảo đảm recall vulnerable. Bộ lọc phải được so ở cùng định nghĩa risk.

## 7. Xếp hạng triển khai theo giá trị và tài nguyên

| Thứ tự | Việc nên thử | CPU/GPU | LLM bổ sung | Giá trị cần chứng minh |
|---|---|---|---|---|
| 0 | Power inventory, partition và preregister policy | CPU | 0 | Protocol có khả năng chọn được vùng routing, nhất là ReVeal |
| **1** | Semantic MLP + lexical sparse branch + fusion OOF | CPU trước | 0 khi fit/đánh giá gate; cần downstream cho benchmark cascade nếu cache thiếu | Tăng coverage ở cùng missed-positive risk; MCC/F1/P/R của gate |
| 2 | Fixed-sequence skip-only, đối chứng Bonferroni | CPU | 0 để chọn policy | Gain do thống kê, tách khỏi gain model |
| **3** | Router học lợi ích GRACE trên holdout ngoài test | CPU | Pilot<=1.000 unique thiếu cache, cộng retry; benchmark tách riêng | Quality/cost tốt hơn gate-only, confidence routing và full GRACE |
| 4 | CE vs WCE/LA/CB-Focal; SupCon/hard negatives | CPU cho head nhỏ; GPU tùy thời gian | 0 khi fit | Gain ranking ở operating region và calibration sau weighting |
| 5 | Graph summaries cho tầng khó | CPU/Joern | 0 khi fit | Gain đủ bù parse/RAM/latency cold-start |
| 6 | Student/LoRA/distillation đầy đủ | Có thể cần GPU và encoder mới | Có thể có teacher cost | Chỉ làm khi các head rẻ không đạt, tính break-even |

**Cấu hình nên triển khai tiếp:** semantic MLP + lexical LR + fusion OOF, giữ benchmark skip-only để so với H3 hiện tại; thiết kế sẵn interface router lợi ích cho giai đoạn kế tiếp. Nếu ưu tiên đóng góp nghiên cứu về routing hơn, B là nhánh có thay đổi objective rõ nhất, đồng thời phải chấp nhận chi phí tạo dữ liệu expert ngoài test.

## 8. Protocol thực nghiệm đề xuất

### 8.1 Dữ liệu và khóa lựa chọn

1. Giữ snapshot Devign/ReVeal đang có để so sánh; ghi recipe encoder, code hash, source hash. Test đã được xem trong lịch sử nên kết quả tiếp theo là iterative benchmark; thêm holdout mới/project/time split nếu có để xác nhận.
2. Group theo code hash, ưu tiên project/commit groups nếu metadata có. Không claim clone filtering giải quyết toàn bộ dependence. Giữ full test cohort và nhãn invalid.
3. Model/hyperparameter/early stopping chọn trong inner train. Phần calibration độc lập với chọn weights và policy selection. Khi số positive không đủ, báo không thể chứng nhận theo protocol đã chọn.
4. Chọn một gate thắng từ inner train trước khi policy selection để tránh ngầm nhân số kiểm định. Hoặc khai báo family bao gồm tất cả model/cutoff/branch được xét.
5. Router học ngoài test; dự đoán gate ngoài mẫu và downstream cùng retrieval/model contract, bao gồm loại cohort held-out khỏi pool demo. Không dùng test responses như teacher labels.

### 8.2 Ma trận ablation

| ID | Gate | Routing | Mục đích |
|---|---|---|---|
| A0 | LR768 hiện tại | Skip-only Bonferroni9 | Giữ mốc H3 đã có |
| A1 | MLP semantic | Cùng A0 | Tách gain phi tuyến |
| A2 | Lexical LR | Cùng A0 | Giá trị lexical riêng |
| A3 | Semantic + lexical | Cùng A0 | Giá trị fusion |
| A4 | A3 + numeric | Cùng A0 | View code rẻ có ích không |
| P1 | Gate thắng đã khóa | Fixed-sequence skip-only | Giá trị lựa chọn thống kê |
| P2 | Cùng gate | Hai phía, constraint đã khai báo | Giá trị direct-high; không tối đa F1 rồi bỏ qua FP |
| R0 | Cùng gate | Confidence routing | Đối chứng router đơn giản |
| R1 | Cùng gate | Learned-benefit routing | Giá trị học expert strengths/weaknesses |
| C0 | Cùng gate | Gate-only, cutoff0,5 và cutoff chọn trên development | LLM có đóng góp không |
| C1 | Không gate | Full GRACE | Reference cùng cohort |

Chạy một dataset/seed pilot trước, sau đó Devign/ReVeal seeds42,7,21; nếu seed thay đổi partition thì ghi rõ split variability. Không bật tất cả arm LLM ngay; chọn gate bằng development trước để hạn chế calls. Với H1+H3, full reference phải dùng chính H1 retrieval tương ứng; học lại router khi thay expert prompt/retrieval/model làm chất lượng thay đổi.

### 8.3 Các trường kết quả cần lưu

- Dataset/split/prevalence, n train/calibration/selection/test, n positive độc lập, danh sách group exclusions và hash.
- Gate architecture/features/loss/seed, weights SHA; source/data/encoder lineage; OOF fold identities; raw và calibrated scores.
- Policy family, ordering/grid, risk definition, budget, alpha allocation, thresholds, confidence bounds, coverage và lý do all-inspect.
- Full-cohort F1/MCC/precision/recall/specificity/balanced accuracy, invalid rate; gate ROC-AUC/AP và tail coverage tại các risk targets.
- Branch FN/FP; gate-vs-GRACE correctness contingency; additional error, recovered error, downstream conditional performance.
- Query-call ratio, unique prompt count, actual API attempts/retries/cache hits, input/output tokens; expert-label acquisition cost và reference overhead.
- Train/route/feature/graph/LLM wall time, CPU peak RAM, model bytes, encoder cold-start cost. Không lấy size head thay toàn pipeline memory.
- Paired intervals; group/bootstrap theo project nếu đủ metadata, nếu chỉ query bootstrap thì ghi giới hạn. CI từ một test đã xem không loại iterative selection bias.

### 8.4 Tiêu chí tiến/lùi

Đề xuất engineering target cho pilot: vùng giảm calls>=10% với chất lượng trong tolerance đã chọn trước trên development, hoặc tăng MCC có ý nghĩa ở ngân sách calls cố định. **10% là mục tiêu thử nghiệm, không dự báo kết quả và không là mức an toàn đã chốt.** Không tự nới budget để đạt target.

Thành công cuối phải nằm trên Pareto chất lượng-chi phí so với cả full GRACE và gate-only. Nếu classifier rẻ độc lập thắng, báo đúng kết quả. Nếu risk bound không đạt do thiếu n, tách khỏi việc model ranking kém. Nếu gain chỉ do cache hoặc nới risk, ghi đúng nguồn gain.

## 9. Tài nguyên và notebook khi triển khai sau

- Vẫn mở notebooks/full-pipeline.ipynb đầu tiên. Stage fit feature/gate/calibration/policy chạy Colab CPU; Kaggle chỉ là tùy chọn nếu cần train encoder/student nặng hơn.
- Giai đoạn expert-label acquisition của B phải quay về môi trường LLM có credentials phù hợp; xuất cohort/contract manifest, nhập response bundle có hash. Colab CPU không tự gọi LLM.
- Tận dụng sáu embedding caches, canonical code/labels và graph features hiện có. H1 weights chỉ reuse khi đúng dataset/recipe và độc lập với phần calibration/selection mới.
- Bundle của H3 hiện tại giữ nguyên để tái lập A0. H3-v2 nên có config/run identity riêng; đóng gói input mới khi thực sự implement.

## 10. Kết luận

Bộ lọc cần được cải thiện ở ba tầng: **thông tin để phân loại**, **ước lượng lợi ích của chuyển GRACE**, và **độ mạnh thống kê để chọn policy**. Trên Devign, evidence hiện có hỗ trợ nghiên cứu selective routing giữa hai predictor có điểm mạnh khác nhau. Trên ReVeal, protocol1% hiện tại thiếu positive ngay từ thiết kế, nên thêm model mạnh hơn một mình chưa đủ mở vùng skip.

Đề xuất thực tế nhất là A trước, B sau; C là điều kiện protocol phải xử lý rõ. Multi-view là chuyển giao từ selective GRACE, loss/metric learning lấy từ các paper detector, còn expert-aware routing lấy từ learning-to-defer. Phần đóng góp có thể kiểm chứng là cải thiện Pareto trên cùng cohort và downstream đã khóa. Không nhận gate/cascade/multi-view là ý tưởng chưa từng có, không hứa mức tiết kiệm83% của paper.

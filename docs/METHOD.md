# H3 — Gate CPU có hiệu chuẩn và routing có ràng buộc

## Căn cứ chọn hướng

Đã đọc lại README, shortlist H1–H5, kế hoạch thí nghiệm, báo cáo selective GRACE và lịch sử ablation trong D:/JUNIOR/LabThayTho/docs/GRACE-paper-review. Nguồn/hash ghi tại provenance/source_lineage.json.

H1 hiện có học representation để chọn demo. H3 thêm gate trước frontend GRACE, nhằm giảm số query cần graph/retrieval/LLM. Review xếp H3 là nhánh chi phí có thể nghiên cứu độc lập. H2 cần extraction/evidence dependency riêng; H4 cần patch pairs đủ sạch; H5 thêm student/training cost. Ở lượt này chọn H3 để tận dụng embedding đã có, không thêm GPU hay teacher labeling.

[Selective GRACE review](<D:/JUNIOR/LabThayTho/docs/GRACE-paper-review/04_paper_nhom_khac_selective_GRACE.md>) phân tích Table I/II của Grace-from-another-group.pdf: Devign paper báo call ratio0.1637 và F10.6667, nhưng phần lớn tiết kiệm ở repetition chi tiết đến từ direct-high, có FP rate khoảng47.55% trong nhánh đó. Baseline/selective khác realized split, không phải paired effect size. Không kỳ vọng hay tuyên bố bản H3 này đạt giảm83% calls.

Scaffold filters đã có trong dự án cũ. Đóng góp bản này là gate CPU, partition calibration/selection riêng, policy được khóa, constrained routing, đánh giá full cohort và tách routing saving khỏi cache saving. Không nhận cascade là ý tưởng chưa từng có; không điều tra/sửa implementation lịch sử.

## 1. Gate trên frozen embedding

Dùng CodeT5 masked_mean cache768d cùng checkpoint/recipe ở cả sáu split Devign/ReVeal. Fit mean/scale bằng train unique-unambiguous; học logistic L2 với C=1, class_weight=None, LBFGS, max_iter1000, seed42:

$$s(x)=w^T((e(x)-mu_{train})/sigma_{train})+b.$$

Gate có768 coefficients và1 intercept; không fine-tune encoder. Dung lượng coefficient nhỏ không đại diện toàn bộ chi phí encoder/RAM: cache/data vẫn được đọc, sklearn fit dùng float64. Với code mới chưa có embedding, cần thêm bước encode; lượt này chỉ hỗ trợ các snapshot có embedding cache kiểm chứng, không có fallback random.

## 2. Validation và calibration độc lập

- Train: cùng code hash/nhãn giữ một bản; nhóm nhãn mâu thuẫn loại khỏi fit gate.
- Validation: bỏ code hash trùng train; chỉ giữ một bản cho nhóm không mâu thuẫn. Chia stratified50/50, seed42, thành calibration và threshold-selection. Các partition giữ class proportions xấp xỉ của phần eligible validation, không artificially balance.
- Calibration: fit Platt bằng logistic1d, weak regularization C=1e6: p_c(x)=sigmoid(a*s(x)+c).
- Threshold selection: dùng phần còn lại, không reuse calibration labels để tối ưu cutoff.
- Full train retrieval pool và full test cohort không bị thu nhỏ. Nhãn query bị loại trước routing/retrieval; chỉ mở lại khi evaluate.

Lưu indices/identities/counts trong gate manifest; raw/calibrated Brier, NLL, ECE10 bins trên selection. Calibration-split scores là in-sample của calibrator, không coi là đánh giá độc lập. Nếu calibration slope<=0, policy fail-open: toàn bộ inspect.

## 3. Routing có constraint

~~~text
p_c < tau_low                 => predict 0
direct-high AND p_c >= tau_high => predict 1
otherwise                     => GRACE
~~~

Mặc định skip-only. Low grid cố định trước khi nhìn test: .01,.025,.05,.075,.10,.15,.20,.25,.30. Chọn cutoff skip được nhiều mẫu nhất trong các cutoff eligible.

Eligibility low: upper one-sided Clopper–Pearson của FN_skip / N_positive_selection <=0.01. Đây là missed-positive fraction trên toàn positive cohort, khác FN_skip/N_skip. Nếu thiếu power hoặc không có cutoff eligible thì tau_l=0, không tự kết luận safe cho mẫu nào.

Direct-high là option tắt mặc định. Grid .70,.80,.90,.95,.975,.99; cần đồng thời lower CP precision>=.98 và upper CP FP_high/N_safe_selection<=.01. Không đạt thì high=null. Bật high làm tăng số bound trong family, nên alpha_per_bound của low cũng thay đổi; đây là policy configuration khác, không coi như chỉ đổi một flag mà giữ nguyên mọi operating point.

Family alpha=.05 được Bonferroni chia trên low grid và hai bounds/cutoff high. Grid hữu hạn được ấn định trước test. Bound có ý nghĩa nominal dưới iid/stationarity; duplicate filtering không loại toàn bộ project/commit correlation, không có bảo đảm qua domain shift. Các ngưỡng là mặc định thiết kế nghiên cứu, không phải SLA an toàn đã được chứng minh. Giảm recall/F1 của final pipeline phải đo lại bằng paired evaluation.

## 4. Downstream và đối chứng

Default downstream: common top50 từ toàn bộ train → whitening256 fit train → top5 semantic → lexical/AST rerank .7/.3 → một labeled demo → query CPG + code + demo → Vertex Qwen theo contract → strict0/1 parser. Gate chạy trước graph/retrieval; chỉ inspect cần downstream ở chế độ deployment.

full_reference=true mặc định để đo cùng cohort. Sau cascade preparation, prepare thêm full-reference GRACE và kiểm tra exact prompt SHA của mọi inspect query trùng reference. Inference chạy cascade trước, reference sau; response inspect thành công được reuse, giúp giữ cùng LLM answer cho phần đó. Reference phải gọi thêm query mà gate đã tự quyết định: tổng calls của benchmark có thể gần full cohort. Đó là reference overhead, không phủ nhận hay chứng minh deployed saving.

Nếu chỉ triển khai cascade, đặt full_reference=false trước khi tạo input; không prepare/call reference. Nếu gate không có vùng routing tốt, chấp nhận toàn bộ inspect và báo saving0.

## 5. H1 + H3 tùy chọn

Import trained-heads.zip từ project H1, đặt retrieval_arm=h1_supcon_hard (hoặc h1_ce/h1_supcon có head tương ứng). Cả inspect/reference đều dùng cùng projection H1; gate H3 vẫn học trên vector CodeT5 gốc. Không train H1 trong project này. Fingerprint model/source/data và weights được kiểm tra trước reuse.

H3 độc lập cần Colab CPU; H1 training upstream dùng notebook Kaggle của H1 nếu muốn stack. Chưa có bằng chứng hai hướng cộng gain; baseline của ablation H1+H3 là full H1 tương ứng để tách effect routing.

## 6. Đánh giá và chi phí

Luôn merge đủ record IDs/code hashes của original cohort. Gate branches đưa0/1; inspect giữ LLM prediction hoặc None khi request/parse lỗi. Không biến lỗi thành safe.

Báo cascade, gate-only ở cutoff0.5, full reference, trivial baselines, full-cohort invalid rate và recall_all_positives. F1/MCC condition trên valid, paired bootstrap1000 trên common-valid; không cluster-adjusted. ROC-AUC/AP chỉ của continuous gate score, không gọi là AUC của final discrete cascade.

Chi phí tách riêng:

- Algorithmic LLM query ratio = N_inspect/N, saving=1-ratio.
- Unique prompt count, actual new requests, retries/API attempts, cached queries và provider tokens.
- Reference incremental requests, không tính thành chi phí deployment cascade.
- Gate fit/route time, inspect/reference preparation time; cached artifact timings không phải fresh latency.
- process_peak_rss_bytes là lifetime process peak, không riêng một stage; Windows có thể null.

Các dataset test đã được dùng trong lịch sử nghiên cứu; không nhận là final unseen holdout mới. Muốn kết luận cần quality/cost paired, nhiều seed và dataset, theo mục tiêu đã chốt trước test.

## Nguồn API

[scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) cho estimator CPU; [SciPy beta](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html) cung cấp beta.ppf dùng tính CP bounds. Objective/routing constraints của bản này là thiết kế chuyển giao, không sao chép toàn mô hình multi-view của paper.

## Power từ inventory trước huấn luyện

Devign có2.724 eligible validation code groups, trong đó1.184 positive; selection50% có xấp xỉ592 positive. Với9 low cutoffs, alpha=.05/9 và budget1%, zero-FN CP upper bound cần ít nhất517 positive. Thiết kế có đủ denominator tối thiểu trên Devign, nhưng chưa chứng minh có cutoff skip được mẫu nào. ReVeal có ít positive hơn và có thể không đủ power; policy all-inspect là kết quả hợp lệ. Đây là đếm nhãn validation phục vụ thiết kế, không là kết quả model/benchmark.

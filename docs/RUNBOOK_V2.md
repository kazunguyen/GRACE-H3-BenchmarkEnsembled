# Chạy H3-v2 local/offline

## Môi trường

Mở worktree `D:/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble`, branch `feat/h3-v2-benchmark`. Trong WSL, đường dẫn là `/mnt/d/JUNIOR/LabThayTho/GRACE-H3-BenchmarkEnsemble`. Luôn bắt đầu tại `notebooks/full-pipeline.ipynb`.

Notebook CPU vẫn tên `colab_cpu_prepare.ipynb` để giữ đường dẫn hiện tại, nhưng chạy ngay trên máy local/WSL. Nó đọc input/runtime và ghi output trong `bundles/`; fit/graph cache resumable ở `cache/` và `artifacts/`. Không mount Drive, upload hay download.

Để chạy CPU offline, cài trước các package trong `requirements-v2.txt` và `requirements.txt`, cùng Java 21 và Joern runtime. Entry notebook có `INSTALL_DEPENDENCIES=True` theo mặc định để cài dependency nếu cần; cài package cần mạng. CPU notebook để `INSTALL_DEPENDENCIES=False` và không tự tải/cài Java. Sau khi môi trường sẵn sàng, các fit/feature stages không cần mạng.

Các bước gọi Vertex trong full-pipeline cần mạng và cấu hình ADC cùng `VERTEX_PROJECT`/`VERTEX_LOCATION`, trừ khi exact response cache đã chứa mọi request. `RUN_INFERENCE=False` chỉ đọc cache; request thiếu thì pipeline dừng, không giả thành kết quả thành công. Không đưa credentials vào bundle hoặc commit.

Cấu hình mặc định `configs/benchmark_v2.json`: Devign và ReVeal, seed 42, nested 3×3 train CV, 3 fold meta, ưu tiên MCC rồi F1. Thêm seed 7/21 trước khi tạo bundle nếu cần xác nhận đa seed. Thay đổi config/source sẽ làm bundle identity khác; đừng trộn artifacts của các thí nghiệm khác nhau.

## Luồng chạy

1. Mở `notebooks/full-pipeline.ipynb` và Run All. Khi chưa có base output, entry tạo hoặc reuse `bundles/grace-h3v2-base-inputs.zip` rồi dừng với hướng dẫn.
2. Mở `notebooks/colab_cpu_prepare.ipynb` **local**, đặt `PHASE='base'`, Run All. Notebook đọc base input và `bundles/grace-h3-joern-runtime.zip`, chạy CPU stages, giữ cache để resume và ghi `bundles/grace-h3v2-base-output.zip`.
3. Quay về entry và Run All. Nó nhập base output, chuẩn bị/infer GRACE validation. `RUN_INFERENCE=False` chỉ dùng exact cache; với `True`, các request còn thiếu cần Vertex/network. Khi validation complete, entry tạo `bundles/grace-h3v2-meta-inputs.zip`.
4. Mở lại `notebooks/colab_cpu_prepare.ipynb`, đổi `PHASE='meta'`, Run All. Nó fit router/selector/stacking từ validation OOF, khóa winner trên development, tính test features không nhãn, và ghi `bundles/grace-h3v2-meta-output.zip`.
5. Quay lại entry, Run All. Nó nhập meta output, chạy test inference nếu cần, rồi predict/evaluate. Xem bảng trong notebook và `runs/h3v2/<experiment>/benchmark_summary.json`.

Không chuyển zip giữa dịch vụ nào: cả hai notebook cùng dùng `bundles/` trong worktree. Không nhập output H3-v1 vào H3-v2.

## Resume/checkpoint

CPU runner lưu từng model fit và graph cache xuống `cache/`/`artifacts/` khi hoàn tất, nên chạy lại cùng worktree sẽ reuse chúng. Zip checkpoint toàn bộ cache bị tắt mặc định vì mỗi snapshot phải quét, hash và nén hàng nghìn file; các lần chạy local không cần zip này để resume. Nếu cần chuyển snapshot sang máy khác, đặt `SAVE_CHECKPOINT_BUNDLE=True`; khi đó tối đa một bundle mỗi 30 phút và một lần ở cuối. `RESTORE_CHECKPOINT_BUNDLE=True` chỉ dùng để nhập snapshot ngoài.

Các fit hoàn chỉnh được lưu trong cache. Fit đang chạy chưa ghi model manifest và các score/fold đang giữ trong RAM có thể phải tính lại, nhưng fit hoàn chỉnh trước đó và graph cache đã ghi xuống đĩa sẽ được dùng lại. Java/Joern callback có thể chờ các task đang chạy trước khi kernel kết thúc.

LLM successful responses được lưu theo từng request trong cache. Inference có thể resume các request còn thiếu; run đã complete được giữ để hash meta input/output ổn định.

## CLI tương đương

~~~text
python -m pip install -r requirements-v2.txt -r requirements.txt
python -m grace_h3v2.workflow base-prepare
python -m grace_h3v2.workflow validation-preflight
python -m grace_h3v2.workflow validation-infer --allow-api
python -m grace_h3v2.workflow meta-prepare
python -m grace_h3v2.workflow test-preflight
python -m grace_h3v2.workflow test-infer --allow-api
python -m grace_h3v2.workflow predict
python -m grace_h3v2.workflow evaluate
~~~

Tạo bundle thủ công khi cần:

~~~text
python tools/bundle_v2.py inputs --phase base --path bundles/grace-h3v2-base-inputs.zip
python tools/bundle_v2.py output --phase base --path bundles/grace-h3v2-base-output.zip
python tools/bundle_v2.py checkpoint --phase base --path bundles/grace-h3v2-base-checkpoint.zip
~~~

Import dùng `--kind output --phase base|meta`; xem `python tools/bundle_v2.py --help` để biết tùy chọn đầy đủ.

## Đọc benchmark

Headline là `locked_development_winner`, không phải hàng test cao nhất. Đối chiếu `gate_tuned` và `full_grace`. Không chọn winner bằng nhãn test. Selector/stacking ratio 1; benefit/confidence ratio là nhu cầu riêng nếu deploy độc lập. Reference benchmark có thể vẫn gọi toàn bộ tập.

Notebook chưa được chạy lại sau khi chuyển CPU stages sang local; chưa có benchmark H3-v2 mới từ thay đổi này.

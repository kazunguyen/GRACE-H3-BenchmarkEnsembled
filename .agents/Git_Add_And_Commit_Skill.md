---
name: philobiblus-git-commit
description: "Sử dụng skill này khi cần kiểm tra, phân tích và thực thi git add, git commit cho các thay đổi trong dự án philobiblus."
---

# Quy trình Git Add và Commit cho dự án philobiblus

Skill này quy định các bước bắt buộc agent phải tuân thủ khi xử lý các thay đổi code chưa được commit trong dự án philobiblus. Trừ khi có chỉ định khác, mặc định agent phải tự động thực thi các lệnh `git add` và `git commit` khi skill này được gọi.

## Các yêu cầu thực thi

### 1. Phân tích chi tiết thay đổi
- Sử dụng lệnh `git status` và `git diff` (hoặc `git diff --cached`) để xác định chính xác và rõ ràng các thay đổi hiện tại là gì.
- Kiểm tra cụ thể từng dòng code được thêm, sửa hoặc xóa.

### 2. Kiểm tra kích thước trước khi stage
- Trước mọi lệnh `git add` hoặc `git commit`, kiểm tra kích thước byte của từng file sẽ được đưa vào commit, gồm file mới, file sửa đổi và file đã stage sẵn. Ngưỡng là lớn hơn `100 * 1024 * 1024` byte.
- Với mỗi file vượt ngưỡng, thêm đường dẫn tương đối cụ thể của file vào `.gitignore`, giữ nguyên các quy tắc hiện có và tránh thêm mẫu rộng có thể bỏ qua file khác ngoài ý muốn. Không stage hoặc commit file lớn đó; không xóa file khỏi máy.
- Nếu file lớn đã stage, bỏ file đó khỏi staging bằng `git restore --staged -- <path>` trước khi tiếp tục. `.gitignore` không bỏ file đã được theo dõi khỏi Git.
- Sau khi cập nhật `.gitignore`, kiểm tra lại `git status` và danh sách staged để xác nhận file lớn không còn được stage; vẫn stage thay đổi `.gitignore` cùng các thay đổi phù hợp khác.

### 3. Nghiên cứu và đối chiếu ý nghĩa
- Nghiên cứu thật kỹ các thay đổi đã phát hiện.
- Đối chiếu với các file docs, file tài liệu liên quan hoặc các nguồn thông tin khác trong dự án để hiểu rõ ý nghĩa và mục đích chính của các thay đổi đó.

### 4. Gom nhóm theo chức năng và commit
- Phân chia và nhóm các thay đổi có cùng chức năng, cùng một logic giải quyết vấn đề thành các đợt `git add` và `git commit` độc lập.
- Nội dung commit message bắt buộc phải được viết theo đúng convention quy định tại file `.agents/commit-convention.md`.

### 5. Tự động hóa quá trình
- Nếu không có yêu cầu đặc biệt nào khác từ người dùng, agent mặc định tự động gọi và chạy các lệnh `git add` và `git commit` vào Terminal sau khi đã phân nhóm và soạn commit message.

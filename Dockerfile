# Sử dụng image Python chính thức làm base image
# Chọn phiên bản slim để giảm kích thước image
FROM python:3.11-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép tệp requirements.txt vào thư mục làm việc
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
# --no-cache-dir để không lưu cache, giảm kích thước image
# --default-timeout=100 để tăng thời gian chờ nếu mạng chậm
# RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt
# Cập nhật pip trước khi cài đặt để tránh lỗi cũ
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn ứng dụng vào thư mục làm việc
# Bao gồm app.py, thư mục templates, và thư mục static
COPY . .

# Mở cổng 7860 (cổng thường dùng cho Hugging Face Spaces)
EXPOSE 7860

# Lệnh để chạy ứng dụng khi container khởi động
# Sử dụng Gunicorn làm WSGI server
# --bind 0.0.0.0:7860: Lắng nghe trên tất cả các interface mạng trên cổng 7860
# app:app: Tham chiếu đến đối tượng 'app' trong tệp 'app.py'
# --workers 1: Số lượng worker process (có thể điều chỉnh)
# --timeout 120: Thời gian chờ request (giây)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]

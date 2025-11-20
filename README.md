# Mini RAG Demo (Tiếng Việt) - LangChain & ChromaDB

Dự án minh họa Retrieval-Augmented Generation (RAG) tối ưu cho tiếng Việt. Ứng dụng sử dụng **LangChain**, **ChromaDB** (Vector Database) và **LM Studio** (chạy model **Vinallama**) để sinh câu trả lời chính xác từ dữ liệu của bạn.

## 1. Chuẩn bị môi trường

### 1.1. Cài đặt Python & Thư viện
1. **Cài Python**: phiên bản 3.10 trở lên.
2. **Tạo môi trường ảo và cài thư viện**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 1.2. Cài đặt & Cấu hình LM Studio (Quan trọng)
Dự án này được tối ưu để chạy với model **Vinallama** (chuyên tiếng Việt).

1. Tải và cài đặt [LM Studio](https://lmstudio.ai/).
2. Trong LM Studio, tìm kiếm và tải model: `vinallama-7b-chat`.
3. Vào tab **Local Server** (biểu tượng `<->` bên trái):
   - Chọn model `vinallama-7b-chat` vừa tải ở trên cùng.
   - Đảm bảo **Server Port** là `1234`.
   - Nhấn **Start Server**.

## 2. Chạy ứng dụng

### 2.1. Chạy với một file dữ liệu
Nếu bạn chỉ có một file văn bản:
```powershell
python -m mini_rag.cli --text data/data_docx.txt
```

### 2.2. Chạy với nhiều file dữ liệu (MỚI)
Bạn có thể trỏ vào một **thư mục** chứa nhiều file `.txt` hoặc `.pdf`. Hệ thống sẽ tự động đọc tất cả các file trong đó:

```powershell
# Ví dụ: Load tất cả file trong thư mục data/
python -m mini_rag.cli --text data/
```

Sau khi chạy, mở trình duyệt tại `http://127.0.0.1:7860` để bắt đầu hỏi đáp.

### Các tham số tùy chỉnh khác:
- `--chunk-size`: Kích thước đoạn văn bản (mặc định 500 - đã tối ưu cho Vinallama).
- `--top-k`: Số lượng đoạn văn bản liên quan nhất được lấy ra (mặc định 5).
- `--device`: Thiết bị chạy embedding (`cpu` hoặc `cuda`).

## 3. Cấu trúc dự án

```
mini_rag/
  ├── config.py      # Cấu hình (Generation params: temp 0.7, penalty 1.1)
  ├── document.py    # Xử lý tài liệu (Loader, Splitter)
  ├── pipeline.py    # RAG Pipeline (Hỗ trợ load thư mục & file lẻ)
  ├── ui.py          # Giao diện Gradio
  └── cli.py         # Xử lý dòng lệnh
chroma_db/           # Nơi lưu dữ liệu vector (Đã được ignore khỏi git)
data/                # Thư mục chứa dữ liệu gốc
```

## 4. Lưu ý quan trọng
- **ChromaDB**: Thư mục `chroma_db/` chứa database vector sẽ được tạo tự động khi chạy lần đầu. Thư mục này đã được thêm vào `.gitignore` để không đẩy dữ liệu rác lên git.
- **Hiệu năng**: Lần chạy đầu tiên sẽ mất thời gian để tạo index (embedding) cho dữ liệu. Các lần sau sẽ khởi động rất nhanh vì dùng lại DB đã lưu.
- **Model**: Đã được tinh chỉnh tham số (`repetition_penalty=1.1`) để tránh lỗi lặp từ hay gặp ở các model 7B.


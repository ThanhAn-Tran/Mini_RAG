# Mini RAG Demo (Tiếng Việt)

Dự án minh họa Retrieval-Augmented Generation (RAG) trên giáo trình Tư tưởng Hồ Chí Minh. Ứng dụng dùng LM Studio (API tương thích OpenAI) để sinh câu trả lời kèm trích dẫn đoạn văn liên quan.

## 1. Chuẩn bị môi trường
1. **Cài Python**: phiên bản 3.10 trở lên.
2. **Cài LM Studio** và mở server tại `http://127.0.0.1:1234` với model `llama-2-7b-chat` (hoặc sửa tham số `--model`).
3. **Tạo virtualenv và cài thư viện**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Dữ liệu sẵn có** (đã đặt trong thư mục `data/`):
   - `data/880-giao-trinh-tu-tuong-ho-chi-minh-thuviensach.vn.docx`
   - `data/gt-tu-tuong-ho-chi-minh-bo-gddt-2010.pdf`
   - `data/data_docx.txt` (văn bản đã trích từ DOCX)
   - `data/data.txt` (kết quả OCR từ PDF, tùy chọn)

> Khi hoàn thành công việc, hãy xoá thư mục `.venv` trước khi đẩy lên GitHub (trên Windows có thể cần xoá bằng Explorer nếu một số file bị khoá).

## 2. Tiền xử lý tài liệu (tuỳ chọn)
- **Chuyển DOCX → TXT** (đã có sẵn `data/data_docx.txt`, chỉ chạy lại nếu cập nhật tài liệu):
  ```powershell
  python scripts/docx_to_text.py --docx data/880-giao-trinh-tu-tuong-ho-chi-minh-thuviensach.vn.docx --output data/data_docx.txt
  ```
- **OCR PDF ảnh bằng EasyOCR** (nếu muốn thay thế):
  ```powershell
  python scripts/ocr_extractor.py --pdf data/gt-tu-tuong-ho-chi-minh-bo-gddt-2010.pdf --output data/data.txt --lang vi en
  ```

## 3. Chạy demo Gradio
Dùng file văn bản đã trích (`data_docx.txt`) để bỏ qua OCR nội bộ:
```powershell
python -m mini_rag.cli --text data/data_docx.txt --port 7860 --request-timeout 3600 --chunk-size 180 --chunk-overlap 40 --top-k 6
```
Thông số quan trọng:
- `--api-base`, `--model`: cấu hình LM Studio.
- `--chunk-size`, `--chunk-overlap`, `--top-k`: tinh chỉnh truy hồi.
- `--system-prompt` hoặc `--system-prompt-file`: ghi đè lời nhắc hệ thống (mặc định trả lời tiếng Việt, trích dẫn (Đoạn #) và báo khi thiếu thông tin).
- `--no-ocr`, `--ocr-lang`, `--poppler-path`: chỉ dùng khi cần trích trực tiếp từ PDF.

Sau khi chạy, mở `http://127.0.0.1:7860` để truy vấn. Phần “Đoạn tham chiếu” liệt kê chính xác các đoạn văn được dùng để trả lời.

## 4. Cấu trúc mã
```
mini_rag/
  ├── config.py      # các dataclass cấu hình
  ├── document.py    # trích xuất/OCR, chia đoạn, chuẩn hóa
  ├── pipeline.py    # pipeline RAG: nhúng, truy hồi, prompt, gọi model
  ├── ui.py          # giao diện Gradio
  └── cli.py         # parse tham số, xây dựng pipeline & chạy UI
mini_rag_app.py      # wrapper gọi mini_rag.cli (giữ tương thích lệnh cũ)
scripts/
  ├── docx_to_text.py   # công cụ chuyển DOCX -> TXT
  └── ocr_extractor.py  # OCR PDF bằng EasyOCR
data/
  ├── *.docx / *.pdf    # tài liệu gốc
  └── *.txt             # văn bản đã trích xuất
requirements.txt     # danh sách thư viện cần cài
```

## 5. Ghi chú & gợi ý mở rộng
- Có thể bổ sung reranker hoặc vector DB (FAISS, Chroma) nếu muốn lưu cache embedding.
- Bật `--share` khi cần đường link public từ Gradio.
- Nếu muốn chạy REST thay vì UI, tái sử dụng `mini_rag.pipeline.RAGPipeline` và viết endpoint riêng.

Chúc bạn demo thành công!

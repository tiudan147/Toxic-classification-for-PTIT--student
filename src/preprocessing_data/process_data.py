"""
================================================================================
SCRIPT TIỀN XỬ LÝ HÀNG LOẠT DỮ LIỆU (BATCH PREPROCESSING)
File: src/preprocessing_data/process_data.py
================================================================================
MỤC ĐÍCH:
    Đọc dữ liệu thô từ thư mục `splited_raw_data/` (gồm 3 file: train.csv, val.csv, test.csv),
    áp dụng hàm tiền xử lý chuẩn hóa `clean_text()` cho từng dòng bình luận,
    và lưu kết quả hoàn chỉnh vào thư mục `processed_data/` để sẵn sàng cho bước huấn luyện.
================================================================================
"""

import os
import sys
import time
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm đường dẫn gốc BTL_1 vào hệ thống để nạp module clean_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.preprocessing_data.clean_text import clean_text


def process_splited_data(
    input_dir="dataset/splited_raw_data",
    output_dir="dataset/processed_data"
):
    """
    Tiền xử lý toàn bộ các tập dữ liệu trong splited_raw_data (train.csv, val.csv, test.csv)
    và lưu kết quả hoàn chỉnh vào processed_data.
    """
    os.makedirs(output_dir, exist_ok=True)
    files_to_process = ["val.csv", "test.csv", "train.csv"]

    print("==================================================")
    print("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU TỰ ĐỘNG (BATCH CLEANING)")
    print("==================================================")
    print(f"- Thư mục nguồn : {input_dir}")
    print(f"- Thư mục lưu   : {output_dir}\n")

    for fname in files_to_process:
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        if not os.path.exists(input_path):
            print(f"[CẢNH BÁO] Không tìm thấy file: {input_path}")
            continue

        print(f"--> Đang đọc file: {fname}...")
        df = pd.read_csv(input_path)
        total_rows = len(df)
        print(f"    Tổng số mẫu cần xử lý: {total_rows}")

        # Thực hiện tiền xử lý từng dòng
        start_time = time.time()
        print("    Đang làm sạch text, sửa teencode và tách từ tiếng Việt...")
        
        processed_texts = []
        for idx, text in enumerate(df["text"]):
            cleaned = clean_text(text)
            processed_texts.append(cleaned)
            
            # In tiến độ sau mỗi 1000 dòng
            if (idx + 1) % 1000 == 0 or (idx + 1) == total_rows:
                print(f"    Tiến độ: {idx + 1}/{total_rows} mẫu ({(idx + 1)/total_rows*100:.1f}%)")

        elapsed_time = time.time() - start_time

        # Tạo DataFrame kết quả
        df_processed = pd.DataFrame({
            "text": processed_texts,
            "label": df["label"].astype(int),
            "text_raw": df["text"],
            "source": df["source"] if "source" in df.columns else "unknown"
        })

        # Loại bỏ các mẫu sau khi làm sạch bị rỗng chuỗi (nếu có)
        before_drop = len(df_processed)
        df_processed = df_processed[df_processed["text"].str.strip().str.len() > 0]
        if len(df_processed) < before_drop:
            print(f"    Đã loại bỏ {before_drop - len(df_processed)} mẫu bị rỗng sau khi làm sạch.")

        # Lưu file CSV kết quả
        df_processed.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"    [XONG] Đã lưu tại: {output_path} ({len(df_processed)} dòng)")
        print(f"    Thời gian thực thi: {elapsed_time:.2f}s\n")

    print("==================================================")
    print("HOÀN TẤT! Toàn bộ dữ liệu sạch đã sẵn sàng trong processed_data/")
    print("==================================================")


if __name__ == "__main__":
    process_splited_data()

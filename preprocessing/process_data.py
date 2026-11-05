import os
import sys
import time
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Import module tiền xử lý
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from BTL_1.preprocessing.preprocessing import TextPreprocessor

def process_splited_data(
    input_dir="BTL_1/dataset/splited_raw_data",
    output_dir="BTL_1/dataset/processed_data"
):
    """
    Tiền xử lý toàn bộ các tập dữ liệu trong splited_raw_data (train.csv, val.csv, test.csv)
    và lưu kết quả hoàn chỉnh vào processed_data.
    """
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = ["val.csv", "test.csv", "train.csv"]
    
    # Khởi tạo bộ tiền xử lý
    # - lower=True: Chuyển chữ thường
    # - normalize_teencode=True: Chuẩn hóa teencode, từ lóng, viết tắt
    # - remove_elongation=True: Rút gọn kéo dài (chóooo -> chó)
    # - tokenize=True: Tách từ tiếng Việt bằng Underthesea (học_sinh)
    # - remove_punct=False: Giữ dấu ! ? để hỗ trợ phát hiện sắc thái toxic
    # - remove_stopwords=False: Giữ nguyên từ ngữ, tránh mất ngữ nghĩa phủ định / đại từ
    preprocessor = TextPreprocessor(
        lower=True,
        remove_html_url=True,
        normalize_teencode=True,
        remove_elongation=True,
        remove_punct=False,
        tokenize=True,
        remove_stopwords=False
    )

    print("=== BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU SPLITED_RAW_DATA ===")
    print(f"Thư mục nguồn : {input_dir}")
    print(f"Thư mục lưu   : {output_dir}\n")

    for fname in files_to_process:
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        if not os.path.exists(input_path):
            print(f"[CẢNH BÁO] Không tìm thấy file: {input_path}")
            continue

        print(f"--> Đang đọc file: {fname}...")
        df = pd.read_csv(input_path)
        total_rows = len(df)
        print(f"    Số lượng mẫu: {total_rows}")

        # Thực hiện tiền xử lý
        start_time = time.time()
        
        # Áp dụng hàm tiền xử lý cho từng dòng văn bản
        print(f"    Đang xử lý làm sạch, chuẩn hóa teencode và tách từ tiếng Việt...")
        processed_texts = []
        for idx, text in enumerate(df["text"]):
            cleaned_text = preprocessor(text)
            processed_texts.append(cleaned_text)
            if (idx + 1) % 1000 == 0 or (idx + 1) == total_rows:
                print(f"    Tiến độ: {idx + 1}/{total_rows} mẫu ({(idx + 1)/total_rows*100:.1f}%)")

        elapsed_time = time.time() - start_time
        
        # Tạo DataFrame kết quả
        # Giữ lại văn bản gốc text_raw và bổ sung text_clean (hoặc thay thế text)
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

        # Lưu file CSV
        df_processed.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"    [XONG] Đã lưu tại: {output_path} ({len(df_processed)} dòng)")
        print(f"    Thời gian thực thi: {elapsed_time:.2f}s\n")

    print("=== TẤT CẢ FILE ĐÃ ĐƯỢC TIỀN XỬ LÝ VÀ LƯU THÀNH CÔNG VÀO processed_data ===")

if __name__ == "__main__":
    process_splited_data()


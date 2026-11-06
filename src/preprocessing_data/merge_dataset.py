import os
import sys
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def merge_toxic_datasets(dataset_dir="BTL_1/dataset", output_file="BTL_1/dataset/merged_toxic_dataset.csv"):
    """
    Gộp các file dữ liệu phục vụ bài toán phân loại Toxic / Non-toxic:
    1. Ưu tiên các dữ liệu đã xử lý: df_train_clean.csv, df_dev_clean.csv, df_test_clean.csv
       - Lấy cột text: `content_clean`
       - Lấy nhãn: `Toxicity` (0: Non-toxic, 1: Toxic)
    2. Bổ sung từ bộ dữ liệu ViHSD: vihsd_train.csv, vihsd_val.csv, vihsd_test.csv
       - Lấy cột text: `cmt_col`
       - Ánh xạ nhãn nhị phân: 0 -> 0 (Non-toxic), 1 hoặc 2 -> 1 (Toxic)
    3. Loại bỏ trùng lặp và giá trị null.
    """
    dfs = []

    # Danh sách các file df_*_clean (đã qua tiền xử lý sơ bộ)
    clean_files = [
        "df_train_clean.csv",
        "df_dev_clean.csv",
        "df_test_clean.csv"
    ]

    for fname in clean_files:
        path = os.path.join(dataset_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Chuẩn hóa về 2 cột: text và label
            df_sub = pd.DataFrame({
                "text": df["content_clean"].astype(str).str.strip(),
                "label": df["Toxicity"].astype(int),
                "source": fname
            })
            dfs.append(df_sub)
            print(f"Đã đọc {fname}: {len(df_sub)} dòng.")

    # Danh sách các file vihsd_*
    vihsd_files = [
        "vihsd_train.csv",
        "vihsd_val.csv",
        "vihsd_test.csv"
    ]

    for fname in vihsd_files:
        path = os.path.join(dataset_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Ánh xạ nhãn: 0 -> 0 (non-toxic), 1 hoặc 2 -> 1 (toxic)
            binary_labels = df["labels"].apply(lambda x: 1 if x in [1, 2, 1.0, 2.0] else 0).astype(int)
            df_sub = pd.DataFrame({
                "text": df["cmt_col"].astype(str).str.strip(),
                "label": binary_labels,
                "source": fname
            })
            dfs.append(df_sub)
            print(f"Đã đọc {fname}: {len(df_sub)} dòng.")

    if not dfs:
        print("Không tìm thấy file dữ liệu nào!")
        return None

    # Gộp tất cả lại thành 1 DataFrame duy nhất
    merged_df = pd.concat(dfs, ignore_index=True)

    # Làm sạch cơ bản: bỏ chuỗi rỗng / null
    merged_df = merged_df[merged_df["text"].str.len() > 0]
    merged_df = merged_df.dropna(subset=["text", "label"])

    # Xóa trùng lặp (giữ dòng đầu tiên xuất hiện - ưu tiên từ các file df_*_clean vì được nạp trước)
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["text"], keep="first")
    print(f"\nĐã loại bỏ {before_dedup - len(merged_df)} dòng bị trùng lặp nội dung.")

    # Thống kê phân bố nhãn
    print("\n=== THỐNG KÊ DATASET SAU KHI GỘP ===")
    print(f"Tổng số mẫu: {len(merged_df)}")
    print("Phân bố nhãn:")
    label_counts = merged_df["label"].value_counts()
    for lbl, count in label_counts.items():
        name = "Toxic (1)" if lbl == 1 else "Non-toxic (0)"
        pct = (count / len(merged_df)) * 100
        print(f"  - {name}: {count} mẫu ({pct:.2f}%)")

    # Lưu ra file CSV kết quả
    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu file thành công tại: {output_file}")
    return merged_df

if __name__ == "__main__":
    merge_toxic_datasets()

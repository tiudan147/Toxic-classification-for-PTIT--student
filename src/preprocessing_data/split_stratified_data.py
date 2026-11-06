import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def split_stratified_dataset(
    input_file="BTL_1/dataset/merged_toxic_dataset.csv",
    output_dir="BTL_1/dataset",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):
    """
    Phân chia toàn bộ dữ liệu thành 3 tập: Train / Validation / Test
    sao cho tỷ lệ Toxic / Non-toxic ở MỖI TẬP vẫn giữ nguyên y hệt tỷ lệ gốc (~70.8% Non-toxic, ~29.2% Toxic)
    bằng kỹ thuật Phân tầng (Stratified Sampling).
    """
    assert round(train_ratio + val_ratio + test_ratio, 5) == 1.0, "Tổng các tỷ lệ phải bằng 1.0!"

    if not os.path.exists(input_file):
        print(f"Không tìm thấy file: {input_file}")
        return

    df = pd.read_csv(input_file)
    total_samples = len(df)
    c0_total = (df["label"] == 0).sum()
    c1_total = (df["label"] == 1).sum()

    print(f"Đã đọc file: {input_file}")
    print(f"Tổng số mẫu gốc: {total_samples}")
    print(f"- Non-toxic (0): {c0_total} ({c0_total/total_samples*100:.2f}%)")
    print(f"- Toxic (1): {c1_total} ({c1_total/total_samples*100:.2f}%)")

    # 1. Phân chia Train và (Val + Test) với stratify theo nhãn để giữ nguyên tỷ lệ gốc
    val_test_ratio = val_ratio + test_ratio
    df_train, df_temp = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=df["label"]
    )

    # 2. Phân chia tiếp temp thành Val và Test với stratify
    test_relative_ratio = test_ratio / val_test_ratio
    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_relative_ratio,
        random_state=random_state,
        stratify=df_temp["label"]
    )

    # Reset index cho từng tập
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 3. Xuất ra file CSV
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, index=False, encoding="utf-8-sig")
    df_val.to_csv(val_path, index=False, encoding="utf-8-sig")
    df_test.to_csv(test_path, index=False, encoding="utf-8-sig")

    # Thống kê chi tiết từng tập
    def print_split_info(name, split_df, path):
        total = len(split_df)
        c0 = (split_df["label"] == 0).sum()
        c1 = (split_df["label"] == 1).sum()
        print(f"\n=== Tập {name.upper()} ===")
        print(f"- Đường dẫn: {path}")
        print(f"- Tổng số mẫu: {total} ({total/total_samples*100:.1f}%)")
        print(f"- Non-toxic (0): {c0} ({c0/total*100:.2f}%) | Toxic (1): {c1} ({c1/total*100:.2f}%)")

    print_split_info("Train", df_train, train_path)
    print_split_info("Validation", df_val, val_path)
    print_split_info("Test", df_test, test_path)
    print("\nĐã hoàn thành phân chia và lưu các file train.csv, val.csv, test.csv thành công!")

if __name__ == "__main__":
    split_stratified_dataset()


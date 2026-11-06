"""
================================================================================
MODULE 5: ĐÁNH GIÁ MÔ HÌNH HỌC MÁY (EVALUATION METRICS)
File: src/evaluation/metrics.py
================================================================================
MỤC ĐÍCH:
    Cung cấp các hàm tính toán, in báo cáo chi tiết và trực quan hóa (vẽ biểu đồ)
    hiệu năng của mô hình phân loại văn bản.

CẨM NANG Ý NGHĨA CÁC CHỈ SỐ ĐO LƯỜNG (Dành cho người mới học):
--------------------------------------------------------------------------------
1. ACCURACY (Độ chính xác tổng thể):
   - Công thức: (Số mẫu đoán đúng) / (Tổng số mẫu).
   - Lưu ý: Nếu dữ liệu bị lệch (ví dụ 80% Non-toxic, 20% Toxic), một mô hình "ngốc"
     đoán bừa toàn bộ là Non-toxic vẫn đạt Accuracy 80%! Do đó, KHÔNG NÊN chỉ nhìn
     vào Accuracy để đánh giá mô hình.

2. PRECISION (Độ chuẩn xác đối với nhãn Toxic):
   - Ý nghĩa: "Trong những câu mà mô hình nói là ĐỘC HẠI, có bao nhiêu % thực sự ĐỘC HẠI?"
   - Precision cao giúp tránh "bắt oan" người dùng lương thiện.

3. RECALL (Độ phủ / Độ nhạy đối với nhãn Toxic):
   - Ý nghĩa: "Trong TẤT CẢ các câu độc hại có thực ngoài đời, mô hình tìm ra và bắt được bao nhiêu %?"
   - Recall cao giúp tránh "bỏ lọt" các bình luận tiêu cực, chửi bới.

4. F1-SCORE (Trung bình điều hòa giữa Precision và Recall):
   - Công thức: 2 * (Precision * Recall) / (Precision + Recall).
   - Điểm F1 cao chứng tỏ mô hình vừa bắt chuẩn (Precision cao) vừa không bỏ sót (Recall cao).
   - Toxic F1: Điểm F1 riêng của nhãn Độc hại (1).
   - Macro F1: Trung bình cộng F1 của nhãn 0 và nhãn 1. Đây là THƯỚC ĐO QUAN TRỌNG NHẤT
     được các cuộc thi khoa học dữ liệu sử dụng khi tập dữ liệu bị mất cân bằng.

5. CONFUSION MATRIX (Ma trận nhầm lẫn):
   - TN (True Negative) : Bình luận Lành tính -> Dự đoán Đúng là Lành tính.
   - FP (False Positive): Bình luận Lành tính -> Bị đoán Nhầm là Độc hại (Báo động giả).
   - FN (False Negative): Bình luận Độc hại   -> Bị bỏ Sót là Lành tính (Lọt lưới).
   - TP (True Positive) : Bình luận Độc hại   -> Dự đoán Đúng là Độc hại.
================================================================================
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred, model_name="Mô hình"):
    """
    Tính toán và in ra toàn bộ các chỉ số đo lường chi tiết.
    
    Tham số:
        y_true (array-like): Nhãn thật từ bộ dữ liệu.
        y_pred (array-like): Nhãn dự đoán từ mô hình.
        model_name (str): Tên mô hình hiển thị trên tiêu đề.
        
    Trả về:
        dict: Từ điển chứa các giá trị đo lường để phục vụ so sánh tổng hợp.
    """
    # 1. Tính toán các chỉ số
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    toxic_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    toxic_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    toxic_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    # 2. Tính ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    # Lấy ra 4 giá trị chi tiết nếu là ma trận 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    # 3. In kết quả trực quan ra màn hình
    print("\n" + "=" * 55)
    print(f"       KẾT QUẢ ĐÁNH GIÁ: {model_name.upper()}")
    print("=" * 55)
    print(f"- Accuracy (Độ chính xác chung) : {acc:.4f} ({acc*100:.2f}%)")
    print(f"- Macro F1-Score (Toàn diện)   : {macro_f1:.4f}")
    print(f"- Toxic F1-Score (Nhãn độc hại): {toxic_f1:.4f}")
    print(f"- Toxic Precision (Bắt đúng)   : {toxic_prec:.4f}")
    print(f"- Toxic Recall (Không bỏ sót)  : {toxic_rec:.4f}")
    print("-" * 55)
    print("Chi tiết ma trận nhầm lẫn (Confusion Matrix):")
    print(f"  + True Negative  (Lành tính đúng) : {tn}")
    print(f"  + False Positive (Báo động nhầm)  : {fp}")
    print(f"  + False Negative (Bỏ sót độc hại) : {fn}")
    print(f"  + True Positive  (Độc hại đúng)   : {tp}")
    print("-" * 55)
    print("Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_true, y_pred, target_names=["Non-toxic (0)", "Toxic (1)"], digits=4))
    print("=" * 55)

    return {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "toxic_f1": toxic_f1,
        "toxic_precision": toxic_prec,
        "toxic_recall": toxic_rec,
        "confusion_matrix": cm
    }


def plot_confusion_matrix(y_true, y_pred, model_name="Mô hình", save_path=None):
    """
    Vẽ và lưu biểu đồ nhiệt (Heatmap) của ma trận nhầm lẫn.
    
    Tham số:
        y_true (array-like): Nhãn thật.
        y_pred (array-like): Nhãn dự đoán.
        model_name (str): Tên mô hình để đặt tiêu đề cho biểu đồ.
        save_path (str, optional): Đường dẫn file ảnh để lưu (ví dụ: 'results/cm_svm.png').
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Thiết lập kích thước khung hình
    plt.figure(figsize=(6, 5))
    
    # Vẽ biểu đồ nhiệt bằng Seaborn
    sns.heatmap(
        cm,
        annot=True,                 # Hiển thị số lượng mẫu trong từng ô
        fmt="d",                    # Định dạng số nguyên
        cmap="Blues",               # Bảng màu xanh dương
        cbar=True,                  # Hiển thị thanh đo màu sắc
        xticklabels=["Non-toxic", "Toxic"],  # Nhãn dự đoán (trục hoành)
        yticklabels=["Non-toxic", "Toxic"]   # Nhãn thực tế (trục tung)
    )
    
    plt.title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Dự đoán của mô hình (Predicted)", fontsize=11, labelpad=8)
    plt.ylabel("Nhãn thực tế (Actual)", fontsize=11, labelpad=8)
    plt.tight_layout()

    # Nếu có truyền đường dẫn lưu file thì lưu ra ảnh chất lượng cao 300 DPI
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    plt.close()


# ==============================================================================
# PHẦN CHẠY THỬ NGHIỆM (DEMO KHI CHẠY TRỰC TIẾP FILE NÀY)
# Lệnh chạy: python src/evaluation/metrics.py
# ==============================================================================
if __name__ == "__main__":
    # Giả lập 10 mẫu nhãn thật và nhãn dự đoán
    # 0: Non-toxic, 1: Toxic
    y_true_demo = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    y_pred_demo = [0, 0, 0, 0, 1, 0, 1, 1, 1, 0]  # 1 mẫu lành tính bị đoán nhầm, 1 mẫu độc hại bị bỏ sót

    print("DEMO ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH VỚI DỮ LIỆU GIẢ LẬP:")
    res = evaluate_model(y_true_demo, y_pred_demo, model_name="Demo Model")
    plot_confusion_matrix(y_true_demo, y_pred_demo, model_name="Demo Model", save_path="demo_cm.png")
    print("\n[OK] Đã vẽ và lưu thử biểu đồ tại: demo_cm.png")
    
    # Xóa file demo sau khi test
    import os
    if os.path.exists("demo_cm.png"):
        os.remove("demo_cm.png")

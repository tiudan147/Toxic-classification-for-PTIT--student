"""
================================================================================
MODULE 4: MÔ HÌNH HỌC MÁY SUPPORT VECTOR MACHINE (LINEAR SVM)
File: src/models/svm_model.py
================================================================================
MỤC ĐÍCH:
    Cung cấp hàm khởi tạo mô hình Linear Support Vector Classifier (LinearSVC)
    từ thư viện scikit-learn.

NGUYÊN LÝ HOẠT ĐỘNG (Dành cho người mới học):
    1. Cơ chế của SVM:
       SVM tìm kiếm một "siêu phẳng" (hyperplane) trong không gian nhiều chiều để
       phân tách 2 lớp (Non-toxic và Toxic) sao cho khoảng cách an toàn (Lề - Margin)
       từ siêu phẳng đến các điểm dữ liệu gần nhất của 2 lớp là LỚN NHẤT.
       Lề càng rộng thì khả năng tổng quát hóa trên dữ liệu mới càng tốt.

    2. Tại sao chọn "LinearSVC" thay vì "SVC(kernel='linear')"?
       - Khi dùng vector đặc trưng TF-IDF, số chiều rất lớn (hàng nghìn từ vựng).
         Trong không gian nhiều chiều này, dữ liệu văn bản gần như phân tách tuyến tính.
       - LinearSVC được cài đặt dựa trên thư viện LibLinear (chuyên biệt cho bài toán tuyến tính),
         cho tốc độ huấn luyện nhanh hơn gấp 10-50 lần so với SVC thông thường và tiết kiệm RAM.

    3. Ý nghĩa tham số C (Hệ số điều hòa - Regularization):
       - C điều chỉnh mức độ đánh đổi giữa việc giữ lề rộng và việc giảm thiểu lỗi phân loại.
       - C lớn: Mô hình phạt rất nặng các điểm phân loại sai trên tập Train -> Dễ bị Overfitting.
       - C nhỏ: Mô hình chấp nhận bỏ qua một số lỗi nhỏ để lề được phẳng và rộng hơn -> Tổng quát tốt hơn.
       - Giá trị C=1.0 thường là điểm cân bằng lý tưởng.

    4. Ý nghĩa tham số class_weight='balanced':
       - Trong thực tế, số bình luận Non-toxic (~70%) luôn nhiều hơn bình luận Toxic (~30%).
       - Nếu không can thiệp, mô hình sẽ có xu hướng "lười biếng" đoán mọi thứ là Non-toxic
         để dễ dàng đạt độ chính xác (Accuracy) 70%.
       - `class_weight='balanced'` tự động nhân trọng số phạt cao hơn khi mô hình đoán sai
         lớp thiểu số (Toxic), giúp mô hình tập trung bắt đúng các câu Toxic (tăng mạnh Recall và F1).
================================================================================
"""

from sklearn.svm import LinearSVC


def get_svm_model(C=1.0, class_weight="balanced"):
    """
    Khởi tạo mô hình LinearSVC tối ưu cho phân loại văn bản TF-IDF.
    
    Tham số:
        C (float = 1.0): Tham số nghịch đảo của điều hòa (Regularization).
        class_weight (str = "balanced"): Cân bằng trọng số giữa các lớp dữ liệu lệch.
        
    Trả về:
        LinearSVC: Đối tượng mô hình SVM của scikit-learn.
    """
    model = LinearSVC(
        C=C,
        class_weight=class_weight,
        random_state=42,    # Giữ hạt giống ngẫu nhiên cố định để kết quả luôn giống nhau
        max_iter=3000       # Số bước lặp tối đa để đảm bảo mô hình hội tụ tốt
    )
    return model


# ==============================================================================
# PHẦN CHẠY THỬ NGHIỆM (DEMO KHI CHẠY TRỰC TIẾP FILE NÀY)
# Lệnh chạy: python src/models/svm_model.py
# ==============================================================================
if __name__ == "__main__":
    import numpy as np

    print("==================================================")
    print("DEMO KHỞI TẠO VÀ CHẠY THỬ MÔ HÌNH LINEAR SVM")
    print("==================================================")

    # Dữ liệu giả lập 4 mẫu (2 mẫu lành tính nhãn 0, 2 mẫu độc hại nhãn 1)
    X_dummy = np.array([
        [0.9, 0.1, 0.0],  # Lành tính
        [0.8, 0.2, 0.1],  # Lành tính
        [0.1, 0.7, 0.9],  # Độc hại
        [0.0, 0.8, 0.7],  # Độc hại
    ])
    y_dummy = np.array([0, 0, 1, 1])

    # Khởi tạo mô hình SVM
    svm = get_svm_model(C=1.0, class_weight="balanced")
    print(f"Mô hình được tạo: {svm}")

    # Huấn luyện mô hình
    svm.fit(X_dummy, y_dummy)
    print("- Đã huấn luyện mô hình SVM thành công.")

    # Thử nghiệm dự đoán
    X_test_sample = np.array([[0.05, 0.85, 0.95]])
    pred = svm.predict(X_test_sample)
    decision = svm.decision_function(X_test_sample)

    print(f"- Mẫu kiểm tra     : {X_test_sample[0]}")
    print(f"- Điểm số khoảng cách (Decision score): {decision[0]:.4f} (>0 là Toxic, <0 là Non-toxic)")
    print(f"- Dự đoán nhãn     : {pred[0]} ({'Toxic' if pred[0] == 1 else 'Non-toxic'})")

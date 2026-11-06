"""
================================================================================
MODULE 3: MÔ HÌNH NAIVE BAYES (MULTINOMIAL NAIVE BAYES)
File: src/models/naive_bayes_model.py
================================================================================
MỤC ĐÍCH:
    Cung cấp hàm khởi tạo mô hình Multinomial Naive Bayes chuẩn từ scikit-learn.

NGUYÊN LÝ HOẠT ĐỘNG (Dành cho người mới học):
    1. Định lý Bayes:
       Tính xác suất một câu thuộc lớp C (0: Non-toxic, 1: Toxic) dựa trên xác suất
       xuất hiện của từng từ có trong câu đó:
       P(Lớp | Các_từ) ~ P(Lớp) * P(Từ_1 | Lớp) * P(Từ_2 | Lớp) * ...
       
    2. Tại sao gọi là "Naive" (Ngây thơ)?
       Vì mô hình giả định rằng sự xuất hiện của các từ là hoàn toàn độc lập với nhau.
       Mặc dù trong thực tế các từ có sự liên kết, nhưng giả định đơn giản này lại
       khiến mô hình chạy cực kỳ nhanh, tốn rất ít bộ nhớ và đạt hiệu quả rất cao
       với dữ liệu văn bản.

    3. Tại sao chọn "MultinomialNB"?
       Multinomial Naive Bayes được thiết kế riêng cho các bài toán phân loại văn bản
       với đầu vào là số lần đếm từ hoặc trọng số tần suất TF-IDF.

    4. Tham số làm mịn Laplace (alpha):
       - Nếu gặp một từ mới lạ chưa từng xuất hiện trong tập Train, xác suất của từ
         đó bằng 0. Khi nhân các xác suất lại với nhau, cả tích sẽ bằng 0!
       - Tham số `alpha` giúp cộng thêm một lượng nhỏ vào xác suất (Smoothing) để
         tránh lỗi xác suất bằng 0 này.
       - Thường chọn: alpha = 1.0 (Laplace smoothing) hoặc alpha = 0.5 (Lidstone smoothing).
================================================================================
"""

from sklearn.naive_bayes import MultinomialNB


def get_naive_bayes_model(alpha=0.5):
    """
    Khởi tạo và trả về mô hình Multinomial Naive Bayes.
    
    Tham số:
        alpha (float = 0.5): Hệ số làm mịn (Smoothing parameter).
                             Giá trị càng nhỏ thì mô hình càng phụ thuộc vào dữ liệu train,
                             giá trị lớn giúp mô hình mượt mà hơn và tránh overfit.
    
    Trả về:
        MultinomialNB: Đối tượng mô hình Naive Bayes của scikit-learn.
    """
    model = MultinomialNB(alpha=alpha)
    return model


# ==============================================================================
# PHẦN CHẠY THỬ NGHIỆM (DEMO KHI CHẠY TRỰC TIẾP FILE NÀY)
# Lệnh chạy: python src/models/naive_bayes_model.py
# ==============================================================================
if __name__ == "__main__":
    import numpy as np

    print("==================================================")
    print("DEMO KHỞI TẠO VÀ CHẠY THỬ MÔ HÌNH NAIVE BAYES")
    print("==================================================")

    # Giả lập ma trận đặc trưng X (4 câu, mỗi câu có 3 đặc trưng)
    # Câu 1 và 2 là Non-toxic (nhãn 0); Câu 3 và 4 là Toxic (nhãn 1)
    X_dummy = np.array([
        [0.8, 0.1, 0.0],  # Câu 1 (lành tính)
        [0.7, 0.2, 0.1],  # Câu 2 (lành tính)
        [0.1, 0.8, 0.9],  # Câu 3 (độc hại)
        [0.0, 0.7, 0.8],  # Câu 4 (độc hại)
    ])
    y_dummy = np.array([0, 0, 1, 1])

    # Khởi tạo mô hình
    nb = get_naive_bayes_model(alpha=0.5)
    print(f"Mô hình được tạo: {nb}")

    # Huấn luyện mô hình
    nb.fit(X_dummy, y_dummy)
    print("- Đã fit mô hình thành công trên dữ liệu mẫu.")

    # Dự đoán thử một mẫu mới
    X_new = np.array([[0.0, 0.9, 0.8]])
    prediction = nb.predict(X_new)
    probability = nb.predict_proba(X_new)

    print(f"- Mẫu kiểm tra mới  : {X_new[0]}")
    print(f"- Dự đoán nhãn      : {prediction[0]} (0: Non-toxic, 1: Toxic)")
    print(f"- Xác suất từng lớp : Non-toxic: {probability[0][0]*100:.2f}% | Toxic: {probability[0][1]*100:.2f}%")

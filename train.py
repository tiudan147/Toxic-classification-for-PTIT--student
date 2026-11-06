"""
================================================================================
CHƯƠNG TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ TOÀN DIỆN MÔ HÌNH (TRAINING PIPELINE)
File: train.py
================================================================================
MỤC ĐÍCH:
    Kịch bản chính thực hiện toàn bộ luồng học máy (Machine Learning Pipeline):
    1. Đọc dữ liệu Train, Validation, Test từ thư mục `dataset/processed_data/`.
    2. Trích chọn đặc trưng:
       - TF-IDF Vectorizer (5000 chiều) -> Dùng cho Naive Bayes & Linear SVM.
       - Word2Vec Vectorizer (150 chiều) -> Dùng cho Linear SVM (Word2Vec).
    3. Huấn luyện và đánh giá:
       - Mô hình 1: Multinomial Naive Bayes (trên TF-IDF).
       - Mô hình 2: Linear SVM (trên TF-IDF).
       - Mô hình 3: Linear SVM (trên Word2Vec).
    4. So sánh trực quan hiệu năng của các mô hình trên cả Validation và Test.
    5. Lưu toàn bộ các thành phần vào `saved_models/`:
       - `tfidf_vectorizer.pkl`   : Bộ trích xuất đặc trưng TF-IDF
       - `w2v_vectorizer.pkl`     : Bộ trích xuất đặc trưng Word2Vec
       - `naive_bayes_model.pkl`  : Mô hình Naive Bayes (TF-IDF)
       - `best_svm_model.pkl`     : Mô hình Linear SVM (TF-IDF)
       - `svm_w2v_model.pkl`      : Mô hình Linear SVM (Word2Vec)
================================================================================
"""

import os
import sys
import pandas as pd
import joblib

# Đảm bảo terminal Windows in đúng tiếng Việt UTF-8 không bị lỗi ký tự
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm thư mục BTL_1 vào đường dẫn hệ thống để gọi các module trong src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Nạp các hàm từ các module chức năng đã xây dựng trong src/
from src.features import get_tfidf_vectorizer, Word2VecVectorizer
from src.models import get_naive_bayes_model, get_svm_model
from src.evaluation import evaluate_model, plot_confusion_matrix


def main():
    print("=" * 65)
    print("   BÀI TẬP LỚN 1: HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI BÌNH LUẬN ĐỘC HẠI")
    print("=" * 65)

    # --------------------------------------------------------------------------
    # BƯỚC 1: ĐỌC DỮ LIỆU ĐÃ QUA TIỀN XỬ LÝ (Train, Validation, Test)
    # --------------------------------------------------------------------------
    train_path = "dataset/processed_data/train.csv"
    val_path = "dataset/processed_data/val.csv"
    test_path = "dataset/processed_data/test.csv"

    print("\n[BƯỚC 1] Đang nạp dữ liệu từ thư mục dataset/processed_data/...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    print(f"- Tập Huấn luyện (Train) : {len(df_train)} dòng")
    print(f"- Tập Kiểm tra (Val)     : {len(df_val)} dòng (chọn mô hình & chỉnh tham số)")
    print(f"- Tập Đánh giá (Test)    : {len(df_test)} dòng (đánh giá hiệu năng cuối cùng)")

    # Tách riêng cột văn bản (X) và cột nhãn mục tiêu (y)
    X_train_text, y_train = df_train["text"], df_train["label"]
    X_val_text, y_val = df_val["text"], df_val["label"]
    X_test_text, y_test = df_test["text"], df_test["label"]

    # --------------------------------------------------------------------------
    # BƯỚC 2: TRÍCH CHỌN ĐẶC TRƯNG VĂN BẢN (TF-IDF & WORD2VEC)
    # --------------------------------------------------------------------------
    # 2.1. Trích chọn đặc trưng TF-IDF (5000 chiều)
    print("\n[BƯỚC 2.1] Trích xuất đặc trưng bằng TF-IDF Vectorizer (5000 chiều)...")
    tfidf_vectorizer = get_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2))

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_val_tfidf = tfidf_vectorizer.transform(X_val_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    print(f"- Kích thước ma trận Train (TF-IDF): {X_train_tfidf.shape}")
    print(f"- Kích thước ma trận Val   (TF-IDF): {X_val_tfidf.shape}")
    print(f"- Kích thước ma trận Test  (TF-IDF): {X_test_tfidf.shape}")

    # 2.2. Huấn luyện bộ nhúng từ Word2Vec (150 chiều)
    print("\n[BƯỚC 2.2] Huấn luyện bộ trích chọn đặc trưng Word2Vec (150 chiều)...")
    w2v_vectorizer = Word2VecVectorizer(vector_size=150, window=5, min_count=2, sg=1, epochs=10)
    w2v_vectorizer.fit(X_train_text)

    # Chuyển đổi các tập dữ liệu sang vector Word2Vec
    X_train_w2v = w2v_vectorizer.transform(X_train_text)
    X_val_w2v = w2v_vectorizer.transform(X_val_text)
    X_test_w2v = w2v_vectorizer.transform(X_test_text)

    print(f"- Kích thước ma trận Train (Word2Vec): {X_train_w2v.shape}")
    print(f"- Kích thước ma trận Val   (Word2Vec): {X_val_w2v.shape}")
    print(f"- Kích thước ma trận Test  (Word2Vec): {X_test_w2v.shape}")

    # Tạo thư mục lưu kết quả và mô hình (nếu chưa có)
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # --------------------------------------------------------------------------
    # BƯỚC 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH 1 - NAIVE BAYES (TF-IDF)
    # --------------------------------------------------------------------------
    print("\n[BƯỚC 3] Huấn luyện Mô hình 1: Multinomial Naive Bayes (trên TF-IDF)...")
    nb_model = get_naive_bayes_model(alpha=0.5)
    nb_model.fit(X_train_tfidf, y_train)

    y_val_pred_nb = nb_model.predict(X_val_tfidf)
    nb_val_metrics = evaluate_model(y_val, y_val_pred_nb, model_name="Naive Bayes (Val)")

    y_test_pred_nb = nb_model.predict(X_test_tfidf)
    nb_test_metrics = evaluate_model(y_test, y_test_pred_nb, model_name="Naive Bayes (Test)")
    plot_confusion_matrix(y_test, y_test_pred_nb, model_name="Naive Bayes (Test)", save_path="results/cm_naive_bayes.png")

    # --------------------------------------------------------------------------
    # BƯỚC 4: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH 2 - LINEAR SVM (TF-IDF)
    # --------------------------------------------------------------------------
    print("\n[BƯỚC 4] Huấn luyện Mô hình 2: Linear SVM (trên TF-IDF)...")
    svm_tfidf_model = get_svm_model(C=1.0, class_weight="balanced")
    svm_tfidf_model.fit(X_train_tfidf, y_train)

    y_val_pred_svm = svm_tfidf_model.predict(X_val_tfidf)
    svm_val_metrics = evaluate_model(y_val, y_val_pred_svm, model_name="Linear SVM TF-IDF (Val)")

    y_test_pred_svm = svm_tfidf_model.predict(X_test_tfidf)
    svm_test_metrics = evaluate_model(y_test, y_test_pred_svm, model_name="Linear SVM TF-IDF (Test)")
    plot_confusion_matrix(y_test, y_test_pred_svm, model_name="Linear SVM (Test)", save_path="results/cm_svm.png")

    # --------------------------------------------------------------------------
    # BƯỚC 5: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH 3 - LINEAR SVM (WORD2VEC)
    # --------------------------------------------------------------------------
    print("\n[BƯỚC 5] Huấn luyện Mô hình 3: Linear SVM (trên Word2Vec 150 chiều)...")
    svm_w2v_model = get_svm_model(C=1.0, class_weight="balanced")
    svm_w2v_model.fit(X_train_w2v, y_train)

    y_val_pred_w2v = svm_w2v_model.predict(X_val_w2v)
    w2v_val_metrics = evaluate_model(y_val, y_val_pred_w2v, model_name="Linear SVM W2V (Val)")

    y_test_pred_w2v = svm_w2v_model.predict(X_test_w2v)
    w2v_test_metrics = evaluate_model(y_test, y_test_pred_w2v, model_name="Linear SVM W2V (Test)")

    # --------------------------------------------------------------------------
    # BƯỚC 6: TỔNG HỢP VÀ SO SÁNH HIỆU NĂNG TẤT CẢ CÁC MÔ HÌNH
    # --------------------------------------------------------------------------
    print("\n[BƯỚC 6] BẢNG TỔNG HỢP VÀ SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH:")
    summary_data = [
        {
            "Mô hình": "Naive Bayes (TF-IDF)",
            "Tập": "Validation",
            "Accuracy (%)": round(nb_val_metrics["accuracy"] * 100, 2),
            "Macro F1": round(nb_val_metrics["macro_f1"], 4),
            "Toxic F1": round(nb_val_metrics["toxic_f1"], 4),
            "Toxic Recall": round(nb_val_metrics["toxic_recall"], 4)
        },
        {
            "Mô hình": "Naive Bayes (TF-IDF)",
            "Tập": "Test (Cuối)",
            "Accuracy (%)": round(nb_test_metrics["accuracy"] * 100, 2),
            "Macro F1": round(nb_test_metrics["macro_f1"], 4),
            "Toxic F1": round(nb_test_metrics["toxic_f1"], 4),
            "Toxic Recall": round(nb_test_metrics["toxic_recall"], 4)
        },
        {
            "Mô hình": "Linear SVM (TF-IDF)",
            "Tập": "Validation",
            "Accuracy (%)": round(svm_val_metrics["accuracy"] * 100, 2),
            "Macro F1": round(svm_val_metrics["macro_f1"], 4),
            "Toxic F1": round(svm_val_metrics["toxic_f1"], 4),
            "Toxic Recall": round(svm_val_metrics["toxic_recall"], 4)
        },
        {
            "Mô hình": "Linear SVM (TF-IDF)",
            "Tập": "Test (Cuối)",
            "Accuracy (%)": round(svm_test_metrics["accuracy"] * 100, 2),
            "Macro F1": round(svm_test_metrics["macro_f1"], 4),
            "Toxic F1": round(svm_test_metrics["toxic_f1"], 4),
            "Toxic Recall": round(svm_test_metrics["toxic_recall"], 4)
        },
        {
            "Mô hình": "Linear SVM (Word2Vec)",
            "Tập": "Validation",
            "Accuracy (%)": round(w2v_val_metrics["accuracy"] * 100, 2),
            "Macro F1": round(w2v_val_metrics["macro_f1"], 4),
            "Toxic F1": round(w2v_val_metrics["toxic_f1"], 4),
            "Toxic Recall": round(w2v_val_metrics["toxic_recall"], 4)
        },
        {
            "Mô hình": "Linear SVM (Word2Vec)",
            "Tập": "Test (Cuối)",
            "Accuracy (%)": round(w2v_test_metrics["accuracy"] * 100, 2),
            "Macro F1": round(w2v_test_metrics["macro_f1"], 4),
            "Toxic F1": round(w2v_test_metrics["toxic_f1"], 4),
            "Toxic Recall": round(w2v_test_metrics["toxic_recall"], 4)
        }
    ]
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # --------------------------------------------------------------------------
    # BƯỚC 7: LƯU TẤT CẢ MÔ HÌNH VÀ BỘ VECTORIZER VÀO saved_models/
    # --------------------------------------------------------------------------
    print("\n[BƯỚC 7] Đang lưu các mô hình và bộ vectorizer vào saved_models/...")
    
    # 1. Nhóm TF-IDF
    tfidf_file = "saved_models/tfidf_vectorizer.pkl"
    svm_tfidf_file = "saved_models/best_svm_model.pkl"
    nb_file = "saved_models/naive_bayes_model.pkl"

    joblib.dump(tfidf_vectorizer, tfidf_file)
    joblib.dump(svm_tfidf_model, svm_tfidf_file)
    joblib.dump(nb_model, nb_file)

    print(f"- [ĐÃ LƯU] TF-IDF Vectorizer (5000 chiều) -> {tfidf_file}")
    print(f"- [ĐÃ LƯU] Mô hình Linear SVM (TF-IDF)     -> {svm_tfidf_file}")
    print(f"- [ĐÃ LƯU] Mô hình Naive Bayes (TF-IDF)    -> {nb_file}")

    # 2. Nhóm Word2Vec
    w2v_file = "saved_models/w2v_vectorizer.pkl"
    svm_w2v_file = "saved_models/svm_w2v_model.pkl"

    joblib.dump(w2v_vectorizer, w2v_file)
    joblib.dump(svm_w2v_model, svm_w2v_file)

    print(f"- [ĐÃ LƯU] Word2Vec Vectorizer (150 chiều) -> {w2v_file}")
    print(f"- [ĐÃ LƯU] Mô hình Linear SVM (Word2Vec)   -> {svm_w2v_file}")

    print("- [ĐÃ LƯU] Biểu đồ ma trận nhầm lẫn -> results/cm_naive_bayes.png, results/cm_svm.png")
    print("\n" + "=" * 65)
    print("   [HOÀN TẤT] Huấn luyện và lưu toàn bộ mô hình thành công!")
    print("=" * 65)


if __name__ == "__main__":
    main()

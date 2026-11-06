"""
================================================================================
CHƯƠNG TRÌNH DỰ ĐOÁN VÀ SO SÁNH CÁC CẶP MÔ HÌNH (INFERENCE & COMPARISON)
File: predict.py
================================================================================
MỤC ĐÍCH:
    1. Kiểm thử đồng thời tất cả các cặp (Mô hình + Bộ trích chọn đặc trưng)
       đã được huấn luyện và lưu trong `saved_models/` trên dữ liệu thực tế:
       - Cặp 1: Naive Bayes  + TF-IDF Vectorizer
       - Cặp 2: Linear SVM   + TF-IDF Vectorizer
       - Cặp 3: Linear SVM   + Word2Vec Vectorizer
    2. Xuất bảng so sánh tổng hợp hiệu năng giữa các cặp trên dữ liệu cào FB.
    3. Vẽ và lưu ma trận nhầm lẫn cho từng cặp vào thư mục `results/`.
    4. Xuất file kết quả chi tiết `results/fb_test_predictions.csv`.
    5. Cung cấp hàm `predict_single_text()` cho phép gõ một câu tự chọn và
       xem kết quả phán đoán của CẢ 3 CẶP MÔ HÌNH cùng lúc để so sánh!
================================================================================
"""

import os
import sys
import warnings
import pandas as pd
import joblib

# Bỏ qua các cảnh báo không ảnh hưởng kết quả (như phiên bản thư viện nhỏ)
warnings.filterwarnings("ignore")

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

from src.preprocessing_data import clean_text
from src.evaluation import evaluate_model, plot_confusion_matrix


# ==============================================================================
# ĐỊNH NGHĨA DANH SÁCH CÁC CẶP MÔ HÌNH VÀ BỘ TRÍCH CHỌN ĐẶC TRƯNG HỢP LỆ
# ==============================================================================
MODEL_PAIRS = [
    {
        "id": "nb_tfidf",
        "name": "Naive Bayes (TF-IDF)",
        "model_file": "saved_models/naive_bayes_model.pkl",
        "vec_file": "saved_models/tfidf_vectorizer.pkl",
        "cm_file": "results/cm_nb_tfidf.png"
    },
    {
        "id": "svm_tfidf",
        "name": "Linear SVM (TF-IDF)",
        "model_file": "saved_models/best_svm_model.pkl",
        "vec_file": "saved_models/tfidf_vectorizer.pkl",
        "cm_file": "results/cm_svm_tfidf.png"
    },
    {
        "id": "svm_w2v",
        "name": "Linear SVM (Word2Vec)",
        "model_file": "saved_models/svm_w2v_model.pkl",
        "vec_file": "saved_models/w2v_vectorizer.pkl",
        "cm_file": "results/cm_svm_w2v.png"
    }
]


# ==============================================================================
# HÀM 1: DỰ ĐOÁN 1 CÂU TỰ CHỌN VỚI TẤT CẢ CÁC CẶP MÔ HÌNH (ĐỂ SO SÁNH TRỰC QUAN)
# ==============================================================================
def predict_single_text(text: str):
    """
    Nhận vào một câu văn bản bất kỳ, chạy qua cả 3 cặp mô hình và in kết quả so sánh.
    
    Tham số:
        text (str): Câu bình luận cần kiểm tra.
        
    Trả về:
        dict: Kết quả phân loại của từng cặp mô hình.
    """
    cleaned = clean_text(text)
    results = {
        "text_raw": text,
        "text_clean": cleaned,
        "predictions": {}
    }

    # Nạp sẵn các vectorizer để không phải nạp lại nhiều lần
    loaded_vecs = {}
    if os.path.exists("saved_models/tfidf_vectorizer.pkl"):
        loaded_vecs["tfidf"] = joblib.load("saved_models/tfidf_vectorizer.pkl")
    if os.path.exists("saved_models/w2v_vectorizer.pkl"):
        loaded_vecs["w2v"] = joblib.load("saved_models/w2v_vectorizer.pkl")

    for pair in MODEL_PAIRS:
        m_file = pair["model_file"]
        if not os.path.exists(m_file):
            continue

        model = joblib.load(m_file)
        vec_type = "tfidf" if "tfidf" in pair["vec_file"] else "w2v"
        vectorizer = loaded_vecs.get(vec_type)

        if vectorizer is None:
            continue

        feat = vectorizer.transform([cleaned])
        pred = int(model.predict(feat)[0])
        label_str = "Toxic (Độc hại)" if pred == 1 else "Non-toxic (Lành tính)"

        results["predictions"][pair["name"]] = {
            "label": pred,
            "label_name": label_str
        }

    return results


# ==============================================================================
# HÀM 2: KIỂM THỬ VÀ SO SÁNH TẤT CẢ CÁC CẶP MÔ HÌNH TRÊN TẬP DỮ LIỆU FILE CSV
# ==============================================================================
def test_all_pairs(csv_path="dataset/fb_test.csv"):
    """
    Đọc dữ liệu kiểm thử thực tế, làm sạch một lần duy nhất,
    sau đó lần lượt đánh giá từng cặp mô hình và in bảng so sánh toàn diện.
    """
    if not os.path.exists(csv_path):
        print(f"\n[LỖI] Không tìm thấy file dữ liệu kiểm thử tại: {csv_path}\n")
        return

    # Kiểm tra sự tồn tại của các file mô hình
    available_pairs = []
    for pair in MODEL_PAIRS:
        if os.path.exists(pair["model_file"]) and os.path.exists(pair["vec_file"]):
            available_pairs.append(pair)

    if not available_pairs:
        print("\n[LỖI] Chưa tìm thấy mô hình trong saved_models/! Hãy chạy `python train.py` trước.\n")
        return

    print("=" * 70)
    print(f"   KIỂM THỬ VÀ SO SÁNH CÁC CẶP MÔ HÌNH TRÊN TẬP DỮ LIỆU: {csv_path}")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # BƯỚC 1: Đọc dữ liệu từ file CSV
    # --------------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    c0_count = (df["label"] == 0).sum()
    c1_count = (df["label"] == 1).sum()

    print(f"\n[BƯỚC 1] Đã nạp {total_samples} dòng dữ liệu.")
    print("         Phân bố nhãn thực tế:")
    print(f"         - Non-toxic (0 - Lành tính) : {c0_count} mẫu ({c0_count/total_samples*100:.1f}%)")
    print(f"         - Toxic (1 - Độc hại)       : {c1_count} mẫu ({c1_count/total_samples*100:.1f}%)")

    # --------------------------------------------------------------------------
    # BƯỚC 2: Tiền xử lý văn bản một lần duy nhất (Tiết kiệm thời gian)
    # --------------------------------------------------------------------------
    print("\n[BƯỚC 2] Đang tiền xử lý toàn bộ văn bản (clean_text)...")
    cleaned_texts = [clean_text(t) for t in df["text"]]
    df["text_clean"] = cleaned_texts
    y_true = df["label"]

    # --------------------------------------------------------------------------
    # BƯỚC 3: Trích xuất đặc trưng trước cho cả 2 loại TF-IDF và Word2Vec
    # --------------------------------------------------------------------------
    print("\n[BƯỚC 3] Nạp vectorizer và trích xuất đặc trưng cho dữ liệu kiểm thử...")
    features_cache = {}

    if os.path.exists("saved_models/tfidf_vectorizer.pkl"):
        tfidf_vec = joblib.load("saved_models/tfidf_vectorizer.pkl")
        features_cache["tfidf"] = tfidf_vec.transform(cleaned_texts)
        print(f"- Đã biến đổi ma trận TF-IDF   : {features_cache['tfidf'].shape}")

    if os.path.exists("saved_models/w2v_vectorizer.pkl"):
        w2v_vec = joblib.load("saved_models/w2v_vectorizer.pkl")
        features_cache["w2v"] = w2v_vec.transform(cleaned_texts)
        print(f"- Đã biến đổi ma trận Word2Vec : {features_cache['w2v'].shape}")

    # Tạo thư mục results nếu chưa có
    os.makedirs("results", exist_ok=True)

    # --------------------------------------------------------------------------
    # BƯỚC 4: Dự đoán và đánh giá chi tiết từng cặp mô hình
    # --------------------------------------------------------------------------
    print("\n[BƯỚC 4] Đang đánh giá chi tiết từng cặp mô hình...")
    comparison_summary = []

    for pair in available_pairs:
        pair_name = pair["name"]
        model = joblib.load(pair["model_file"])
        vec_type = "tfidf" if "tfidf" in pair["vec_file"] else "w2v"
        X_test = features_cache[vec_type]

        # Dự đoán
        y_pred = model.predict(X_test)
        df[f"pred_{pair['id']}"] = y_pred

        # Đánh giá bằng hàm evaluate_model
        metrics = evaluate_model(y_true, y_pred, model_name=pair_name)

        # Lưu biểu đồ ma trận nhầm lẫn
        plot_confusion_matrix(y_true, y_pred, model_name=pair_name, save_path=pair["cm_file"])

        comparison_summary.append({
            "Mô hình & Đặc trưng": pair_name,
            "Accuracy (%)": round(metrics["accuracy"] * 100, 2),
            "Macro F1": round(metrics["macro_f1"], 4),
            "Toxic F1": round(metrics["toxic_f1"], 4),
            "Toxic Precision": round(metrics["toxic_precision"], 4),
            "Toxic Recall": round(metrics["toxic_recall"], 4)
        })

    # --------------------------------------------------------------------------
    # BƯỚC 5: IN BẢNG TỔNG HỢP VÀ SO SÁNH TẤT CẢ CÁC CẶP TRÊN FB_TEST
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("   BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG CÁC CẶP TRÊN FB_TEST (605 mẫu):")
    print("=" * 70)
    df_compare = pd.DataFrame(comparison_summary)
    print(df_compare.to_string(index=False))

    # --------------------------------------------------------------------------
    # BƯỚC 6: Lưu kết quả dự đoán chi tiết của tất cả các mô hình ra file CSV
    # --------------------------------------------------------------------------
    output_csv = "results/fb_test_predictions.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[XONG] Đã lưu chi tiết dự đoán của tất cả mô hình vào: {output_csv}")
    print(f"[XONG] Đã lưu ảnh ma trận nhầm lẫn tại thư mục: results/")

    # --------------------------------------------------------------------------
    # BƯỚC 7: Minh họa 5 mẫu thực tế so sánh phán đoán giữa các cặp
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("   MINH HỌA 5 CÂU DỰ ĐOÁN THỰC TẾ (SO SÁNH CÁC CẶP MÔ HÌNH):")
    print("=" * 70)
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        real_lbl = "Toxic (1)" if row["label"] == 1 else "Non-toxic (0)"
        print(f"\n[Mẫu #{i+1}] Nhãn thật: {real_lbl}")
        print(f"- Gốc : {row['text'][:90]}...")
        print(f"- Sạch: {row['text_clean'][:90]}...")
        print("  Dự đoán của từng cặp:")
        for pair in available_pairs:
            p_val = row[f"pred_{pair['id']}"]
            p_str = "Toxic (1)" if p_val == 1 else "Non-toxic (0)"
            status = "ĐÚNG" if p_val == row["label"] else "SAI"
            print(f"  + {pair['name']:25}: {p_str:15} [{status}]")

    print("\n" + "=" * 70)
    print("   [HOÀN TẤT] Quá trình kiểm thử và so sánh đã hoàn thành xuất sắc!")
    print("=" * 70)


# ==============================================================================
# ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH
# ==============================================================================
if __name__ == "__main__":
    # 1. Chạy kiểm thử và so sánh đồng thời tất cả các cặp mô hình trên fb_test.csv
    test_all_pairs(csv_path="dataset/fb_test.csv")

    # 2. Thử nghiệm dự đoán 2 câu tự chọn trên CẢ 3 CẶP MÔ HÌNH để so sánh trực quan
    print("\n" + "=" * 70)
    print("   THỬ NGHIỆM DỰ ĐOÁN CÂU TỰ DO TRÊN TẤT CẢ CÁC CẶP MÔ HÌNH:")
    print("=" * 70)
    custom_sentences = [
        "Cảm ơn bạn nhiều nhé, bài viết rất hữu ích!",
        "Thằng chó này mày ăn nói xấc xược cút đi!"
    ]
    for s in custom_sentences:
        res = predict_single_text(s)
        print(f"\nCâu: \"{res['text_raw']}\"")
        for model_name, p_info in res["predictions"].items():
            print(f"  -> {model_name:25}: {p_info['label_name']}")
    print()

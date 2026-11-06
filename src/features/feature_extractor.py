"""
================================================================================
MODULE 2: TRÍCH CHỌN ĐẶC TRƯNG VĂN BẢN (FEATURE EXTRACTION)
File: src/features/feature_extractor.py
================================================================================
MỤC ĐÍCH:
    Các thuật toán học máy (Machine Learning) như Naive Bayes, SVM chỉ có thể
    làm việc với các con số (vector / ma trận), không thể hiểu trực tiếp các
    từ ngữ.
    Module này cung cấp 2 phương pháp kinh điển và phổ biến nhất trong Xử lý
    ngôn ngữ tự nhiên (NLP) để biến đổi câu văn thành vector số:
    
    1. TF-IDF (Term Frequency - Inverse Document Frequency):
       - Phương pháp dựa trên tần suất xuất hiện và độ quý hiếm của từ.
       - Rất mạnh mẽ, tính toán cực nhanh, phù hợp hoàn hảo với Naive Bayes và SVM.
       
    2. Word2Vec (Word Embeddings):
       - Phương pháp dựa trên ngữ cảnh ngữ nghĩa (các từ có nghĩa tương tự nhau
         sẽ nằm gần nhau trong không gian vector nhiều chiều).
       - Vector của câu được tính bằng trung bình cộng (Mean Pooling) vector của các từ.
================================================================================
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


# ==============================================================================
# 1. PHƯƠNG PHÁP 1: TF-IDF VECTORIZER
# ==============================================================================
def get_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2), min_df=1):
    """
    Khởi tạo bộ trích chọn đặc trưng TF-IDF tiêu chuẩn từ thư viện scikit-learn.

    Ý NGHĨA CỦA CÁC THAM SỐ (Dễ dàng tinh chỉnh để nâng cao hiệu năng):
    ------------------------------------------------------------------
    - max_features (int = 5000): 
        Giới hạn từ điển chỉ giữ lại 5000 từ/cụm từ có điểm TF-IDF cao nhất.
        Giúp loại bỏ những từ quá hiếm (nhiễu) và kiểm soát kích thước ma trận.
        
    - ngram_range (tuple = (1, 2)):
        Lấy cả từ đơn (1-gram) và cụm 2 từ liền kề (2-gram / bigram).
        Ví dụ: Trong câu "rất ngu", mô hình sẽ học cả từ "rất", "ngu" và cụm "rất ngu".
        Cụm 2 từ cực kỳ quan trọng trong phân loại toxic (ví dụ: "không thích", "đồ khốn").
        
    - min_df (int = 1):
        Số lần xuất hiện tối thiểu của một từ trong tập dữ liệu để được đưa vào từ điển.
        min_df=1 nghĩa là từ chỉ cần xuất hiện 1 lần trở lên là được giữ lại.
        
    - sublinear_tf (bool = True):
        Dùng công thức 1 + log(tf) thay vì tần số tuyệt đối tf.
        Giúp giảm nhẹ sự áp đảo của các từ lặp lại quá nhiều lần trong cùng 1 câu.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True
    )
    return vectorizer


# ==============================================================================
# 2. PHƯƠNG PHÁP 2: WORD2VEC VECTORIZER (MÔ HÌNH NHÚNG TỪ)
# ==============================================================================
class Word2VecVectorizer:
    """
    Lớp trích xuất đặc trưng bằng mô hình Word2Vec của thư viện Gensim.
    
    CƠ CHẾ HOẠT ĐỘNG:
    1. Huấn luyện mô hình Word2Vec trên tập câu văn để học vector cho từng từ.
    2. Với mỗi câu văn (document):
       - Tìm vector của từng từ trong câu.
       - Lấy trung bình cộng (np.mean) của các vector từ đó để tạo thành 1 vector
         duy nhất đại diện cho toàn bộ câu.
       - Nếu câu không có từ nào có trong từ điển, gán bằng vector toàn số 0.
    """
    def __init__(self, vector_size=150, window=5, min_count=2, sg=1, epochs=10):
        """
        Ý NGHĨA CÁC THAM SỐ WORD2VEC:
        -----------------------------
        - vector_size (150): Số chiều của mỗi vector từ (mỗi từ biểu diễn bằng 150 số thực).
        - window (5): Khoảng cách ngữ cảnh (xét 5 từ phía trước và 5 từ phía sau).
        - min_count (2): Bỏ qua những từ xuất hiện ít hơn 2 lần trong toàn bộ kho ngữ liệu.
        - sg (1 = Skip-gram):
            + sg=0: CBOW (nhanh hơn, tốt cho từ xuất hiện nhiều).
            + sg=1: Skip-gram (chính xác hơn cho các tập dữ liệu nhỏ và từ hiếm).
        - epochs (10): Số vòng lặp huấn luyện Word2Vec.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.model = None

    def fit(self, texts, y=None):
        """
        Huấn luyện mô hình Word2Vec trên danh sách câu văn đầu vào.
        """
        # Tách mỗi câu văn bản thành danh sách các từ (token list)
        sentences = [doc.split() for doc in texts if isinstance(doc, str)]
        
        # Huấn luyện mô hình Gensim Word2Vec
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            seed=42
        )
        return self

    def transform(self, texts):
        """
        Biến đổi danh sách câu thành ma trận 2 chiều kích thước (Số_câu, vector_size).
        """
        wv = self.model.wv  # Kho từ vựng và vector đã học
        vectors = []

        for doc in texts:
            words = doc.split() if isinstance(doc, str) else []
            # Lấy vector của các từ hợp lệ (có xuất hiện trong từ điển mô hình)
            valid_word_vectors = [wv[w] for w in words if w in wv]

            if valid_word_vectors:
                # Tính trung bình cộng vector của các từ trong câu
                doc_vector = np.mean(valid_word_vectors, axis=0)
                vectors.append(doc_vector)
            else:
                # Nếu không có từ nào hợp lệ, trả về vector toàn số 0
                vectors.append(np.zeros(self.vector_size, dtype=np.float32))

        return np.array(vectors, dtype=np.float32)

    def fit_transform(self, texts, y=None):
        """
        Vừa huấn luyện mô hình vừa chuyển đổi văn bản sang vector trong 1 bước.
        """
        self.fit(texts, y)
        return self.transform(texts)


# ==============================================================================
# PHẦN CHẠY THỬ NGHIỆM (DEMO KHI CHẠY TRỰC TIẾP FILE NÀY)
# Lệnh chạy: python src/features/feature_extractor.py
# ==============================================================================
if __name__ == "__main__":
    sample_texts = [
        "mày ngu_ngốc quá đấy",
        "hôm_nay thời_tiết rất đẹp và dễ_chịu",
        "đồ ngu lừa_đảo cút đi"
    ]

    print("==================================================")
    print("DEMO TRÍCH XUẤT ĐẶC TRƯNG VĂN BẢN (TF-IDF & WORD2VEC)")
    print("==================================================")

    # 1. Thử nghiệm TF-IDF
    print("\n--- 1. KẾT QUẢ VỚI TF-IDF ---")
    tfidf = get_tfidf_vectorizer(max_features=20, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(sample_texts)
    print(f"- Số mẫu văn bản       : {X_tfidf.shape[0]}")
    print(f"- Số lượng đặc trưng   : {X_tfidf.shape[1]}")
    print(f"- Danh sách từ vựng    : {tfidf.get_feature_names_out()[:10]}...")
    print(f"- Ma trận TF-IDF (dạng mảng numpy):\n{X_tfidf.toarray()[:2]}")

    # 2. Thử nghiệm Word2Vec
    print("\n--- 2. KẾT QUẢ VỚI WORD2VEC (Mean Vector) ---")
    w2v = Word2VecVectorizer(vector_size=10, min_count=1, epochs=5)
    X_w2v = w2v.fit_transform(sample_texts)
    print(f"- Kích thước ma trận Word2Vec : {X_w2v.shape} (3 câu, mỗi câu là 1 vector 10 chiều)")
    print(f"- Vector của câu đầu tiên:\n{X_w2v[0]}")

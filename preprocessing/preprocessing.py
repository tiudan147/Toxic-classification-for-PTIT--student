"""
Module: preprocessing.py
Mục đích:
    Tiền xử lý văn bản tiếng Việt phục vụ bài toán Phân loại bình luận Toxic / Non-toxic
    và khai phá văn bản.
    Xây dựng chuẩn theo kiến thức:
    - Chương 1.1: Cơ sở ngôn ngữ tự nhiên (Đặc trưng tiếng Việt, từ đơn/từ ghép, mơ hồ ngữ nghĩa).
    - Chương 2: Xử lý văn bản (Làm sạch văn bản, chuẩn hóa từ ngữ, xử lý teencode/viết tắt,
               tách từ tiếng Việt - Word Segmentation, loại bỏ ký tự đặc biệt và từ dừng có chọn lọc).
"""

import os
import sys
import re
import html
import unicodedata
from typing import List, Union, Set

# Đảm bảo có thể import underthesea kể cả khi chạy bằng Python toàn cục
venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "Lib", "site-packages"))
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

# Thử import Underthesea để tách từ tiếng Việt chuẩn
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False


# ==========================================
# 1. BẢNG TỪ ĐIỂN CHUẨN HÓA TEENCODE & VIẾT TẮT
# ==========================================
TEENCODE_DICT = {
    # Từ phủ định & liên từ
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "hổng": "không", "hem": "không",
    "đc": "được", "dc": "được", "đk": "được",
    "bt": "biết", "bit": "biết",
    "vs": "với", "w": "với",
    "ng": "người", "ngừoi": "người", "người": "người", "ngta": "người ta", "mn": "mọi người",
    "ad": "admin", "qtv": "quản trị viên",
    "mik": "mình", "mk": "mình", "t": "tao", "m": "mày", "cl": "cậu",
    "j": "gì", "gi": "gì",
    "ntn": "như thế nào", "sao": "sao", "thê": "thế",
    "hnay": "hôm nay", "hqua": "hôm qua", "trc": "trước", "bh": "bây giờ", "h": "giờ",
    "r": "rồi", "rùi": "rồi", "roy": "rồi", "roài": "rồi",
    "ch": "chưa", "chx": "chưa",
    "cx": "cũng",
    "nhiu": "nhiều", "nhìu": "nhiều",
    "ae": "anh em", "ac": "anh chị",
    "fb": "facebook", "ytb": "youtube", "cmt": "bình luận", "stt": "trạng thái",
    
    # Từ chửi bậy viết tắt / lách luật (Rất quan trọng cho bài toán Toxic)
    "đm": "địt mẹ", "dm": "địt mẹ", "đcm": "địt con mẹ", "dcm": "địt con mẹ",
    "vcl": "vãi cả lồn", "vl": "vãi lồn", "vcc": "vãi cả cặc",
    "clgt": "cái lồn gì thế", "đmm": "địt mẹ mày", "dmm": "địt mẹ mày",
    "đb": "đầu buồi", "db": "đầu buồi", "cl": "cái lồn",
    "đéo": "đéo", "del": "đéo", "dell": "đéo", "đell": "đéo",
    "dcmm": "địt con mẹ mày", "đcmm": "địt con mẹ mày", "bome": "bố mẹ",
    "ml": "mặt lồn", "cđm": "cộng đồng mạng"
}

# ==========================================
# 2. BẢNG TỪ DỪNG (STOPWORDS AN TOÀN CHO TOXIC)
# ==========================================
# LƯU Ý: Trong bài toán Toxic, các từ phủ định như "không", "đừng", "chẳng" và
# các đại từ xưng hô "mày", "tao", "nó" TUYỆT ĐỐI KHÔNG ĐƯỢC XÓA vì mang đặc trưng cảm xúc/xúc phạm.
SAFE_STOPWORDS: Set[str] = {
    "là", "và", "của", "một", "những", "các", "có_thể", "thì", "mà", "được",
    "trong", "trên", "dưới", "về", "để", "lúc", "khi", "cho_đến", "bởi_vì",
    "tại_vì", "do_đó", "vì_vậy", "do_vậy", "nhằm", "này", "kia", "ấy", "đó"
}


class TextPreprocessor:
    """
    Pipeline tiền xử lý văn bản tiếng Việt:
    1. Làm sạch thô: HTML tags, URL, Email, khoảng trắng thừa (Chương 2 - Mục 2.1)
    2. Chuẩn hóa tiếng Việt: Unicode dựng sẵn (NFC), chữ thường (lowercase)
    3. Chuẩn hóa kéo dài ký tự (elongation): ví dụ "quáaaa" -> "quá"
    4. Thay thế Teencode & từ viết tắt theo từ điển (Chương 2 - Mục 2.4)
    5. Xử lý ký tự đặc biệt / Emoji (Chương 2 - Mục 2.3)
    6. Tách từ tiếng Việt (Word Segmentation) bằng Underthesea (Chương 2 - Mục 2.2)
    7. Loại bỏ từ dừng an toàn (Optional)
    """

    def __init__(
        self,
        lower: bool = True,
        remove_html_url: bool = True,
        normalize_teencode: bool = True,
        remove_elongation: bool = True,
        remove_punct: bool = False,
        tokenize: bool = True,
        remove_stopwords: bool = False,
        stopwords: Set[str] = None
    ):
        self.lower = lower
        self.remove_html_url = remove_html_url
        self.normalize_teencode = normalize_teencode
        self.remove_elongation = remove_elongation
        self.remove_punct = remove_punct
        self.tokenize = tokenize
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords if stopwords is not None else SAFE_STOPWORDS

    @staticmethod
    def clean_html_and_url(text: str) -> str:
        """Xóa thẻ HTML, URL và địa chỉ email."""
        if not text:
            return ""
        # Giải mã html entities (ví dụ: &amp; -> &)
        text = html.unescape(text)
        # Xóa thẻ HTML (<br>, <a>,...)
        text = re.sub(r"<[^>]+>", " ", text)
        # Xóa URL
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Xóa email
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        return text

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Chuẩn hóa Unicode về dạng dựng sẵn NFC."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def reduce_elongated_chars(text: str) -> str:
        """
        Rút gọn các ký tự kéo dài lặp lại liên tiếp:
        Ví dụ: "chóooo" -> "chó", "đẹpppp" -> "đẹp", "quáaaaa" -> "quá"
        """
        # 1. Rút gọn nguyên âm có dấu kết hợp ký tự không dấu bị lặp (ví dụ: 'óooo' -> 'ó')
        vowels_map = [
            ('ó', 'o'), ('ò', 'o'), ('ỏ', 'o'), ('õ', 'o'), ('ọ', 'o'),
            ('á', 'a'), ('à', 'a'), ('ả', 'a'), ('ã', 'a'), ('ạ', 'a'),
            ('é', 'e'), ('è', 'e'), ('ẻ', 'e'), ('ẽ', 'e'), ('ẹ', 'e'),
            ('í', 'i'), ('ì', 'i'), ('ỉ', 'i'), ('ĩ', 'i'), ('ị', 'i'),
            ('ú', 'u'), ('ù', 'u'), ('ủ', 'u'), ('ũ', 'u'), ('ụ', 'u'),
            ('ý', 'y'), ('ỳ', 'y'), ('ỷ', 'y'), ('ỹ', 'y'), ('ỵ', 'y'),
        ]
        for v, base in vowels_map:
            text = re.sub(rf"{v}{base}+", v, text)
        
        # 2. Rút gọn bất kỳ ký tự chữ cái đơn lẻ nào lặp từ 2 lần trở lên
        text = re.sub(r"([a-zA-Z])\1+", r"\1", text)
        return text

    @staticmethod
    def replace_teencode(text: str) -> str:
        """Thay thế teencode và từ viết tắt dựa trên từ điển chuẩn."""
        words = text.split()
        normalized_words = [TEENCODE_DICT.get(w.lower(), w) for w in words]
        return " ".join(normalized_words)

    @staticmethod
    def clean_special_characters(text: str, keep_punct: bool = True) -> str:
        """
        Xử lý ký tự đặc biệt, biểu tượng, số:
        - Nếu keep_punct=True: Giữ lại các dấu câu cơ bản (. , ! ?), loại bỏ các ký tự rác.
        - Nếu keep_punct=False: Loại bỏ toàn bộ dấu câu chỉ giữ chữ và số.
        """
        if keep_punct:
            # Chỉ giữ chữ cái tiếng Việt, số, và các dấu ngắt câu cơ bản
            text = re.sub(r"[^\w\s.,!?]", " ", text, flags=re.UNICODE)
        else:
            # Loại bỏ toàn bộ dấu câu
            text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Xóa khoảng trắng dư thừa ở đầu, cuối và giữa các từ."""
        return re.sub(r"\s+", " ", text).strip()

    def segment_words(self, text: str) -> str:
        """
        Tách từ tiếng Việt (Word Segmentation).
        Nối các từ ghép bằng dấu gạch dưới `_` (ví dụ: 'học sinh' -> 'học_sinh').
        """
        if not text:
            return ""
        if HAS_UNDERTHESEA:
            try:
                return word_tokenize(text, format="text")
            except Exception:
                return text
        return text

    def filter_stopwords(self, text: str) -> str:
        """Loại bỏ từ dừng (danh sách an toàn không làm mất dấu toxic)."""
        tokens = text.split()
        filtered = [tok for tok in tokens if tok.lower() not in self.stopwords]
        return " ".join(filtered)

    def preprocess_text(self, text: Union[str, float]) -> str:
        """
        Hàm thực hiện trọn vẹn quy trình tiền xử lý cho 1 chuỗi văn bản.
        """
        if not isinstance(text, str):
            return ""

        # 1. Chuẩn hóa Unicode
        text = self.normalize_unicode(text)

        # 2. Xóa HTML & URL
        if self.remove_html_url:
            text = self.clean_html_and_url(text)

        # 3. Chuyển về chữ thường
        if self.lower:
            text = text.lower()

        # 4. Rút gọn ký tự kéo dài
        if self.remove_elongation:
            text = self.reduce_elongated_chars(text)

        # 5. Chuẩn hóa teencode & viết tắt
        if self.normalize_teencode:
            text = self.replace_teencode(text)

        # 6. Loại bỏ ký tự đặc biệt / dấu câu
        text = self.clean_special_characters(text, keep_punct=not self.remove_punct)

        # 7. Xóa khoảng trắng thừa sơ bộ
        text = self.normalize_whitespace(text)

        # 8. Tách từ tiếng Việt
        if self.tokenize:
            text = self.segment_words(text)

        # 9. Lọc từ dừng
        if self.remove_stopwords:
            text = self.filter_stopwords(text)

        # Chuẩn hóa khoảng trắng lần cuối
        text = self.normalize_whitespace(text)
        return text

    def __call__(self, text: Union[str, float]) -> str:
        """Cho phép gọi object như một hàm: preprocessor(text)"""
        return self.preprocess_text(text)


# ==========================================
# TIỆN ÍCH DÀNH CHO XỬ LÝ HÀNG LOẠT (DATAFRAME)
# ==========================================
def preprocess_dataframe(
    df,
    text_col="text",
    output_col="text_clean",
    preprocessor: TextPreprocessor = None
):
    """Áp dụng tiền xử lý trên một cột của pandas DataFrame."""
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    df[output_col] = df[text_col].apply(preprocessor.preprocess_text)
    return df


# ==========================================
# VÍ DỤ CHẠY THỬ
# ==========================================
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    preprocessor = TextPreprocessor(
        lower=True,
        normalize_teencode=True,
        tokenize=True,
        remove_punct=False,
        remove_stopwords=False
    )

    sample_texts = [
        "Sản phẩm này <br> thật tuyệt vời! Mua ngay tại https://shopee.vn/abc. Giá chỉ 99$! 😍😍😍",
        "đr hết thời r cày rank bk còn ngợp clgt đm thằng chóooo",
        "Cuốn sach nay rat hay, toi da doc no 2 lan.",
        "Hôm nay là thứ Tư. Chúng tôi có tiết học về phân tích và xử lý dữ liệu văn bản."
    ]

    print("=== KIỂM THỬ PIPELINE TIỀN XỬ LÝ ===")
    for text in sample_texts:
        cleaned = preprocessor(text)
        print("\n[GỐC]  :", text)
        print("[CLEAN]:", cleaned)

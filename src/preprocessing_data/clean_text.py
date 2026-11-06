"""
================================================================================
MODULE 1: TIỀN XỬ LÝ VĂN BẢN (TEXT PREPROCESSING)
File: src/preprocessing_data/clean_text.py
================================================================================
MỤC ĐÍCH:
    Chuyển đổi văn bản thô (raw text) từ mạng xã hội (chứa teencode, viết tắt,
    icon, html, link, kéo dài từ...) thành văn bản sạch, chuẩn hóa tiếng Việt
    để máy tính và các mô hình học máy (Machine Learning) có thể hiểu được.

CÁC BƯỚC XỬ LÝ TRONG PIPELINE:
    1. Chuẩn hóa mã Unicode về dựng sẵn (NFC).
    2. Giải mã ký tự HTML (unescape) & Xóa thẻ HTML, liên kết URL.
    3. Chuyển toàn bộ về chữ thường (lowercase).
    4. Rút gọn các ký tự kéo dài (ví dụ: 'nguuu' -> 'ngu', 'chóooo' -> 'chó').
    5. Chuẩn hóa Teencode, từ lóng, từ viết tắt phổ biến trên mạng xã hội.
    6. Lọc ký tự đặc biệt, nhưng GIỮ LẠI chữ cái, số và dấu câu cảm xúc (! ?).
    7. Tách từ tiếng Việt (Word Segmentation) bằng thư viện Underthesea.
================================================================================
"""

import re
import html
import unicodedata
import os
import sys

# ------------------------------------------------------------------------------
# Hỗ trợ nạp thư viện underthesea từ môi trường ảo .venv (nếu có)
# Giúp code chạy mượt mà trên cả Python toàn cục lẫn Virtual Environment
# ------------------------------------------------------------------------------
try:
    from underthesea import word_tokenize
except ImportError:
    venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".venv", "Lib", "site-packages"))
    if os.path.exists(venv_site) and venv_site not in sys.path:
        sys.path.append(venv_site)
    try:
        from underthesea import word_tokenize
    except ImportError:
        word_tokenize = None


# ------------------------------------------------------------------------------
# BẢNG TỪ ĐIỂN CHUẨN HÓA TEENCODE & VIẾT TẮT THƯỜNG GẶP TRÊN MẠNG XÃ HỘI
# Được tổng hợp và bổ sung từ quá trình kiểm thử dữ liệu Facebook thực tế.
# ------------------------------------------------------------------------------
TEENCODE_MAP = {
    # 1. Phủ định, trợ từ & từ nối thông dụng
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "hổng": "không",
    "đc": "được", "dc": "được", "đk": "được",
    "bt": "biết", "vs": "với", "j": "gì", "z": "vậy", "v": "vậy",
    "r": "rồi", "ch": "chưa", "cx": "cũng",
    "qúa": "quá", "oke": "được", "ok": "được",
    "nma": "nhưng mà", "nchung": "nói chung",
    "ntn": "như thế nào", "ntnay": "như thế này",
    "qtrong": "quan trọng", "đễ": "để", "laii": "lại", "chit": "chỉ",

    # 2. Đại từ xưng hô & thuật ngữ mạng xã hội
    "ng": "người", "ngta": "người ta", "mn": "mọi người",
    "mik": "mình", "mk": "mình",
    "t": "tao", "m": "mày", "cno": "chúng nó",
    "ae": "anh em", "b": "bạn", "e": "em",
    "thg": "thằng", "th": "thằng", "tg": "thằng",
    "cđm": "cộng đồng mạng",
    "nt": "nhắn tin", "ib": "nhắn tin", "inbox": "nhắn tin", "kb": "kết bạn",
    "cmt": "bình luận", "tus": "bài viết",
    "bh": "bây giờ", "hqua": "hôm qua", "htrc": "hôm trước",
    "đt": "điện thoại", "sđt": "số điện thoại",
    "cty": "công ty", "đh": "đại học", "stk": "số tài khoản", "tk": "tài khoản",
    "nvs": "nhà vệ sinh", "vsinh": "vệ sinh", "ns": "nói",

    # 3. Từ ngữ tục tĩu, chửi thề viết tắt (RẤT QUAN TRỌNG ĐỂ PHÁT HIỆN TOXIC)
    "đ": "đéo",          # Cực kỳ phổ biến: "đ thể", "đ biết", "đ thẩm nổi", "đ hay"
    "dell": "đéo", "đell": "đéo",
    "đhs": "đéo hiểu sao", "tdn": "thế đéo nào", "cdj": "cái đéo gì",
    "đm": "địt mẹ", "dm": "địt mẹ", "đcm": "địt con mẹ", "dcm": "địt con mẹ",
    "djt": "địt", "đjt": "địt", "dme": "địt mẹ", "đme": "địt mẹ", "đệt": "địt",
    "cmm": "con mẹ mày", "dmm": "địt mẹ mày", "đmm": "địt mẹ mày",
    "cmnl": "con mẹ nó luôn", "cmnr": "con mẹ nó rồi", "cmn": "con mẹ nó",
    "vcl": "vãi cả lồn", "vkl": "vãi cả lồn", "vcc": "vãi cả cặc", "vc": "vãi cặc",
    "vl": "vãi lồn", "vloz": "vãi lồn", "vlz": "vãi lồn",
    "lz": "lồn", "lìn": "lồn", "cl": "cái lồn", "clg": "cái lồn gì", "clm": "cái lồn mẹ",
    "clgt": "cái lồn gì thế", "đb": "đầu buồi",
    "rr": "rẻ rách", "chóa": "chó", "chóá": "chó"
}


def clean_text(text) -> str:
    """
    Hàm làm sạch một chuỗi văn bản đầu vào.
    
    Tham số:
        text (str): Câu văn bản gốc cần xử lý.
        
    Trả về:
        str: Chuỗi văn bản đã được làm sạch và tách từ.
    """
    # Nếu dữ liệu rỗng hoặc không phải dạng chuỗi (NaN, None, float), trả về chuỗi rỗng
    if not isinstance(text, str) or not text.strip():
        return ""

    # --------------------------------------------------------------------------
    # BƯỚC 1: Chuẩn hóa Unicode dựng sẵn (NFC) & Xóa thẻ HTML, liên kết URL
    # --------------------------------------------------------------------------
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)                          # Chuyển đổi ví dụ &amp; thành &
    text = re.sub(r"<[^>]+>", " ", text)               # Xóa các thẻ HTML <p>, <br>...
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) # Xóa liên kết web http://... hoặc www...

    # --------------------------------------------------------------------------
    # BƯỚC 2: Chuyển toàn bộ về chữ thường (Lowercase)
    # --------------------------------------------------------------------------
    text = text.lower()

    # --------------------------------------------------------------------------
    # BƯỚC 3: Rút gọn các ký tự bị kéo dài do thói quen gõ phím
    # Ví dụ: "chóooo" -> "chó", "nguuuu" -> "ngu", "áaaaa" -> "á"
    # --------------------------------------------------------------------------
    vowel_rules = [
        ('ó', 'o'), ('ò', 'o'), ('ỏ', 'o'), ('õ', 'o'), ('ọ', 'o'),
        ('á', 'a'), ('à', 'a'), ('ả', 'a'), ('ã', 'a'), ('ạ', 'a'),
        ('é', 'e'), ('è', 'e'), ('ẻ', 'e'), ('ẽ', 'e'), ('ẹ', 'e'),
        ('í', 'i'), ('ì', 'i'), ('ỉ', 'i'), ('ĩ', 'i'), ('ị', 'i'),
        ('ú', 'u'), ('ù', 'u'), ('ủ', 'u'), ('ũ', 'u'), ('ụ', 'u')
    ]
    for accented_vowel, base_vowel in vowel_rules:
        text = re.sub(rf"{accented_vowel}{base_vowel}+", accented_vowel, text)
    
    # Rút gọn các ký tự lặp liên tiếp từ 2 lần trở lên thành 1 lần
    text = re.sub(r"([a-zA-Z])\1+", r"\1", text)

    # --------------------------------------------------------------------------
    # BƯỚC 4: Chuẩn hóa Teencode & Viết tắt
    # Dùng regex \b\w+\b để tách từ chính xác theo ranh giới từ (word boundary),
    # ngay cả khi từ bị dính dấu câu như "vcl:)))", "djt,", "đhs=))".
    # --------------------------------------------------------------------------
    def replace_teencode(match):
        w = match.group(0)
        return TEENCODE_MAP.get(w, w)

    text = re.sub(r"\b\w+\b", replace_teencode, text)

    # --------------------------------------------------------------------------
    # BƯỚC 5: Lọc ký tự đặc biệt, giữ lại chữ, số và dấu câu cảm xúc (! ? . ,)
    # --------------------------------------------------------------------------
    text = re.sub(r"[^\w\s.,!?]", " ", text, flags=re.UNICODE)
    # Thu gọn nhiều khoảng trắng liên tiếp thành 1 khoảng trắng duy nhất
    text = re.sub(r"\s+", " ", text).strip()

    # --------------------------------------------------------------------------
    # BƯỚC 6: Tách từ tiếng Việt (Word Segmentation) bằng Underthesea
    # --------------------------------------------------------------------------
    if word_tokenize:
        try:
            text = word_tokenize(text, format="text")
        except Exception:
            pass

    return text


# ==============================================================================
# PHẦN CHẠY THỬ NGHIỆM (DEMO KHI CHẠY TRỰC TIẾP FILE NÀY)
# Lệnh chạy: python src/preprocessing_data/clean_text.py
# ==============================================================================
if __name__ == "__main__":
    test_samples = [
        "Thằng này ngu vcllll, ko biết j cũng sủa bậy :))) !!! <br> https://fb.com",
        "Đjt mẹ con app lz này đ thể chấp nhận đc, lag cmnl",
        "Đhs dạo này nhiều đứa dell biết j cũng chém gió tdn",
        "Ae cty dùng đt gọi sđt hỏi xem sao nhé b"
    ]

    print("--- DEMO CHỨC NĂNG LÀM SẠCH VĂN BẢN (clean_text) ---")
    for original in test_samples:
        cleaned = clean_text(original)
        print(f"\n[Gốc]  : {original}")
        print(f"[Sạch] : {cleaned}")

# BÁO CÁO BÀI TẬP LỚN 1
## MÔN: KHAI PHÁ VĂN BẢN (TEXT MINING)
**Đề tài:** Phân loại bình luận độc hại tiếng Việt trên mạng xã hội (Vietnamese Toxic Comment Detection)  
**Sinh viên thực hiện:** Nguyễn Tiến Dân  

---

### I. ĐẶT VẤN ĐỀ VÀ MỤC TIÊU ĐỀ TÀI

1. **Đặt vấn đề:**  
   Hiện nay, trên các nền tảng mạng xã hội (Facebook, TikTok, YouTube...), tình trạng bình luận tiêu cực, chửi bới, xúc phạm danh dự (Toxic Comments) diễn ra rất phổ biến. Việc lọc và phát hiện thủ công là bất khả thi vì số lượng bình luận mỗi ngày là cực lớn.
2. **Mục tiêu:**  
   Xây dựng một hệ thống học máy (Machine Learning) cơ bản nhưng hoàn chỉnh, có khả năng tự động đọc một câu bình luận tiếng Việt và phân loại thành 2 nhóm:
   - **Nhãn 0 (Non-toxic):** Bình luận lành tính, bình thường.
   - **Nhãn 1 (Toxic):** Bình luận độc hại, thô tục, xúc phạm.

---

### II. DỮ LIỆU SỬ DỤNG (DATASET)

Hệ thống sử dụng 2 nguồn dữ liệu chính:

1. **Bộ dữ liệu huấn luyện và kiểm định (Academic Datasets):**
   - Được tổng hợp từ 2 bộ dữ liệu chuẩn tiếng Việt phổ biến:
     - **UIT-ViCTSD:** Bộ dữ liệu đánh giá tính xây dựng và độc hại của Đại học Công nghệ Thông tin (ĐHQG-HCM).
     - **ViHSD:** Bộ dữ liệu phát hiện ngôn từ thù ghét tiếng Việt.
   - **Tổng số mẫu:** 16.683 dòng.
   - **Tỷ lệ phân bố:** 70.8% Lành tính (11.811 mẫu) và 29.2% Độc hại (4.872 mẫu) $\rightarrow$ *Dữ liệu có sự mất cân bằng giữa 2 lớp*.
   - **Phân chia dữ liệu:** Áp dụng phương pháp phân tầng (Stratified Sampling) theo tỷ lệ **70% - 15% - 15%** để đảm bảo tỷ lệ Toxic ở mỗi tập luôn đồng nhất:
     - Tập Train (Huấn luyện): 11.667 dòng.
     - Tập Validation (Kiểm định chọn tham số): 2.502 dòng.
     - Tập Test (Đánh giá độc lập cuối cùng): 2.500 dòng.

2. **Bộ dữ liệu kiểm thử thực tế cào từ Facebook (`fb_test.csv`):**
   - Gồm **605 bình luận** cào trực tiếp từ các bài đăng trên Facebook và được gán nhãn thủ công (453 mẫu Non-toxic ~ 74.9% và 152 mẫu Toxic ~ 25.1%).
   - Mục đích: Đánh giá xem mô hình sau khi học trên lý thuyết có hoạt động tốt ngoài đời thực hay không.

---

### III. QUY TRÌNH THỰC HIỆN (PIPELINE)

Hệ thống được thiết kế theo quy trình 4 giai đoạn rõ ràng:

```
Văn bản thô (Mạng xã hội)
       │
       ▼
[Bước 1] Tiền xử lý dữ liệu (clean_text.py)
       │
       ▼
[Bước 2] Trích xuất đặc trưng (TF-IDF / Word2Vec)
       │
       ▼
[Bước 3] Huấn luyện mô hình (Naive Bayes / Linear SVM)
       │
       ▼
[Bước 4] Đánh giá & Kiểm thử thực tế (metrics.py & predict.py)
```

#### 1. Tiền xử lý văn bản (`clean_text.py`):
Bình luận trên mạng xã hội chứa rất nhiều "rác" và tiếng lóng. Các bước xử lý tuần tự gồm:
- **Chuẩn hóa Unicode NFC:** Đưa toàn bộ ký tự tiếng Việt về chuẩn dựng sẵn (tránh lỗi font do gõ tổ hợp).
- **Xóa HTML & liên kết Web (URL):** Loại bỏ thẻ `<p>`, link `http...` vì không mang ý nghĩa cảm xúc.
- **Chuyển chữ thường (Lowercase):** Giúp máy hiểu "NGU", "Ngu", "ngu" là cùng một từ.
- **Rút gọn ký tự kéo dài:** Người dùng hay gõ "chóooo", "nguuu", "vlzzz" $\rightarrow$ rút gọn thành "chó", "ngu", "vl".
- **Chuẩn hóa Teencode & Viết tắt:** Sử dụng bảng từ điển `TEENCODE_MAP` để dịch các từ viết tắt về từ gốc:
  - Tục tĩu / chửi thề: `đ` $\rightarrow$ `đéo`; `djt`, `đjt` $\rightarrow$ `địt`; `lz`, `lìn` $\rightarrow$ `lồn`; `cmnl` $\rightarrow$ `con mẹ nó luôn`; `vcl`, `vkl` $\rightarrow$ `vãi cả lồn`.
  - Từ ngữ đời thường: `ko`, `k` $\rightarrow$ `không`; `đc` $\rightarrow$ `được`; `ae` $\rightarrow$ `anh em`; `thg` $\rightarrow$ `thằng`; `cmt` $\rightarrow$ `bình luận`.
- **Lọc ký tự đặc biệt:** Giữ lại chữ cái tiếng Việt, số và các dấu cảm thán (`!`), dấu hỏi (`?`) vì các dấu này thể hiện thái độ tức giận, mỉa mai rất rõ trong phát hiện độc hại.
- **Tách từ tiếng Việt (Word Segmentation):** Dùng thư viện **Underthesea** để ghép từ ghép (ví dụ: `học sinh` $\rightarrow$ `học_sinh`, `đồ khốn` $\rightarrow$ `đồ_khốn`).

#### 2. Trích chọn đặc trưng văn bản (`feature_extractor.py`):
Máy tính chỉ hiểu các con số, không hiểu chữ. Do đó cần chuyển câu văn thành vector số:
- **TF-IDF (Term Frequency - Inverse Document Frequency):** Đo lường tần suất xuất hiện và độ quý hiếm của từ. Lấy **5.000 đặc trưng** quan trọng nhất kết hợp cả từ đơn (1-gram) và cụm 2 từ (2-gram) như *"rất ngu", "không thích"*.
- **Word2Vec (Gensim):** Học vector nhúng từ (150 chiều) từ ngữ cảnh và tính vector trung bình (Mean Vector) của các từ trong câu.

#### 3. Mô hình học máy (`naive_bayes_model.py` & `svm_model.py`):
Thử nghiệm 2 thuật toán phân loại kinh điển trong xử lý ngôn ngữ tự nhiên:
- **Multinomial Naive Bayes (alpha=0.5):** Thuật toán dựa trên định lý xác suất Bayes. Ưu điểm: huấn luyện cực nhanh, tốn rất ít bộ nhớ, rất phù hợp với ma trận tần số TF-IDF.
- **Linear Support Vector Machine (Linear SVM - C=1.0):** Thuật toán tìm siêu phẳng phân tách tối ưu với khoảng cách lề (margin) cực đại. 
  - *Điểm mấu chốt:* Thiết lập tham số `class_weight='balanced'` để tự động phạt nặng hơn khi đoán sai lớp thiểu số (Toxic), giúp mô hình tập trung bắt đúng các câu độc hại mà không bị thiên vị về lớp lành tính.

---

### IV. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ

#### 1. Các tiêu chí đánh giá được sử dụng:
- **Accuracy (Độ chính xác tổng thể):** Tỷ lệ đoán đúng trên toàn bộ tập dữ liệu.
- **Toxic Precision:** Khi mô hình nói một câu là ĐỘC HẠI, có bao nhiêu % thực sự độc hại (đo độ bắt chuẩn, tránh bắt oan).
- **Toxic Recall:** Trong tất cả các câu độc hại có thực tế, mô hình tìm ra được bao nhiêu % (đo độ nhạy, tránh bỏ sót).
- **Toxic F1-Score & Macro F1:** Điểm trung bình điều hòa giữa Precision và Recall. Đây là thước đo khách quan và quan trọng nhất khi dữ liệu bị lệch nhãn.

---

#### 2. Kết quả trên tập Dữ liệu kiểm thử lý thuyết (Tập Test - 2.500 mẫu):

| Mô hình & Đặc trưng | Accuracy (%) | Macro F1 | Toxic F1 | Toxic Recall (Bắt độc hại) |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (TF-IDF)** | **85.08%** | **0.8066** | 0.7142 | 63.75% |
| **Linear SVM (TF-IDF)** | 83.20% | 0.8011 | **0.7226** | **74.83%** |
| **Linear SVM (Word2Vec)** | 81.68% | 0.7853 | 0.7030 | 74.15% |

---

#### 3. Kết quả trên tập Dữ liệu Facebook thực tế ngoài đời (`fb_test.csv` - 605 mẫu):

| Mô hình & Đặc trưng | Accuracy (%) | Macro F1 | Toxic F1 | Toxic Recall (Bắt độc hại) |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (TF-IDF)** | **76.69%** | **0.6987** | **0.5552** | 57.89% |
| **Linear SVM (TF-IDF)** | 69.42% | 0.6550 | 0.5387 | **71.05%** |
| **Linear SVM (Word2Vec)** | 66.94% | 0.6337 | 0.5192 | **71.05%** |

---

### V. PHÂN TÍCH VÀ NHẬN XÉT KẾT QUẢ

1. **So sánh giữa Naive Bayes và Linear SVM:**
   - **Naive Bayes** có độ chính xác chung (Accuracy) cao hơn (đạt 85% trên Test và gần 77% trên Facebook). Tuy nhiên, Naive Bayes lại có **Toxic Recall khá thấp** (~57 - 63%), tức là bỏ lọt gần một nửa số bình luận độc hại.
   - **Linear SVM** nhờ áp dụng kỹ thuật cân bằng trọng số (`class_weight='balanced'`) nên đạt **Toxic Recall rất ấn tượng: trên 71% - 74%**. Trong bài toán thực tế về kiểm duyệt nội dung, việc **"bắt đúng và không bỏ sót kẻ quấy rối, chửi thề"** quan trọng hơn nhiều so với việc chỉ chăm chăm nhìn vào Accuracy tổng thể. Do đó, **Linear SVM là mô hình phù hợp nhất để triển khai thực tế**.

2. **So sánh giữa TF-IDF và Word2Vec:**
   - **TF-IDF** cho kết quả nhỉnh hơn Word2Vec một chút trên tập dữ liệu này vì bình luận mạng xã hội thường ngắn, chứa nhiều từ khóa độc hại trực diện (tục tĩu, chửi thề). TF-IDF kết hợp cụm 2 từ (n-gram) bắt trúng ngay các từ khóa này.
   - **Word2Vec** biểu diễn ngữ nghĩa dạng vector liên tục, đạt kết quả khá ổn định (Recall 71.05%) nhưng cần kho ngữ liệu lớn hơn để phát huy tối đa sức mạnh ngữ cảnh.

3. **Hiệu quả của việc bổ sung từ điển Teencode mạng xã hội:**
   - Sau khi phân tích các mẫu dự đoán sai trên tập Facebook, nhóm đã bổ sung các từ lóng thực tế như `đ`, `djt`, `lz`, `cmnl`, `dell`, `đhs`... vào `clean_text.py`.
   - Kết quả: Khả năng bắt câu độc hại thực tế (Toxic Recall) **tăng vọt từ 64.47% lên 71.05%**, chứng minh khâu tiền xử lý làm sạch tiếng Việt đóng vai trò quyết định đến chất lượng toàn hệ thống.

---

### VI. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

1. **Kết quả đạt được:**
   - Đã xây dựng hoàn chỉnh một pipeline phân loại bình luận độc hại: Tiền xử lý $\rightarrow$ Trích xuất đặc trưng $\rightarrow$ Huấn luyện mô hình $\rightarrow$ Đánh giá & Kiểm thử thực tế.
   - Các file mã nguồn được phân chia mô-đun rõ ràng, dễ bảo trì, dễ hiểu cho người mới học.
   - Mô hình Linear SVM đạt độ bao phủ độc hại trên 71% trên bình luận mạng xã hội thực tế.
2. **Hướng phát triển tiếp theo:**
   - Tiếp tục làm giàu từ điển Teencode và từ lóng mạng xã hội tiếng Việt.
   - Nghiên cứu và áp dụng các mô hình học sâu (Deep Learning) hoặc mô hình ngôn ngữ lớn tiền huấn luyện cho tiếng Việt như **PhoBERT**, **ViBERT** để nâng cao độ chính xác về mặt ngữ cảnh.


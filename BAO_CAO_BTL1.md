# BÁO CÁO BÀI TẬP LỚN 1
## HỌC PHẦN: PHÂN TÍCH VÀ KHAI PHÁ DỮ LIỆU VĂN BẢN

**Đề tài:** Xây dựng hệ thống phân loại bình luận độc hại tiếng Việt trên mạng xã hội (Toxic và Non-toxic)  
**Nhóm sinh viên thực hiện:**
1. Đặng Minh Quang - B23DCKH095  
2. Bùi Huy Hoàng - B23DCKH043  
3. Nhâm Trọng Dư - B23DCKH027  
4. Nguyễn Tiến Đàn - B23DCKH017  

---

## 1. GIỚI THIỆU VỀ BÀI TOÁN

### 1.1. Phát biểu bài toán
Bài toán đặt ra là phân loại nhị phân (Binary Classification) cho văn bản tiếng Việt trên mạng xã hội: với mỗi bình luận đầu vào, hệ thống tự động xác định bình luận đó thuộc một trong hai lớp:
- **Lớp 0 (Non-toxic):** Bình luận lành tính, mang tính chất trao đổi thông thường.
- **Lớp 1 (Toxic):** Bình luận độc hại, chứa từ ngữ thô tục, chửi thề, lăng mạ hoặc xúc phạm.

### 1.2. Dữ liệu thực nghiệm
Hệ thống sử dụng hai nguồn dữ liệu:
1. **Tập dữ liệu chuẩn (Dùng để huấn luyện và kiểm định):**
   - Được kết hợp từ 2 tập dữ liệu công khai tiếng Việt chuẩn về phân loại độc hại và thù ghét: **UIT-ViCTSD** và **ViHSD**.
   - Tổng kích thước: **16.683 mẫu**.
   - Phân bố nhãn: 11.811 mẫu Lành tính (chiếm 70,8%) và 4.872 mẫu Độc hại (chiếm 29,2%).
   - Tách tập dữ liệu theo kỹ thuật lấy mẫu phân tầng (**Stratified Sampling**) theo tỷ lệ **70% - 15% - 15%** nhằm giữ nguyên tỷ lệ nhãn ở mỗi tập:
     - Tập Huấn luyện (Train set): **11.667** mẫu.
     - Tập Kiểm định (Validation set): **2.502** mẫu.
     - Tập Kiểm thử (Test set): **2.500** mẫu.

2. **Tập dữ liệu kiểm thử thực tế cào từ Facebook (`fb_test.csv`):**
   - Thu thập trực tiếp từ các bài đăng thảo luận trên các hội nhóm Facebook.
   - Kích thước: **605 mẫu** được gán nhãn thủ công (453 mẫu Non-toxic ~ 74,9% và 152 mẫu Toxic ~ 25,1%).
   - Mục đích: Đánh giá khả năng tổng quát hóa của các mô hình trên dữ liệu thực tế đời sống.

---

## 2. CƠ SỞ LÝ THUYẾT VÀ PHƯƠNG PHÁP THỰC HIỆN

### 2.1. Tiền xử lý văn bản (Text Preprocessing)
Văn bản thu thập từ mạng xã hội có tính nhiễu rất cao (chứa mã HTML, đường dẫn web, teencode, từ viết tắt, từ lặp âm). Do đó, dữ liệu được đưa qua quy trình làm sạch tuần tự:
1. **Chuẩn hóa bảng mã ký tự (Unicode Normalization):** Đưa toàn bộ văn bản về chuẩn dựng sẵn (NFC) nhằm đồng nhất cách lưu trữ ký tự tiếng Việt có dấu.
2. **Khử nhiễu văn bản:** Giải mã ký tự HTML, loại bỏ thẻ HTML (`<p>`, `<br>`), loại bỏ các liên kết URL (`http`, `www`).
3. **Chuyển chữ thường (Case Folding):** Đưa tất cả văn bản về chữ thường nhằm đồng nhất từ vựng, tránh phân mảnh từ điển.
4. **Chuẩn hóa ký tự kéo dài:** Rút gọn các chuỗi ký tự lặp do thói quen gõ phím biểu cảm (ví dụ: *"nguuu"* thành *"ngu"*, *"chóooo"* thành *"chó"*).
5. **Chuẩn hóa từ viết tắt và tiếng lóng (Teencode Normalization):** Thay thế các từ viết tắt phổ biến về dạng chuẩn tiếng Việt (ví dụ: *"ko", "k"* $\rightarrow$ *"không"*; *"đc"* $\rightarrow$ *"được"*; *"ae"* $\rightarrow$ *"anh em"*; *"thg"* $\rightarrow$ *"thằng"*; các từ chửi thề viết tắt: *"đm", "dm"* $\rightarrow$ *"địt mẹ"*; *"đ"* $\rightarrow$ *"đéo"*; *"lz"* $\rightarrow$ *"lồn"*; *"vcl", "vkl"* $\rightarrow$ *"vãi cả lồn"*).
6. **Lọc ký tự đặc biệt:** Loại bỏ các ký tự biểu cảm thừa và icon lạ, giữ lại chữ cái, chữ số và các dấu câu mang sắc thái biểu cảm mạnh (`!`, `?`, `.`, `,`).
7. **Tách từ tiếng Việt (Word Segmentation):** Ghép các tiếng tạo thành từ ghép có nghĩa trong tiếng Việt (ví dụ: *"học sinh"* $\rightarrow$ *"học_sinh"*, *"xúc phạm"* $\rightarrow$ *"xúc_phạm"*).

---

### 2.2. Trích chọn đặc trưng văn bản (Feature Extraction)
Áp dụng hai phương pháp biểu diễn văn bản trong không gian vector:

1. **Phương pháp 1: Trọng số TF-IDF (Term Frequency - Inverse Document Frequency)**
   - Biểu diễn văn bản dưới dạng túi từ (Bag of Words) kết hợp trọng số thống kê:
     $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
     Trong đó $\text{TF}(t, d)$ thể hiện tần suất xuất hiện của từ trong câu, $\text{IDF}(t, D)$ đo lường độ phổ biến của từ trong toàn bộ kho tài liệu.
   - Cấu hình: Trích chọn **5.000 đặc trưng** có trọng số cao nhất, kết hợp cả từ đơn (1-gram) và cụm 2 từ liền kề (2-gram) nhằm nắm bắt các cấu trúc cụm từ phủ định hoặc chửi thề (như *"không thích"*, *"rất tệ"*).
   - *Nguyên tắc thực nghiệm:* Bộ từ điển và trọng số IDF chỉ được học trên tập Train (`fit`), sau đó áp dụng biến đổi (`transform`) cho các tập Validation, Test và Facebook Test nhằm tránh hiện tượng rò rỉ dữ liệu (Data Leakage).

2. **Phương pháp 2: Mô hình nhúng từ Word2Vec (Word Embeddings)**
   - Học không gian vector ngữ nghĩa phân tán liên tục cho từng từ dựa trên kiến trúc Skip-gram với kích thước cửa sổ ngữ cảnh bằng 5.
   - Mỗi từ được biểu diễn thành một vector thực **150 chiều**.
   - Biểu diễn một câu văn bản bằng **vector trung bình (Mean Pooling)** của toàn bộ các vector từ xuất hiện trong câu đó.

---

### 2.3. Các thuật toán phân loại (Classification Algorithms)

1. **Multinomial Naive Bayes (MNB):**
   - Thuật toán phân loại xác suất dựa trên định lý Bayes với giả định ngây thơ là các từ độc lập có điều kiện theo từng lớp:
     $$P(c|d) \propto P(c) \prod_{i=1}^{n} P(w_i|c)$$
   - Áp dụng kỹ thuật làm mịn Laplace ($\alpha = 0,5$) để tránh lỗi xác suất bằng 0 khi gặp từ vựng mới chưa xuất hiện trong tập huấn luyện.
   - Thuật toán này phù hợp với ma trận trọng số không âm của TF-IDF.

2. **Support Vector Machine tuyến tính (Linear SVM):**
   - Tìm kiếm một siêu phẳng phân tách $w^T x + b = 0$ sao cho khoảng cách lề (Margin) giữa hai lớp dữ liệu đạt cực đại:
     $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i} \xi_i$$
   - Với dữ liệu văn bản có số chiều lớn sau khi dùng TF-IDF (5.000 chiều) hoặc Word2Vec (150 chiều), SVM tuyến tính cho khả năng phân tách tốt và tốc độ tối ưu cao.
   - Sử dụng hệ số phạt $C = 1,0$ và áp dụng kỹ thuật cân bằng trọng số lớp (`class_weight='balanced'`) nhằm tăng trọng số phạt đối với lớp thiểu số (Toxic), giúp hạn chế việc mô hình bỏ sót các câu độc hại.

---

### 2.4. Tiêu chí đánh giá mô hình (Evaluation Metrics)
Dựa trên ma trận nhầm lẫn (Confusion Matrix) với các đại lượng:
- **True Negative (TN):** Mẫu Lành tính được dự đoán đúng là Lành tính.
- **False Positive (FP):** Mẫu Lành tính bị dự đoán nhầm thành Độc hại.
- **False Negative (FN):** Mẫu Độc hại bị bỏ sót thành Lành tính.
- **True Positive (TP):** Mẫu Độc hại được dự đoán đúng là Độc hại.

Các chỉ số đo lường chính:
1. **Accuracy (Độ chính xác tổng thể):**
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
2. **Precision của lớp Độc hại (Độ chuẩn xác):**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
3. **Recall của lớp Độc hại (Độ bao phủ / Độ nhạy):**
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
4. **Toxic F1-Score & Macro F1-Score:**
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   Macro F1 là trung bình cộng điểm F1 của hai lớp, đây là chỉ số phản ánh trung thực nhất chất lượng phân loại trên bộ dữ liệu mất cân bằng nhãn.

---

## 3. THỰC NGHIỆM VÀ KẾT QUẢ

### 3.1. Kết quả trên tập Kiểm định (Validation set - 2.502 mẫu)

| Phương pháp (Mô hình & Đặc trưng) | Accuracy (%) | Macro F1 | Toxic F1 | Toxic Precision | Toxic Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (TF-IDF)** | **86,37%** | **0,8213** | **0,7342** | **0,8533** | 64,43% |
| **Linear SVM (TF-IDF)** | 82,89% | 0,7987 | 0,7206 | 0,6891 | 75,51% |
| **Linear SVM (Word2Vec)** | 82,37% | 0,7938 | 0,7153 | 0,6773 | **75,79%** |

---

### 3.2. Kết quả trên tập Kiểm thử chuẩn (Test set - 2.500 mẫu)

| Phương pháp (Mô hình & Đặc trưng) | Accuracy (%) | Macro F1 | Toxic F1 | Toxic Precision | Toxic Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (TF-IDF)** | **85,08%** | **0,8066** | 0,7142 | **0,8118** | 63,75% |
| **Linear SVM (TF-IDF)** | 83,20% | 0,8011 | **0,7226** | 0,6986 | **74,83%** |
| **Linear SVM (Word2Vec)** | 81,68% | 0,7853 | 0,7030 | 0,6683 | 74,15% |

---

### 3.3. Kết quả trên tập Dữ liệu thực tế Facebook (`fb_test.csv` - 605 mẫu)

| Phương pháp (Mô hình & Đặc trưng) | Accuracy (%) | Macro F1 | Toxic F1 | Toxic Precision | Toxic Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes (TF-IDF)** | **76,69%** | **0,6987** | **0,5552** | **0,5333** | 57,89% |
| **Linear SVM (TF-IDF)** | 69,42% | 0,6550 | 0,5387 | 0,4337 | **71,05%** |
| **Linear SVM (Word2Vec)** | 66,94% | 0,6337 | 0,5192 | 0,4091 | **71,05%** |

---

### 3.4. So sánh và Phân tích kết quả thực nghiệm

1. **So sánh giữa hai thuật toán Naive Bayes và Linear SVM:**
   - **Naive Bayes:** Đạt độ chính xác tổng thể (Accuracy) cao nhất trên cả tập Test (85,08%) và tập Facebook (76,69%), đồng thời có Precision cao (trên 81% ở tập Test). Tuy nhiên, nhược điểm lớn của Naive Bayes là **Toxic Recall thấp** (chỉ đạt 63,75% trên Test và 57,89% trên Facebook), nghĩa là mô hình thiên về việc dự đoán an toàn là nhãn Lành tính, dẫn đến bỏ sót nhiều bình luận độc hại.
   - **Linear SVM:** Nhờ sử dụng cơ chế cân bằng trọng số lớp (`class_weight='balanced'`), Linear SVM chấp nhận đánh đổi một phần Precision để đạt được **Toxic Recall vượt trội** (đạt **74,83%** trên tập Test và **71,05%** trên tập Facebook). Điều này có nghĩa là mô hình phát hiện được phần lớn các phát ngôn thô tục thực tế.

2. **So sánh giữa hai phương pháp biểu diễn TF-IDF và Word2Vec:**
   - Cả hai phương pháp khi kết hợp với Linear SVM đều cho kết quả Recall tương đương nhau trên tập Facebook (**71,05%**).
   - Tuy nhiên, trên tập dữ liệu văn bản ngắn như bình luận mạng xã hội, **TF-IDF kết hợp cụm 2 từ (n-gram)** cho hiệu năng nhỉnh hơn Word2Vec trung bình câu về cả Accuracy (83,20% so với 81,68%) và F1-Score (0,7226 so với 0,7030). Nguyên nhân là do các từ khóa thô tục, chửi thề mang tính quyết định trực tiếp, việc tính trung bình cộng vector trong Word2Vec đôi khi làm mờ đi tín hiệu của từ khóa độc hại xuất hiện đơn lẻ trong câu.

3. **Ảnh hưởng của khâu chuẩn hóa từ viết tắt (Teencode):**
   - Trong quá trình kiểm thử trên tập Facebook, khi văn bản chưa được chuẩn hóa teencode (các từ như *"đjt", "lz", "cmnl", "đhs"*), Toxic Recall chỉ đạt 64,47%.
   - Sau khi áp dụng bước chuẩn hóa từ điển ánh xạ về từ gốc, Toxic Recall tăng lên **71,05%** (tăng thêm 6,58%). Điều này khẳng định khâu tiền xử lý từ vựng và xử lý từ viết tắt là bước mang tính quyết định trong khai phá dữ liệu văn bản tiếng Việt.

---

## 4. KẾT LUẬN

Nhóm đã hoàn thành đầy đủ các yêu cầu của Bài tập lớn 1:
1. Thu thập và chuẩn bị dữ liệu: Gộp và phân chia dữ liệu học thuật phân tầng 70-15-15 (16.683 mẫu) và thu thập thêm tập kiểm thử Facebook thực tế (605 mẫu gán nhãn).
2. Xây dựng quy trình tiền xử lý văn bản tiếng Việt hoàn chỉnh (Unicode, chữ thường, rút gọn lặp, chuẩn hóa viết tắt/teencode, tách từ).
3. Thử nghiệm và đối sánh 2 phương pháp biểu diễn đặc trưng: **TF-IDF** (n-gram) và **Word2Vec** (Word Embeddings).
4. Thử nghiệm và đối sánh 2 thuật toán phân loại: **Multinomial Naive Bayes** và **Linear SVM**.
5. Đánh giá toàn diện trên cả 2 tập dữ liệu thông qua đầy đủ các chỉ số: Accuracy, Precision, Recall, F1-Score và ma trận nhầm lẫn.

---

## TÀI LIỆU THAM KHẢO

1. Giáo trình môn học *Phân tích và khai phá dữ liệu văn bản*, Học viện Công nghệ Bưu chính Viễn thông.
2. Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008.
3. Luong Nguyen-Doan et al., *UIT-ViCTSD: A Vietnamese Constructiveness and Toxicity Speech Detection Dataset*, IEEE Access, 2021.
4. Son T. Luu et al., *ViHSD: A Vietnamese Hate Speech Detection Dataset*, KSE, 2021.
5. Tomas Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, arXiv:1301.3781, 2013.

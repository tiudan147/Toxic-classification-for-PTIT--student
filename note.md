1. Xử lý dữ liệu:
    - Dữ liệu train đã có:
        df_train_clean: UIT-ViCTSD (Vietnamese Constructiveness and Toxicity Dataset)
        vihsd_train: ViHSD (Vietnamese Hate Speech Detection)
    - Gộp lại dữ liệu lại: 
        - Volume dữ liệu là 16.683 dòng
        - Tỷ lệ dữ liệu hiện tại là ~70.8% Non-toxic và ~29.2% Toxic
    - Tách dữ liệu ra thành train/valid/test với tỷ lệ 70-15-15. Tỷ lệ toxic/non-toxic được dữ nguyên.
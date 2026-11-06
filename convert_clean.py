import pandas as pd

excel_path = r"D:\NguyenTienDan\HK 7\Khai_Pha_Van_Ban\BTL_1\dataset\gan_nhan_fb_scrape_1793870006344.xlsx"
df = pd.read_excel(excel_path, sheet_name="Gan_nhan")

lines = ["text,label\n"]
for _, row in df.iterrows():
    cmt = str(row["Bình luận"]).strip().replace("\r", " ").replace("\n", " ").replace('"', '""')
    lbl = int(row["Nhãn tay (0/1)"])
    lines.append(f'"{cmt}",{lbl}\n')

out_main = r"D:\NguyenTienDan\HK 7\Khai_Pha_Van_Ban\BTL_1\dataset\main_test.csv"
out_test = r"D:\NguyenTienDan\HK 7\Khai_Pha_Van_Ban\BTL_1\dataset\fb_test.csv"

with open(out_main, "w", encoding="utf-8") as f:
    f.writelines(lines)

with open(out_test, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Đã ghi thành công {len(lines)} dòng vào cả 2 file.")


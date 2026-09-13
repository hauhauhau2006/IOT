import glob
import os

import pandas as pd

# Đường dẫn đến thư mục chứa các file CSV lẻ
csv_folder = r'C:\Users\freed\IOT\data\Attack-traffic\csv'
output_file = r'C:\Users\freed\IOT\data\dataset_merged_raw.csv' # Lưu file tổng ra thư mục cha

# Lấy tất cả danh sách file .csv trong thư mục
all_csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
print(f'Tìm thấy {len(all_csv_files)} file CSV. Đang tiến hành gộp...')

if not all_csv_files:
    raise FileNotFoundError(f'Không tìm thấy file CSV nào trong thư mục: {csv_folder}')

df_list = []
for file_path in all_csv_files:
    print(f'-> Đang đọc file: {os.path.basename(file_path)}')
    df = pd.read_csv(file_path, low_memory=False)
    df_list.append(df)

# Gộp tất cả các bảng dữ liệu lại theo chiều dọc (kết hợp các dòng)
merged_df = pd.concat(df_list, ignore_index=True)

# Lưu thành 1 file CSV duy nhất
merged_df.to_csv(output_file, index=False)
print(f'\n✅ Đã gộp thành công {len(merged_df)} dòng dữ liệu vào: {output_file}')

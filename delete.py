import os

# Thư mục cần xử lý
folder_path = "daicwoz"

# Các định dạng được giữ lại
allowed_extensions = {".wav", ".csv"}  # thêm nếu transcript dùng định dạng khác

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
        _, ext = os.path.splitext(filename)
        
        if ext.lower() not in allowed_extensions:
            os.remove(file_path)
            print(f"Deleted: {filename}")
        else:
            print(f"Kept: {filename}")
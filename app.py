import streamlit as st
import os
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from pathlib import Path

# Tiêu đề cho ứng dụng Streamlit
st.title("Table Detection from Images")

# Đường dẫn tệp cấu hình và checkpoint của mô hình
config_file = './CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = './CascadeTabNet/epoch_36.pth'

# Khởi tạo mô hình với cấu hình và checkpoint
model = init_detector(config_file, checkpoint_file, device='cpu')  # Sử dụng 'cpu' nếu không có GPU

# Tải ảnh từ người dùng
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Tạo thư mục tạm để lưu ảnh
    directory = "tempDir"
    path = os.path.join(os.getcwd(), directory)
    p = Path(path)
    if not p.exists():
        os.mkdir(p)

    # Lưu ảnh đã tải lên vào thư mục tạm
    file_loc = os.path.join(path, uploaded_file.name)
    with open(file_loc, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Chạy mô hình để phát hiện và hiển thị kết quả
    result = inference_detector(model, file_loc)

    # Hiển thị kết quả trực tiếp mà không cần phải lưu vào file
    model.show_result(file_loc, result, ('Bordered', 'cell', 'Borderless'), score_thr=0.85, show=False)

    # Đọc và hiển thị ảnh kết quả từ mô hình
    result_img = Image.open(file_loc)  # Đọc lại ảnh đã được mô hình chỉnh sửa
    st.image(result_img, caption='Detection Result', use_column_width=True)

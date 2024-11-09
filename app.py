import streamlit as st
import mmcv
import os
import numpy as np
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from pathlib import Path
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Table Detection from Images")

config_file = './CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = './CascadeTabNet/epoch_36.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

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

    # Lưu ảnh đã tải lên
    file_loc = os.path.join(path, uploaded_file.name)
    with open(file_loc, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Chạy mô hình và hiển thị kết quả
    result = inference_detector(model, file_loc)

    # Hiển thị kết quả bằng matplotlib
    model.show_result(file_loc, result, ('Bordered', 'cell', 'Borderless'), score_thr=0.85, show=False,
                      out_file="result.jpg")

    # Đọc và hiển thị ảnh kết quả
    result_img = Image.open("result.jpg")
    st.image(result_img, caption='Detection Result', use_column_width=True)

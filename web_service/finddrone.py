import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(layout="wide", page_title="드론 탐지")

st.write("## 드론 탐지")
st.write(
    ":dog: 이미지를 업로드하면 드론을 찾아드립니다. :grin:"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    model = YOLO('../best.pt')

    # 이미지 읽기
    # Pillow 이미지를 OpenCV 형식으로 변환 (RGB -> BGR)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # YOLO 모델로 예측
    results = model.predict(source=frame, conf=0.6)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"conf: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    col2.write("Find drone :wrench:")
    image_processed = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    col2.image(image_processed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(image_processed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    fix_image("./drone.jpg")

import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration, VideoProcessorBase
import torch
from detect import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time
import av
from utils.turn import get_ice_servers
# import cv2
# import threading
# detected_labels = []

# CFG
# cfg_model_path = "models/MobileNetV3Small_rev1.pt"
cfg_model_path = "models/mobilenetv3s.pt"

cfg_enable_url_download = False
if cfg_enable_url_download:
    # Configure this if you set cfg_enable_url_download to True
    url = "https://archive.org/download/mobile-net-v-3-small-rev-1/MobileNetV3Small_rev1.pt"
    cfg_model_path = f"models/{url.split('/')[-1:][0]}"  # config model path from url name
# END OF CFG


def imageInput(device, src):

    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom',
                                   path=cfg_model_path, force_reload=True)
            model.cuda() if device == 'cuda' else model.cpu()
            model.conf = 0.7
            model.iou = 0.5
            model.max_det = 1
            pred = model(imgpath, size=256)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # --Display predicton

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

    elif src == 'From test set.':
        # Image selector slider
        imgpath = glob.glob('data/images/*')
        imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1)
        image_file = imgpath[imgsel - 1]
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                # call Model prediction--
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True)
                model.conf = 0.7
                model.iou = 0.5
                model.max_det = 1
                pred = model(image_file, size=256)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                # --Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction(s)')


# def videoInput(device, src):
#     uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
#     if uploaded_video != None:

#         ts = datetime.timestamp(datetime.now())
#         imgpath = os.path.join('data/uploads', str(ts) + uploaded_video.name)
#         outputpath = os.path.join('data/video_output', os.path.basename(imgpath))

#         with open(imgpath, mode='wb') as f:
#             f.write(uploaded_video.read())  # save video to disk

#         st_video = open(imgpath, 'rb')
#         video_bytes = st_video.read()
#         st.video(video_bytes)
#         st.write("Uploaded Video")
#         detect(weights=cfg_model_path, source=imgpath, device=0) if device == 'cuda' else detect(
#             weights=cfg_model_path, source=imgpath, device='cpu')
#         st_video2 = open(outputpath, 'rb')
#         video_bytes2 = st_video2.read()
#         st.video(video_bytes2)
#         st.write("Model Prediction")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=False)


class VideoProcessor:
    # def __init__(self):
    #     # self.lock = threading.Lock()
    #     self.labels = []

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(img)
        st.model.conf = 0.4
        st.model.iou = 0.4
        results = st.model(im_pil, size=256)
        bbox_img = np.array(results.render()[0])
        # labels = results.pandas().xyxy[0]["name"]

        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")


def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])

    option = st.sidebar.radio("Select input type.", ['Image', 'Webcam Stream'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled=False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled=True, index=0)
    # -- End of Sidebar

    st.header('🤚 Indonesian Sign Language Detector (SIBI & BISINDO)')
    st.subheader('👈🏽 Select options left-handed menu bar.')
    st.sidebar.markdown("https://github.com/fransachmadhw/yolov5_mobilenetv3s_streamlit")
    if option == "Image":
        imageInput(deviceoption, datasrc)
    # elif option == "Video":
    #     videoInput(deviceoption, datasrc)
    elif option == "Webcam Stream":
        # global video_processor
        # video_processor = VideoProcessor()
        # threading.Thread(target=update_labels, daemon=True).start()

        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_processor_factory=VideoProcessor,
        )

#         if webrtc_ctx.video_processor:
#             st.write("Detected Labels:")
#             for label in video_processor.labels:
#                 st.write(label)


if __name__ == '__main__':

    main()

# Downlaod Model from url.


# @st.cache
# def loadModel():
#     start_dl = time.time()
#     model_file = wget.download(url, out="models/")
#     finished_dl = time.time()
#     print(f"Model Downloaded, ETA:{finished_dl-start_dl}")


# if cfg_enable_url_download:
#     loadModel()

import streamlit as st
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import cv2
import tempfile

IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 20
MAX_DURATION = 60 
CLASSES_LIST = ["Real", "Fake"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('test_model.h5')

@st.cache_resource
def load_mtcnn():
    return MTCNN()

def preprocess_video(video_file):
    frames_list = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(video_file.read())
            video_reader = cv2.VideoCapture(tmpfile.name)
            detector = load_mtcnn()
            
            fps = video_reader.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            # cek video kalau lebih dari durasi
            if duration > MAX_DURATION:
                st.error(f"Durasi video terlalu lama")
                return None

            #
            frames_to_sample = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)

            # grid frame
            cols = st.columns(5)  # grid 5 kolom
            
            for i, frame_number in enumerate(frames_to_sample):
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = video_reader.read()
                if not success:
                    st.warning(f"Error reading frame {frame_number}. Skipping this frame.")
                    continue

                # deteksi wajah
                faces = detector.detect_faces(frame)
                if faces:
                    x, y, width, height = faces[0]['box']
                    face_frame = frame[y:y+height, x:x+width]
                    resized_frame = cv2.resize(face_frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                    normalized_frame = resized_frame / 255.0
                    frames_list.append(normalized_frame)

                # menampilkan frame
                with cols[i % 5]:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {i + 1}", width=150)

                # stop ketika sudah 20 frame
                if len(frames_list) == SEQUENCE_LENGTH:
                    break

            video_reader.release()

        # kalo ga cukup, nambah frame kosong
        while len(frames_list) < SEQUENCE_LENGTH:
            frames_list.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

        return np.array([frames_list])

    except Exception as e:
        st.error(f"Error {e}")
        return None

model = load_model()

st.title('Deepfake Detection Xception dan LSTM')

uploaded_file = st.file_uploader("Pilih video", type=['mp4'])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button('Deteksi Video'):
        with st.spinner('Memproses...'):
            preprocessed_video = preprocess_video(uploaded_file)
            
            if preprocessed_video is not None:
                prediction = model.predict(preprocessed_video)[0][0]
                
                # menghitung probabilitas
                fake_prob = prediction
                real_prob = 1 - prediction
                
                # hasil dari prediksi
                result = "Fake" if fake_prob > 0.5 else "Real"
                accuracy = fake_prob if result == "Fake" else real_prob

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"<h2 style='text-align: center;'>Hasil Deteksi</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>Video ini terindikasi sebagai: {result}</h3>", unsafe_allow_html=True)
                    # st.markdown(f"<h3 style='text-align: center;'>Akurasi: {accuracy:.2%}</h3>", unsafe_allow_html=True)
            else:
                st.warning("Proses video gagal. Silakan coba lagi dengan video lain.")

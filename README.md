# dog-breath-monitor
Analyze dog  breathinh rate from video
import streamlit as st
import cv2
import numpy as np
import tempfile
from scipy.signal import find_peaks, savgol_filter

st.title("Dog Breathing Monitor")

dog_name = st.text_input("Dog name")

limit_type = st.selectbox(
    "Choose breathing limit type",
    ["Range", "Maximum only", "Minimum only"]
)

min_rate = None
max_rate = None

if limit_type == "Range":
    min_rate = st.number_input(
        "Minimum breaths per minute",
        min_value=0.0,
        value=10.0
    )

    max_rate = st.number_input(
        "Maximum breaths per minute",
        min_value=0.0,
        value=12.0
    )

elif limit_type == "Maximum only":
    max_rate = st.number_input(
        "Maximum breaths per minute",
        min_value=0.0,
        value=20.0
    )

elif limit_type == "Minimum only":
    min_rate = st.number_input(
        "Minimum breaths per minute",
        min_value=0.0,
        value=8.0
    )

video_file = st.file_uploader(
    "Upload dog video",
    type=["mp4", "mov"]
)

if video_file is not None:
    st.video(video_file)

    if st.button("Analyze video"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_video:

            temp_video.write(video_file.read())
            video_path = temp_video.name

        video = cv2.VideoCapture(video_path)

        fps = video.get(cv2.CAP_PROP_FPS)

        success, frame = video.read()

        if not success:
            st.error("Could not read the video.")
            st.stop()

        # ตอนนี้ใช้ตรงกลางภาพเป็นบริเวณวิเคราะห์ก่อน
        height, width = frame.shape[:2]

        x1 = int(width * 0.25)
        x2 = int(width * 0.75)
        y1 = int(height * 0.25)
        y2 = int(height * 0.75)

        area = frame[y1:y2, x1:x2]
        previous = cv2.cvtColor(
            area,
            cv2.COLOR_BGR2GRAY
        )

        movement = []

        while True:
            success, frame = video.read()

            if not success:
                break

            area = frame[y1:y2, x1:x2]

            gray = cv2.cvtColor(
                area,
                cv2.COLOR_BGR2GRAY
            )

            difference = cv2.absdiff(
                previous,
                gray
            )

            movement.append(
                difference.mean()
            )

            previous = gray

        video.release()

        movement = np.array(movement)

        if len(movement) < 31:
            st.error("Video is too short for analysis.")
            st.stop()

        smooth = savgol_filter(
            movement,
            31,
            3
        )

        min_distance = int(fps * 3.0)

        peaks, _ = find_peaks(
            smooth,
            distance=min_distance,
            prominence=np.std(smooth) * 0.5
        )

        duration_seconds = len(movement) / fps

        breaths = len(peaks)

        breaths_per_minute = (
            breaths / duration_seconds
        ) * 60

        st.subheader("Result")

        st.write(
            "Dog:",
            dog_name
        )

        st.write(
            "Video duration:",
            round(duration_seconds, 1),
            "seconds"
        )

        st.write(
            "Breaths detected:",
            breaths
        )

        st.write(
            "Estimated breathing rate:",
            round(breaths_per_minute, 1),
            "breaths/min"
        )

        danger = False

        if limit_type == "Range":
            if (
                breaths_per_minute < min_rate
                or
                breaths_per_minute > max_rate
            ):
                danger = True

        elif limit_type == "Maximum only":
            if breaths_per_minute > max_rate:
                danger = True

        elif limit_type == "Minimum only":
            if breaths_per_minute < min_rate:
                danger = True

        if danger:
            st.error(
                "Run To Hospital Now or take a private jet. "
                "Follow your veterinarian's instructions."
            )
        else:
            st.success(
                "Within the vet-set range."
            )

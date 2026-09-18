import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from scipy.signal import butter, filtfilt, find_peaks

st.title("Dog Breathing Monitor")

st.write(
    "Use the breathing range provided by your veterinarian. "
    "Upload a stable 30–60 second video of your dog resting or sleeping."
)

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
        value=20.0
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
        value=10.0
    )

video_file = st.file_uploader(
    "Upload dog video",
    type=["mp4", "mov", "avi"]
)


def analyze_breathing(video_path):

    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        video.release()
        return None

    success, first_frame = video.read()

    if not success:
        video.release()
        return None

    height, width = first_frame.shape[:2]

    x1 = int(width * 0.25)
    x2 = int(width * 0.75)

    y1 = int(height * 0.25)
    y2 = int(height * 0.75)

    previous_gray = cv2.cvtColor(
        first_frame,
        cv2.COLOR_BGR2GRAY
    )

    breathing_signal = []
    valid_frames = 0
    camera_motion_history = []

    while True:

        success, frame = video.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        vertical_flow = flow[:, :, 1]

        camera_motion = np.median(
            vertical_flow
        )

        camera_motion_history.append(
            abs(camera_motion)
        )

        body_vertical_flow = vertical_flow[
            y1:y2,
            x1:x2
        ]

        body_motion = np.median(
            body_vertical_flow
        )

        corrected_motion = (
            body_motion - camera_motion
        )

        if abs(camera_motion) > 1.5:
            previous_gray = gray
            continue

        breathing_signal.append(
            corrected_motion
        )

        valid_frames += 1

        previous_gray = gray

    video.release()

    breathing_signal = np.array(
        breathing_signal,
        dtype=float
    )

    duration_seconds = total_frames / fps

    if len(breathing_signal) < 30:
        return None

    breathing_signal = (
        breathing_signal
        - np.mean(breathing_signal)
    )

    low_hz = 0.10
    high_hz = 1.00

    nyquist = fps / 2.0

    if high_hz >= nyquist:
        high_hz = nyquist * 0.8

    b, a = butter(
        3,
        [
            low_hz / nyquist,
            high_hz / nyquist
        ],
        btype="band"
    )

    try:
        filtered = filtfilt(
            b,
            a,
            breathing_signal
        )
    except:
        return None

    min_peak_distance = max(
        1,
        int(fps * 2.0)
    )

    prominence = max(
        np.std(filtered) * 0.5,
        0.0001
    )

    peaks, _ = find_peaks(
        filtered,
        distance=min_peak_distance,
        prominence=prominence
    )

    breaths_detected = len(peaks)

    breaths_per_minute = (
        breaths_detected
        / duration_seconds
        * 60
    )

    average_camera_motion = np.mean(
        camera_motion_history
    )

    rejected_ratio = 1 - (
        valid_frames
        /
        max(1, total_frames - 1)
    )

    if rejected_ratio > 0.30:
        quality = "Poor"

    elif average_camera_motion > 0.8:
        quality = "Fair"

    else:
        quality = "Good"

    return {
        "duration": duration_seconds,
        "breaths": breaths_detected,
        "bpm": breaths_per_minute,
        "quality": quality,
        "rejected_ratio": rejected_ratio
    }


if video_file is not None:

    st.video(video_file)

    st.info(
        "Keep the dog's stomach or belly area near the center of the video."
    )

    if st.button("Analyze video"):

        suffix = os.path.splitext(
            video_file.name
        )[1]

        if suffix == "":
            suffix = ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_video:

            temp_video.write(
                video_file.getbuffer()
            )

            video_path = temp_video.name

        with st.spinner("Analyzing breathing..."):

            result = analyze_breathing(
                video_path
            )

        try:
            os.remove(video_path)
        except:
            pass

        if result is None:

            st.error(
                "The video could not be analyzed reliably. "
                "Try a longer, clearer, and more stable video."
            )

        else:

            st.subheader("Result")

            st.write("Dog:", dog_name)

            st.write(
                "Video duration:",
                round(result["duration"], 1),
                "seconds"
            )

            st.write(
                "Breaths detected:",
                result["breaths"]
            )

            st.metric(
                "Estimated breathing rate",
                f'{result["bpm"]:.1f} breaths/min'
            )

            st.write(
                "Signal quality:",
                result["quality"]
            )

            st.write(
                "Frames rejected because of camera shake:",
                f'{result["rejected_ratio"] * 100:.1f}%'
            )

            bpm = result["bpm"]

            if result["quality"] == "Poor":

                st.warning(
                    "Video quality is too poor for a reliable result. "
                    "Record again with a more stable camera."
                )

            else:

                if limit_type == "Range":

                    st.write(
                        f"Vet-set range: "
                        f"{min_rate:.1f} - {max_rate:.1f} breaths/min"
                    )

                    if bpm < min_rate:

                        st.error(
                            f"LOW: {bpm:.1f} breaths/min. "
                            "This is below the vet-set range. "
                            "Verify manually and follow your veterinarian's instructions."
                        )

                    elif bpm > max_rate:

                        st.error(
                            f"HIGH: {bpm:.1f} breaths/min. "
                            "This is above the vet-set range. "
                            "Verify manually and follow your veterinarian's instructions."
                        )

                    else:

                        st.success(
                            f"WITHIN RANGE: {bpm:.1f} breaths/min"
                        )

                elif limit_type == "Maximum only":

                    st.write(
                        f"Maximum set by veterinarian: "
                        f"{max_rate:.1f} breaths/min"
                    )

                    if bpm > max_rate:

                        st.error(
                            f"HIGH: {bpm:.1f} breaths/min. "
                            "This is above the vet-set maximum."
                        )

                    else:

                        st.success(
                            f"WITHIN LIMIT: {bpm:.1f} breaths/min"
                        )

                elif limit_type == "Minimum only":

                    st.write(
                        f"Minimum set by veterinarian: "
                        f"{min_rate:.1f} breaths/min"
                    )

                    if bpm < min_rate:

                        st.error(
                            f"LOW: {bpm:.1f} breaths/min. "
                            "This is below the vet-set minimum."
                        )

                    else:

                        st.success(
                            f"WITHIN LIMIT: {bpm:.1f} breaths/min"
                        )

st.caption(
    "Test for Eric and Samsic. If you want to use this app for your dog.. you can try" 
    "and should not replace veterinary assessment."
)
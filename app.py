import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import pandas as pd
import json
import os



def compute_centroids(video_path: str, progress_callback=None):
    """Compute the centroid of the mouse in each frame using a simple background subtractor.

    Args:
        video_path (str): Path to the video file.
        progress_callback (callable, optional): Optional callback to update progress bar.

    Returns:
        list of tuples: A list where each element is (frame_idx, x, y) for the mouse centroid.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
    kernel = np.ones((3, 3), np.uint8)
    centroids = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        # reduce noise with morphological opening
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # threshold to binary image
        _, thr = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((idx, cx, cy))
        idx += 1
        if progress_callback:
            progress_callback(idx / total_frames)
    cap.release()
    return centroids



def auto_label_arms(centroids):
    """Automatically label arms based on centroid angles relative to the maze centre.

    Args:
        centroids (list of tuples): (frame_idx, x, y)

    Returns:
        list of str: A list of labels ('center', 'A', 'B', 'C') for each centroid.
    """
    if not centroids:
        return []
    coords = np.array([(c[1], c[2]) for c in centroids], dtype=float)
    centre = coords.mean(axis=0)
    # compute polar angle of each coordinate relative to the centre
    angles = np.arctan2(coords[:, 1] - centre[1], coords[:, 0] - centre[0])
    # cluster angles using kmeans on (cos, sin) features
    features = np.column_stack((np.cos(angles), np.sin(angles)))
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)
    arm_clusters = kmeans.labels_
    # compute radial distances to the centre
    distances = np.linalg.norm(coords - centre, axis=1)
    # threshold for centre region: small quantile of distances (0.1 quantile)
    radius = np.quantile(distances, 0.1)
    labels = []
    for i, (frame_idx, x, y) in enumerate(centroids):
        if distances[i] < radius:
            labels.append('center')
        else:
            # map numeric cluster to letters A, B, C consistently
            labels.append(chr(ord('A') + arm_clusters[i]))
    return labels



def compute_arm_entries(labels):
    """Count transitions from centre to any arm.

    Args:
        labels (list of str): labels for each frame ('center', 'A', 'B', 'C').

    Returns:
        list of str: Sequence of arm entries (e.g., ['A', 'B', 'C']).
    """
    if not labels:
        return []
    entries = []
    last_state = labels[0]
    for state in labels[1:]:
        if last_state == 'center' and state in {'A', 'B', 'C'}:
            entries.append(state)
        last_state = state
    return entries



def compute_spontaneous_alternation(entries):
    """Compute spontaneous alternation percentage.

    Args:
        entries (list of str): sequence of arm entries (e.g., ['A', 'B', 'C']).

    Returns:
        float: Alternation percentage.
    """
    if len(entries) < 3:
        return 0.0
    triad_count = 0
    unique_triads = 0
    for i in range(len(entries) - 2):
        triad = entries[i:i + 3]
        triad_count += 1
        if len(set(triad)) == 3:
            unique_triads += 1
    return (unique_triads / triad_count) * 100.0 if triad_count else 0.0



def main():
    st.set_page_config(page_title="Spontaneous Alternation Y/T-Maze", layout="wide")
    st.title("Spontaneous Alternation in Y/T-Maze")
    st.write(
        "This app automatically tracks a mouse in a Y or T maze from a top-down video and computes the spontaneous alternation percentage."
    )
    uploaded_file = st.file_uploader("Upload a Y/T-maze video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file:
        # Save to a temporary file for processing
        tfile_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(tfile_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.video(uploaded_file)
        if st.button("Run Analysis"):
            progress = st.progress(0)
            centroids = compute_centroids(tfile_path, progress_callback=progress.progress)
            progress.progress(1.0)
            if not centroids:
                st.error("No mouse detected.")
                return
            labels = auto_label_arms(centroids)
            entries = compute_arm_entries(labels)
            sa_percent = compute_spontaneous_alternation(entries)
            st.subheader("Results")
            st.write(f"Spontaneous Alternation: **{sa_percent:.2f}%**")
            st.write(f"Total arm entries: {len(entries)}")
            st.write("Arm entry sequence:", entries)
            # timeline plot
            df = pd.DataFrame({
                'Frame': [c[0] for c in centroids],
                'X': [c[1] for c in centroids],
                'Y': [c[2] for c in centroids],
                'State': labels
            })
            st.dataframe(df.head(20))
            # Download data
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            positions_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download positions CSV",
                data=positions_csv,
                file_name=f"positions_and_labels_{now}.csv",
                mime='text/csv'
            )
            entries_csv = pd.DataFrame({'Entry': entries}).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download arm entries CSV",
                data=entries_csv,
                file_name=f"arm_entries_{now}.csv",
                mime='text/csv'
            )
            summary = {
                'spontaneous_alternation_percent': sa_percent,
                'total_entries': len(entries),
                'entries': entries
            }
            summary_json = json.dumps(summary, indent=2).encode('utf-8')
            st.download_button(
                label="Download summary JSON",
                data=summary_json,
                file_name=f"summary_{now}.json",
                mime='application/json'
            )


if __name__ == "__main__":
    import tempfile
    main()

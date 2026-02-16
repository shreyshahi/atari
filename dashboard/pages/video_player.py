from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.utils import discover_runs

st.title("Video Player")
runs = discover_runs()
if not runs:
    st.info("No runs found in outputs/")
    st.stop()

run = st.selectbox("Run", runs, format_func=lambda p: p.name)
checkpoints = sorted((run / "checkpoints").glob("step_*"))
if not checkpoints:
    st.warning("No checkpoints found")
    st.stop()

ckpt = st.selectbox("Checkpoint", checkpoints, format_func=lambda p: p.name)
video = ckpt / "video.mp4"
if video.exists():
    st.video(str(video))
else:
    st.info("No video.mp4 for selected checkpoint")

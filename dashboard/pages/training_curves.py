from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.utils import discover_runs, load_csv

st.title("Training Curves")
runs = discover_runs()
if not runs:
    st.info("No runs found in outputs/")
    st.stop()

run = st.selectbox("Run", runs, format_func=lambda p: p.name)
train_df = load_csv(run / "train_log.csv")
if train_df.empty:
    st.warning("No train_log.csv found")
    st.stop()

metric = st.selectbox(
    "Metric",
    ["episode_return", "episode_length"],
)
fig = px.line(train_df, x="env_frames", y=metric, title=f"{metric} over env frames")
st.plotly_chart(fig, use_container_width=True)

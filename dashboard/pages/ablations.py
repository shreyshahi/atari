from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils import discover_runs, load_csv

st.title("Ablation Comparator")
runs = discover_runs()
if not runs:
    st.info("No runs found in outputs/")
    st.stop()

selected = st.multiselect("Runs", runs, format_func=lambda p: p.name)
if not selected:
    st.stop()

rows = []
for run in selected:
    df = load_csv(run / "eval_log.csv")
    if df.empty:
        continue
    full = df[df["eval_type"].isin(["full", "final"])]
    if full.empty:
        continue
    rows.append(pd.DataFrame({"run": run.name, "env_frames": full["env_frames"], "eval": full["eval_mean_return"]}))

if not rows:
    st.warning("No evaluation data found for selected runs")
    st.stop()

data = pd.concat(rows, ignore_index=True)
fig = px.line(data, x="env_frames", y="eval", color="run", title="Evaluation Progress")
st.plotly_chart(fig, use_container_width=True)

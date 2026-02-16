from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils import discover_runs, load_csv

st.title("Game Comparison")
runs = discover_runs()
if not runs:
    st.info("No runs found in outputs/")
    st.stop()

rows = []
for run in runs:
    df = load_csv(run / "eval_log.csv")
    if df.empty:
        continue
    full = df[df["eval_type"].isin(["full", "final"])]
    if full.empty:
        continue
    rows.append({"run": run.name, "best_eval_mean_return": full["eval_mean_return"].max()})

if not rows:
    st.warning("No evaluation data available")
    st.stop()

comp = pd.DataFrame(rows)
fig = px.bar(comp, x="run", y="best_eval_mean_return", title="Best Full Eval Return by Run")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(comp)

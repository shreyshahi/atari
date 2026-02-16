from __future__ import annotations

import json

import streamlit as st

from dashboard.utils import discover_runs

st.title("Config Diff Viewer")
runs = discover_runs()
if len(runs) < 2:
    st.info("Need at least two runs in outputs/")
    st.stop()

left = st.selectbox("Run A", runs, index=0, format_func=lambda p: p.name)
right = st.selectbox("Run B", runs, index=1, format_func=lambda p: p.name)

left_cfg = left / ".hydra" / "config.yaml"
right_cfg = right / ".hydra" / "config.yaml"

st.subheader("Run A config")
if left_cfg.exists():
    st.code(left_cfg.read_text(encoding="utf-8"), language="yaml")
else:
    st.warning("Run A config missing")

st.subheader("Run B config")
if right_cfg.exists():
    st.code(right_cfg.read_text(encoding="utf-8"), language="yaml")
else:
    st.warning("Run B config missing")

summary = {
    "run_a": left.name,
    "run_b": right.name,
    "note": "Visual side-by-side diff. Add structural YAML diff in future iteration.",
}
st.json(summary)

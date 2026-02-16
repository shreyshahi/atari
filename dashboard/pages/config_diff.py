from __future__ import annotations

import streamlit as st
from omegaconf import OmegaConf

from dashboard.utils import discover_runs


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.update(_flatten(v, key))
    else:
        out[prefix] = obj
    return out


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

if left_cfg.exists() and right_cfg.exists():
    left_dict = OmegaConf.to_container(OmegaConf.load(left_cfg), resolve=True)
    right_dict = OmegaConf.to_container(OmegaConf.load(right_cfg), resolve=True)
    left_flat = _flatten(left_dict)
    right_flat = _flatten(right_dict)

    all_keys = sorted(set(left_flat) | set(right_flat))
    rows = []
    for key in all_keys:
        lv = left_flat.get(key, "<missing>")
        rv = right_flat.get(key, "<missing>")
        if lv != rv:
            rows.append({"key": key, "run_a": lv, "run_b": rv})

    st.subheader("Changed Fields")
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No config differences found.")

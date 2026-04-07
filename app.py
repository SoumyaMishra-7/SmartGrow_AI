import io
import os
import sys
from contextlib import redirect_stdout

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

from inference import InferenceConfig, run_inference

st.set_page_config(page_title="Urban Micro-Farm AI", layout="centered")

st.title("Urban Micro-Farm AI Assistant")
st.write("AI-powered decision system for managing urban gardening resources.")

if "ran" not in st.session_state:
    st.session_state.ran = False
if "logs" not in st.session_state:
    st.session_state.logs = ""

if st.button("Run Simulation") and not st.session_state.ran:
    st.session_state.ran = True
    st.write("Running simulation...")

    config = InferenceConfig()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_inference(config)
    st.session_state.logs = buffer.getvalue()

    st.success("Simulation completed!")
    st.subheader("Execution Logs")
    st.code(st.session_state.logs, language="text")

if st.button("Reset"):
    st.session_state.ran = False
    st.session_state.logs = ""
    st.write("Reset complete")

if st.session_state.logs:
    st.subheader("Execution Logs")
    st.code(st.session_state.logs, language="text")

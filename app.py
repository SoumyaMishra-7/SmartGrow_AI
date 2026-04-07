import streamlit as st

from inference import InferenceConfig, run_inference


st.title("Plant Task Demo")

if st.button("Run Inference"):
    config = InferenceConfig()
    run_inference(config)
    st.success("Inference completed!")
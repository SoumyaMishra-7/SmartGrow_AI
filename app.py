from __future__ import annotations

import gradio as gr

from inference import InferenceConfig, run_inference


def run_app() -> str:
    config = InferenceConfig()
    run_inference(config)
    return "Inference completed successfully!"


iface = gr.Interface(
    fn=run_app,
    inputs=[],
    outputs="text",
    title="OpenEnv Plant Task Demo"
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
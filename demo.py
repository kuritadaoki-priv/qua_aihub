# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import onnxruntime as ort
from qai_hub_models.models._shared.detr.demo import detr_demo
from qai_hub_models.models.detr_resnet50.model import (
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DETRResNet50,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "detr_demo_image.jpg"
)

# NPU向けにONNX Runtimeを設定するための関数
def create_npu_session(model_path):
    # NPU (QNNExecutionProvider) を使用してONNXモデルをロードする
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["QNNExecutionProvider"],
        provider_options=[{"backend_path": "QnnHtp.dll"}]  # Adjust path if needed
    )
    return session

def main(is_test: bool = False):
    # Create NPU session
    session = create_npu_session("detr_resnet50.onnx")

    # Ensure the `detr_demo` function can accept and use this session if possible
    # If `detr_demo` does not accept a session parameter, you might need to modify it
    detr_demo(DETRResNet50, MODEL_ID, DEFAULT_WEIGHTS, IMAGE_ADDRESS, is_test, session=session)

if __name__ == "__main__":
    main()


"""Runtime phía board (KV260, A53): chỉ NumPy và OpenCV, Python 3.10, không PyTorch.

Luồng: khung camera -> preprocess -> infer (mạng) -> postprocess -> types.pack (gói perception_v0).
Chạy:  python -m runtime.pipeline --config config/pipeline_b.json --root THƯ_MỤC_CHỨA_GÓI --images 'anh/*.jpg'
"""

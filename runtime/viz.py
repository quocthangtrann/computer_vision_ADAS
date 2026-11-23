"""Vẽ kết quả lên khung camera để kiểm bằng mắt (cần OpenCV). Không dùng trong vòng lặp thời gian thực."""
import numpy as np

from runtime import types as T

COLORS = np.array([[0, 0, 0], [255, 160, 0], [0, 200, 0], [0, 120, 255]], np.uint8)     # RGB cho lớp 0..3


def overlay(frame_rgb, r: T.PerceptionResult, alpha=0.4):
    """frame_rgb (h, w, 3) uint8; r có mask (process(..., keep=True)). Trả về ảnh RGB đã vẽ."""
    import cv2
    out = frame_rgb.copy()
    if r.mask is not None:
        M = r.mask.grid_to_frame
        h, w = out.shape[:2]
        A = M[:2, :].astype(np.float32)          # lưới -> khung (nguồn -> đích): warpAffine tự lấy nghịch đảo; cùng quy ước tâm pixel
        col = cv2.warpAffine(COLORS[r.mask.m], A, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        lab = cv2.warpAffine(r.mask.m, A, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        sel = (lab > 0) & (lab < 4)
        out[sel] = (out[sel] * (1 - alpha) + col[sel] * alpha).astype(np.uint8)
    t2 = r.tier2
    for i in range(t2.n_cols):
        if t2.col_flags[i] & T.COL_VALID:
            cv2.circle(out, (int(round(t2.col_x[i])), int(round(t2.col_y[i]))), 5, (255, 0, 0), -1)
    for k in range(t2.n_rows):
        y = int(round(t2.row_y[k]))
        for v, c in ((t2.row_left[k], (255, 255, 0)), (t2.row_right[k], (255, 255, 0)), (t2.row_center[k], (255, 0, 255))):
            if not np.isnan(v):
                cv2.drawMarker(out, (int(round(v)), y), c, cv2.MARKER_CROSS, 18, 3)
    return out

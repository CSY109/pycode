import cv2
import numpy as np
import time
from collections import deque
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
    drawing_utils,
)
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

# ============================================================
# 手指关键点索引
# ============================================================
FINGERS = [
    {"name": "thumb",  "mcp": 2,  "pip": 3,  "dip": 3,  "tip": 4},
    {"name": "index",  "mcp": 5,  "pip": 6,  "dip": 7,  "tip": 8},
    {"name": "middle", "mcp": 9,  "pip": 10, "dip": 11, "tip": 12},
    {"name": "ring",   "mcp": 13, "pip": 14, "dip": 15, "tip": 16},
    {"name": "pinky",  "mcp": 17, "pip": 18, "dip": 19, "tip": 20},
]

# ============================================================
# 每指独立阈值 —— 适应不同手指的解剖差异
#   angle: 整体伸直度 MCP→PIP→TIP，越大越直
#   dist:  指尖到 MCP 距离，伸直远 / 弯曲近
#   dip:   DIP 关节最少角度
# ============================================================
FINGER_CFG = {
    "thumb":  {"angle": 145, "dist": 0.14, "dip": 130},
    "index":  {"angle": 148, "dist": 0.18, "dip": 138},
    "middle": {"angle": 148, "dist": 0.18, "dip": 138},
    "ring":   {"angle": 142, "dist": 0.16, "dip": 132},
    "pinky":  {"angle": 140, "dist": 0.14, "dip": 128},
}

DIST_FOLD    = 0.13    # MCP→TIP 小于此值视为明确弯曲
PALM_CENTER  = [0, 5, 9, 13, 17]

# ============================================================
# 系统参数
# ============================================================
DEBOUNCE_LEN = 2             # 防抖帧数（2帧确认，响应快）
SMOOTH_ALPHA = 0.65          # 关键点 EMA 平滑系数（0=不平滑, 1=完全平滑）
CAM_FPS      = 22
CAM_W, CAM_H = 640, 480


class SmoothedLM:
    """轻量容器：替代 NormalizedLandmark 用于平滑后的坐标"""
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y


def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    d = np.linalg.norm(ba) * np.linalg.norm(bc)
    if d == 0:
        return 0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / d, -1.0, 1.0)))


def lm_xy(lm):
    return (float(lm.x) if lm.x is not None else 0.0,
            float(lm.y) if lm.y is not None else 0.0)


def ema_smooth(raw_lms, prev_smoothed, alpha=SMOOTH_ALPHA):
    """EMA 平滑 21 个关键点坐标，返回 SmoothedLM 列表"""
    if prev_smoothed is None:
        return [SmoothedLM(*lm_xy(lm)) for lm in raw_lms]

    result = []
    for i, lm in enumerate(raw_lms):
        nx, ny = lm_xy(lm)
        ox, oy = prev_smoothed[i].x, prev_smoothed[i].y
        result.append(SmoothedLM(
            alpha * nx + (1 - alpha) * ox,
            alpha * ny + (1 - alpha) * oy,
        ))
    return result


def count_fingers(lms):
    """基于平滑后的关键点判断每根手指伸出/弯曲"""
    states = [False] * 5

    px = sum(lms[i].x for i in PALM_CENTER) / len(PALM_CENTER)
    py = sum(lms[i].y for i in PALM_CENTER) / len(PALM_CENTER)

    # ---- 拇指 ----
    cfg = FINGER_CFG["thumb"]
    cmc   = (lms[1].x, lms[1].y)
    t_mcp = (lms[2].x, lms[2].y)
    t_ip  = (lms[3].x, lms[3].y)
    t_tip = (lms[4].x, lms[4].y)
    idx5  = (lms[5].x, lms[5].y)

    a_mcp = calc_angle(cmc, t_mcp, t_ip)
    a_ip  = calc_angle(t_mcp, t_ip, t_tip)
    d_idx = np.hypot(t_tip[0] - idx5[0], t_tip[1] - idx5[1])

    d_palm_mcp = np.hypot(t_mcp[0] - px, t_mcp[1] - py)
    d_palm_tip = np.hypot(t_tip[0] - px, t_tip[1] - py)

    states[0] = (
        a_ip > cfg["angle"]
        and a_mcp > 138
        and d_idx > cfg["dist"]
        and d_palm_tip > d_palm_mcp
    )

    # ---- 四指 ----
    folded_count = 0
    for fi in [1, 2, 3, 4]:
        f = FINGERS[fi]
        cfg = FINGER_CFG[f["name"]]

        mcp = (lms[f["mcp"]].x, lms[f["mcp"]].y)
        pip = (lms[f["pip"]].x, lms[f["pip"]].y)
        dip = (lms[f["dip"]].x, lms[f["dip"]].y)
        tip = (lms[f["tip"]].x, lms[f["tip"]].y)

        a_overall = calc_angle(mcp, pip, tip)
        a_dip     = calc_angle(pip, dip, tip)
        d_mcp_tip = np.hypot(tip[0] - mcp[0], tip[1] - mcp[1])

        ext = (
            a_overall > cfg["angle"]
            and d_mcp_tip > cfg["dist"]
            and a_dip > cfg["dip"]
        )
        states[fi] = ext

        if not ext and d_mcp_tip < DIST_FOLD:
            folded_count += 1

    # 互锁
    if folded_count >= 4 and states[0]:
        states[0] = False

    return states


def status_text(n):
    if n == 0:
        return "Fist"
    if n == 5:
        return "Open Palm"
    return str(n)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    WIN = "Hand Detection"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 720)

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=r"e:\develop\pycode\hand_landmarker.task"),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.8,   # 提高：减少无手误检
        min_hand_presence_confidence=0.6,    # 略提：手在画面中更稳定
        min_tracking_confidence=0.6,         # 降低：快速移动时不易丢跟踪
    )
    landmarker = HandLandmarker.create_from_options(opts)

    t0 = time.time()
    running = True
    count_ring = deque(maxlen=DEBOUNCE_LEN)
    stable_count = 0
    smoothed = None  # 上一帧平滑后的 21 个关键点

    while running:
        ret, raw = cap.read()
        if not ret:
            break

        frame = cv2.resize(raw, (CAM_W, CAM_H))
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - t0) * 1000)
        result = landmarker.detect_for_video(mp_img, ts)

        h, w, _ = frame.shape
        raw_count = -1
        hand_active = bool(result.hand_landmarks)

        if hand_active:
            for hand_lm in result.hand_landmarks:
                # 画原始骨架（原始坐标，兼容 drawing_utils）
                drawing_utils.draw_landmarks(
                    frame, hand_lm,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

                # EMA 平滑用于手指判断
                smoothed = ema_smooth(hand_lm, smoothed)
                states = count_fingers(smoothed)
                raw_count = sum(states)

                # 指尖圆点（用平滑坐标）
                for i, f_cfg in enumerate(FINGERS):
                    lm_tip = smoothed[f_cfg["tip"]]
                    cx = int(lm_tip.x * w)
                    cy = int(lm_tip.y * h)
                    color = (0, 255, 0) if states[i] else (0, 0, 255)
                    cv2.circle(frame, (cx, cy), 8, color, -1)
        else:
            smoothed = None   # 手消失时重置平滑器

        # ---- 防抖 ----
        count_ring.append(raw_count)
        if hand_active and len(count_ring) == DEBOUNCE_LEN and len(set(count_ring)) == 1:
            stable_count = count_ring[0]
        elif not hand_active:
            stable_count = -1

        status = "No Hand" if stable_count == -1 else status_text(stable_count)

        # 底部状态栏
        bar = np.zeros((60, w, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)
        cv2.putText(bar, f"Fingers: {stable_count}  |  {status}",
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2)
        frame = np.vstack([frame, bar])

        cv2.imshow(WIN, frame)

        cv2.waitKey(max(1, 1000 // CAM_FPS))
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) <= 0:
                running = False
        except cv2.error:
            running = False

    # 清理
    try:
        landmarker.close()
    except Exception:
        pass
    try:
        cap.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()

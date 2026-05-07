import cv2
import numpy as np
import time
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
    drawing_utils,
)
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

FINGERS = [
    {"name": "thumb",  "mcp": 2,  "pip": 3,  "dip": 3,  "tip": 4},
    {"name": "index",  "mcp": 5,  "pip": 6,  "dip": 7,  "tip": 8},
    {"name": "middle", "mcp": 9,  "pip": 10, "dip": 11, "tip": 12},
    {"name": "ring",   "mcp": 13, "pip": 14, "dip": 15, "tip": 16},
    {"name": "pinky",  "mcp": 17, "pip": 18, "dip": 19, "tip": 20},
]

ANGLE_STRAIGHT = 145    # MCP→PIP→TIP 整体伸直阈值
DIST_EXTEND   = 0.17    # 指尖到 MCP 距离（伸直>此值）
DIST_FOLD     = 0.13    # MCP→TIP 距离小于此值视为弯曲
PALM_CENTER   = [0, 5, 9, 13, 17]  # 掌心参考点


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


def count_fingers(lms):
    states = [False] * 5
    dbg = {}

    # ---- 掌心位置 ----
    px = sum(lm_xy(lms[i])[0] for i in PALM_CENTER) / len(PALM_CENTER)
    py = sum(lm_xy(lms[i])[1] for i in PALM_CENTER) / len(PALM_CENTER)

    # ---- 拇指 ----
    cmc   = lm_xy(lms[1])
    t_mcp = lm_xy(lms[2])
    t_ip  = lm_xy(lms[3])
    t_tip = lm_xy(lms[4])
    idx5  = lm_xy(lms[5])

    a_mcp = calc_angle(cmc, t_mcp, t_ip)   # 拇指 MCP 关节
    a_ip  = calc_angle(t_mcp, t_ip, t_tip) # 拇指 IP 关节
    d_idx = np.hypot(t_tip[0] - idx5[0], t_tip[1] - idx5[1])

    # 拇指尖vs掌心：真正伸出时拇指尖比 MCP 离掌心更远
    d_palm_mcp = np.hypot(t_mcp[0] - px, t_mcp[1] - py)
    d_palm_tip = np.hypot(t_tip[0] - px, t_tip[1] - py)

    thumb_ok = a_ip > ANGLE_STRAIGHT and a_mcp > 140 and d_idx > 0.14 and d_palm_tip > d_palm_mcp
    states[0] = thumb_ok
    dbg["thumb"] = f"ip={a_ip:.0f} mcp={a_mcp:.0f} d={d_idx:.2f} pt={d_palm_tip:.2f}>{d_palm_mcp:.2f}"

    # ---- 四指（用整体伸直度代替拆分的 PIP/DIP） ----
    folded_count = 0
    for fi in [1, 2, 3, 4]:
        f = FINGERS[fi]
        mcp = lm_xy(lms[f["mcp"]])
        pip = lm_xy(lms[f["pip"]])
        dip = lm_xy(lms[f["dip"]])
        tip = lm_xy(lms[f["tip"]])

        # 整体伸直度：MCP→PIP→TIP 三点一线时角度接近 180°
        a_overall = calc_angle(mcp, pip, tip)
        a_dip     = calc_angle(pip, dip, tip)  # DIP 单独辅助判断
        d_tip     = np.hypot(tip[0] - mcp[0], tip[1] - mcp[1])

        dbg[f["name"]] = f"a={a_overall:.0f} dip={a_dip:.0f} d={d_tip:.2f}"

        # 伸直条件：整体直线度高 + 指尖离 MCP 足够远 + DIP 不能太弯
        extended = a_overall > ANGLE_STRAIGHT and d_tip > DIST_EXTEND and a_dip > 135
        states[fi] = extended

        if not extended and d_tip < DIST_FOLD:
            folded_count += 1

    # 互锁：四指全弯时拇指几乎不可能单独伸出
    if folded_count >= 4 and states[0]:
        states[0] = False

    return states, dbg


def status_text(n):
    if n == 0:
        return "Fist"
    if n == 5:
        return "Open Palm"
    return str(n)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    WIN = "Hand Detection"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 720)

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=r"e:\develop\pycode\hand_landmarker.task"),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.7,
    )
    landmarker = HandLandmarker.create_from_options(opts)
    t0 = time.time()
    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - t0) * 1000)
        result = landmarker.detect_for_video(mp_img, ts)

        count = 0
        status = "No Hand"
        debug_lines = []

        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                # 画骨架
                drawing_utils.draw_landmarks(
                    frame, hand_lm,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                    drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2),
                )

                states, angles = count_fingers(hand_lm)
                count = sum(states)
                status = status_text(count)
                debug_lines = [f"{k}: {v}" for k, v in angles.items()]

                h, w, _ = frame.shape

                # 每根手指的 MCP 和 TIP 标编号
                for fi in [0, 1, 2, 3, 4]:
                    f = FINGERS[fi]
                    for label, idx in [("M", f["mcp"]), ("T", f["tip"])]:
                        lm = hand_lm[idx]
                        cx = int((lm.x if lm.x is not None else 0) * w)
                        cy = int((lm.y if lm.y is not None else 0) * h)
                        cv2.putText(frame, f"{fi}{label}", (cx + 10, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # 指尖圆点：绿色伸出，红色弯曲
                for i, f_cfg in enumerate(FINGERS):
                    lm_tip = hand_lm[f_cfg["tip"]]
                    cx = int((lm_tip.x if lm_tip.x is not None else 0) * w)
                    cy = int((lm_tip.y if lm_tip.y is not None else 0) * h)
                    color = (0, 255, 0) if states[i] else (0, 0, 255)
                    cv2.circle(frame, (cx, cy), 8, color, -1)

        # ---- 右侧调试面板 ----
        h, w, _ = frame.shape
        if debug_lines:
            panel_w = 200
            canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
            canvas[0:h, 0:w] = frame
            cv2.rectangle(canvas, (w, 0), (w + panel_w, h), (30, 30, 30), -1)
            y = 25
            for line in debug_lines:
                cv2.putText(canvas, line, (w + 8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
                y += 20
            frame = canvas
            w += panel_w

        # ---- 底部状态栏 ----
        bar = np.zeros((60, w, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)
        cv2.putText(bar, f"Fingers: {count}  |  {status}",
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2)
        frame = np.vstack([frame, bar])

        cv2.imshow(WIN, frame)

        cv2.waitKey(1)
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) <= 0:
                running = False
        except cv2.error:
            running = False

    # 清理——不阻塞，避免卡住
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

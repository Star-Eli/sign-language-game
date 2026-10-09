import cv2
import mediapipe as mp
import numpy as np
import json
import os
import math
import winsound

# ============================================================
#   TEACH SCREEN  -  "Let's learn X"
# ============================================================

BACKROUND_PATH = r"C:\Users\Elias\letters\backround.jpeg"
LETTER_IMG_DIR = r"C:\Users\Elias\letters"
SESSION_DIR    = r"C:\Users\Elias\sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

C_CARD        = (255, 255, 255)
C_TEXT_DARK   = (45, 35, 50)
C_TEXT_SOFT   = (140, 130, 150)
C_TEXT_WHITE  = (255, 255, 255)
C_MAGENTA     = (200, 40, 230)
C_MAGENTA_D   = (150, 25, 175)
C_TEAL        = (200, 180, 40)
C_GREEN       = (110, 200, 110)
C_HAND        = (60, 160, 240)

WINDOW_NAME = "Sign Language Adventure"
BASE_W, BASE_H = 1280, 720

_bg_img = cv2.imread(BACKROUND_PATH)
if _bg_img is None:
    _bg_img = np.full((720, 546, 3), (200, 40, 230), dtype=np.uint8)
else:
    print(f"✓ Background loaded: {_bg_img.shape[1]}x{_bg_img.shape[0]}")

def get_background(w, h):
    bh, bw = _bg_img.shape[:2]
    scale = max(w / bw, h / bh)
    new_w, new_h = int(bw * scale), int(bh * scale)
    resized = cv2.resize(_bg_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    x_off = (new_w - w) // 2
    y_off = (new_h - h) // 2
    return resized[y_off:y_off + h, x_off:x_off + w].copy()

# ============================================================
#   HELPERS
# ============================================================

def rounded_rect(img, x1, y1, x2, y2, colour, radius=16):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    radius = int(radius)
    if x2 <= x1 or y2 <= y1:
        return
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), colour, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), colour, -1)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, colour, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, colour, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, colour, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, colour, -1)

def centred_text(img, text, y, scale, colour, thickness=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)
    return tw, th

def pill_text(img, text, cy, scale, colour, thickness=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cx = img.shape[1] // 2
    x1 = cx - tw // 2 - 22
    x2 = cx + tw // 2 + 22
    y1 = cy - th // 2 - 12
    y2 = cy + th // 2 + 12
    rounded_rect(img, x1, y1, x2, y2, C_CARD, radius=14)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2),
                font, scale, colour, thickness, cv2.LINE_AA)

def draw_hand(canvas, cx, cy, size, angle, wave_phase):
    parts = []
    palm_r = int(size * 0.42)
    parts.append((0, 0, palm_r, C_HAND))
    finger_len = size * 0.55
    finger_r = int(size * 0.11)
    for i, bx in enumerate([-0.30, -0.10, 0.10, 0.30]):
        wave = math.sin(wave_phase + i * 0.8) * size * 0.06
        fx = bx * size + wave
        fy = -finger_len * 0.9 - abs(math.sin(wave_phase + i) * size * 0.05)
        parts.append((fx, fy, finger_r, C_HAND))
        parts.append((fx, fy - finger_len * 0.35, int(finger_r * 0.9), C_HAND))
    ta = -0.9 + math.sin(wave_phase * 0.7) * 0.15
    tx = math.cos(ta) * size * 0.55 - size * 0.15
    ty = math.sin(ta) * size * 0.55 + size * 0.15
    parts.append((tx, ty, int(finger_r * 1.1), C_HAND))

    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    overlay = canvas.copy()
    for (ox, oy, r, colour) in parts:
        rx = ox * ca - oy * sa
        ry = ox * sa + oy * ca
        cv2.circle(overlay, (int(cx + rx), int(cy + ry)), r, colour, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)

def beep_click():
    try: winsound.Beep(880, 55)
    except Exception: pass

def beep_saved():
    try:
        winsound.Beep(880, 70); winsound.Beep(1320, 110)
    except Exception: pass

# ============================================================
#   MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

def read_hand(frame):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    landmarks = None
    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            frame, hand_lms, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(220, 220, 220), thickness=1)
        )
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
    return frame, landmarks

def draw_camera_card(canvas, frame, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    fh, fw = frame.shape[:2]
    scale = max(w / fw, h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    ox = (new_w - w) // 2
    oy = (new_h - h) // 2
    crop = resized[oy:oy + h, ox:ox + w]
    canvas[y1:y2, x1:x2] = crop
    cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), C_CARD, 4)

def draw_letter_card(canvas, letter_img, x1, y1, x2, y2):
    rounded_rect(canvas, x1, y1, x2, y2, C_CARD, radius=22)
    if letter_img is None:
        return
    lh, lw = letter_img.shape[:2]
    avail_w = x2 - x1 - 30
    avail_h = y2 - y1 - 30
    scale = min(avail_w / lw, avail_h / lh)
    new_w, new_h = int(lw * scale), int(lh * scale)
    resized = cv2.resize(letter_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    px = x1 + (x2 - x1 - new_w) // 2
    py = y1 + (y2 - y1 - new_h) // 2
    canvas[py:py + new_h, px:px + new_w] = resized

# ============================================================
#   TEACH SCREEN
# ============================================================

def teach_screen(name, letters):
    idx = 0
    win_w, win_h = BASE_W, BASE_H
    cached_bg = None
    cached_size = None
    start_time = cv2.getTickCount() / cv2.getTickFrequency()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, BASE_W, BASE_H)

    while idx < len(letters):
        target = letters[idx]

        lp = os.path.join(LETTER_IMG_DIR, f"{target}.png")
        letter_img = cv2.imread(lp) if os.path.exists(lp) else None

        saved = False
        while not saved:
            t = (cv2.getTickCount() / cv2.getTickFrequency()) - start_time
            try:
                rect = cv2.getWindowImageRect(WINDOW_NAME)
                rw, rh = rect[2], rect[3]
                if rw > 0 and rh > 0:
                    win_w, win_h = rw, rh
            except Exception:
                pass

            if cached_size != (win_w, win_h):
                cached_bg = get_background(win_w, win_h)
                cached_size = (win_w, win_h)

            canvas = cached_bg.copy()

            ret, frame = cap.read()
            if not ret:
                break
            frame, landmarks = read_hand(frame)

            # --- Title pill: "Let's learn D" ---
            title = f"Let's learn {target}"
            ts = win_w / 1100
            pill_text(canvas, title, int(win_h * 0.08), ts, C_TEXT_DARK, 3)

            # --- Waving hand top-left ---
            draw_hand(canvas, int(win_w * 0.06), int(win_h * 0.08),
                      int(win_h * 0.09), math.sin(t * 3.0) * 12, t * 4.0)

            # --- Progress dots ---
            total = len(letters)
            dot_r = int(win_h * 0.022)
            gap = dot_r * 3
            total_w = total * gap
            start_x = (win_w - total_w) // 2 + gap // 2
            for i, L in enumerate(letters):
                cx = start_x + i * gap
                cy = int(win_h * 0.16)
                if i < idx:
                    colour = C_GREEN
                    label = "✓"
                elif i == idx:
                    colour = C_MAGENTA
                    label = L
                else:
                    colour = (220, 215, 225)
                    label = L
                cv2.circle(canvas, (cx, cy), dot_r, colour, -1, cv2.LINE_AA)
                font = cv2.FONT_HERSHEY_DUPLEX
                fs = win_w / 1800
                (tw, th), _ = cv2.getTextSize(label, font, fs, 2)
                cv2.putText(canvas, label,
                            (cx - tw // 2, cy + th // 2),
                            font, fs, C_TEXT_WHITE, 2, cv2.LINE_AA)

            # --- Letter card (left) ---
            card_x1 = int(win_w * 0.06)
            card_y1 = int(win_h * 0.26)
            card_x2 = int(win_w * 0.46)
            card_y2 = int(win_h * 0.86)
            draw_letter_card(canvas, letter_img, card_x1, card_y1, card_x2, card_y2)

            # --- Camera card (right) ---
            cam_x1 = int(win_w * 0.52)
            cam_y1 = int(win_h * 0.26)
            cam_x2 = int(win_w * 0.94)
            cam_y2 = int(win_h * 0.86)
            draw_camera_card(canvas, frame, cam_x1, cam_y1, cam_x2, cam_y2)

            # --- Hand status pill ---
            status = "Hand detected!" if landmarks else "Show your hand!"
            scolour = C_GREEN if landmarks else C_MAGENTA
            (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, win_w/2000, 2)
            px1 = cam_x1 + (cam_x2 - cam_x1 - sw) // 2 - 15
            py1 = cam_y2 + 8
            rounded_rect(canvas, px1, py1, px1 + sw + 30, py1 + sh + 20, scolour, radius=12)
            cv2.putText(canvas, status,
                        (px1 + 15, py1 + sh + 6),
                        cv2.FONT_HERSHEY_DUPLEX, win_w/2000, C_TEXT_WHITE, 2, cv2.LINE_AA)

            # --- Instructions bottom: "Copy the shape, then press SPACE" ---
            instr = "Copy the shape, then press SPACE"
            (iw, ih), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_DUPLEX, win_w/2000, 2)
            rounded_rect(canvas, (win_w - iw) // 2 - 25, win_h - 70,
                         (win_w + iw) // 2 + 25, win_h - 25, C_CARD, radius=14)
            cv2.putText(canvas, instr,
                        ((win_w - iw) // 2, win_h - 38),
                        cv2.FONT_HERSHEY_DUPLEX, win_w/2000, C_TEXT_DARK, 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):
                return False
            elif key == ord('s'):
                saved = True
            elif key == ord(' '):
                if landmarks is None:
                    continue
                with open(os.path.join(SESSION_DIR, f"{name}_{target}.json"), "w") as f:
                    json.dump({"name": name, "letter": target,
                               "landmarks": landmarks}, f)
                beep_saved()
                flash = canvas.copy()
                overlay = np.full_like(flash, 0, dtype=np.uint8)
                overlay[:] = C_GREEN
                blended = cv2.addWeighted(flash, 0.55, overlay, 0.45, 0)
                centred_text(blended, f"✓ Got {target}!", int(win_h * 0.5),
                             win_w / 900, C_TEXT_WHITE, 5)
                cv2.imshow(WINDOW_NAME, blended)
                cv2.waitKey(400)
                saved = True
        idx += 1

    return True

# ============================================================
#   RUN
# ============================================================

print("Testing teach screen with name DOG...")
ok = teach_screen("DOG", ["D", "O", "G"])
print("Teach complete!" if ok else "Exited.")
cap.release()
cv2.destroyAllWindows()
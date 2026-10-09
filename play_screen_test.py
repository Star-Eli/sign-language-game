import cv2
import mediapipe as mp
import numpy as np
import json
import os
import math
import winsound
import time

# ============================================================
#   PLAY SCREEN  -  picture shows ONLY after correct / final fail
# ============================================================

BACKROUND_PATH = r"C:\Users\Elias\letters\backround.jpeg"
LETTER_IMG_DIR = r"C:\Users\Elias\letters"
SESSION_DIR    = r"C:\Users\Elias\sessions"

MATCH_THRESHOLD = 0.25
MATCH_MARGIN    = 0.05
HOLD_TIME       = 1.0
MAX_ATTEMPTS    = 5
POINTS_TABLE    = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2}

C_CARD        = (255, 255, 255)
C_TEXT_DARK   = (45, 35, 50)
C_TEXT_SOFT   = (140, 130, 150)
C_TEXT_WHITE  = (255, 255, 255)
C_MAGENTA     = (200, 40, 230)
C_MAGENTA_D   = (150, 25, 175)
C_TEAL        = (200, 180, 40)
C_GREEN       = (110, 200, 110)
C_RED         = (80, 80, 220)
C_YELLOW      = (60, 200, 240)
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
    rounded_rect(img, cx - tw // 2 - 22, cy - th // 2 - 12,
                 cx + tw // 2 + 22, cy + th // 2 + 12, C_CARD, radius=14)
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

def beep_right():
    try:
        winsound.Beep(880, 70); winsound.Beep(1320, 110)
    except Exception: pass

def beep_wrong():
    try: winsound.Beep(300, 150)
    except Exception: pass

def beep_skip():
    try: winsound.Beep(440, 200)
    except Exception: pass

def normalize(landmarks):
    wx, wy, wz = landmarks[0]
    pts = [(x - wx, y - wy, z - wz) for (x, y, z) in landmarks]
    max_d = max(math.hypot(p[0], p[1]) for p in pts) or 1.0
    return [(p[0] / max_d, p[1] / max_d, p[2] / max_d) for p in pts]

def distance(a, b):
    total = 0.0
    for (x1, y1, z1), (x2, y2, z2) in zip(a, b):
        total += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return total / len(a)

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

def ordinal(n):
    if n == 1: return "1st"
    if n == 2: return "2nd"
    if n == 3: return "3rd"
    return f"{n}th"

def draw_letter_on_card(canvas, letter_img, x1, y1, x2, y2):
    rounded_rect(canvas, x1, y1, x2, y2, C_CARD, radius=22)
    if letter_img is None:
        return
    lh, lw = letter_img.shape[:2]
    avail_w = x2 - x1 - 40
    avail_h = y2 - y1 - 40
    scale = min(avail_w / lw, avail_h / lh)
    nw, nh = int(lw * scale), int(lh * scale)
    rz = cv2.resize(letter_img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    px = x1 + (x2 - x1 - nw) // 2
    py = y1 + (y2 - y1 - nh) // 2
    canvas[py:py + nh, px:px + nw] = rz

def show_result(canvas, win_w, win_h, letter_img, card_box,
                header, colour, subtitle=None):
    """Draw the letter picture on the card, put the header text on top."""
    x1, y1, x2, y2 = card_box
    draw_letter_on_card(canvas, letter_img, x1, y1, x2, y2)
    centred_text(canvas, header, int(win_h * 0.32),
                 win_w / 900, colour, 5)
    if subtitle:
        centred_text(canvas, subtitle, int(win_h * 0.40),
                     win_w / 1400, C_TEXT_DARK, 3)

def play_screen(name, letters):
    refs = {}
    for L in set(letters):
        p = os.path.join(SESSION_DIR, f"{name}_{L}.json")
        if not os.path.exists(p):
            print(f"❌ Missing taught file for {L}")
            return None
        with open(p) as f:
            refs[L] = normalize(json.load(f)["landmarks"])

    win_w, win_h = BASE_W, BASE_H
    cached_bg = None
    cached_size = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, BASE_W, BASE_H)

    anim_start = cv2.getTickCount() / cv2.getTickFrequency()

    current_idx = 0
    hold_start  = None
    score       = 0
    start_time_p = time.time()
    attempt     = 1
    feedback    = "Make the shape, then press SPACE"
    feedback_colour = C_MAGENTA
    allow_space = True

    while current_idx < len(letters):
        t = (cv2.getTickCount() / cv2.getTickFrequency()) - anim_start
        target = letters[current_idx]

        # load letter image (only used when showing result)
        lp = os.path.join(LETTER_IMG_DIR, f"{target}.png")
        letter_img = cv2.imread(lp) if os.path.exists(lp) else None

        ret, frame = cap.read()
        if not ret:
            return None
        frame, landmarks = read_hand(frame)

        is_match = False
        cl_letter, cl_dist = None, None
        if landmarks:
            live_norm = normalize(landmarks)
            distances = sorted(
                [(L, distance(live_norm, refs[L])) for L in refs],
                key=lambda x: x[1]
            )
            cl_letter, cl_dist = distances[0]
            rn_dist = distances[1][1] if len(distances) > 1 else 999
            is_match = (cl_letter == target
                        and cl_dist < MATCH_THRESHOLD
                        and (rn_dist - cl_dist) > MATCH_MARGIN)

        # ---------- resolve state changes ----------
        if is_match:
            if hold_start is None:
                hold_start = time.time()
            held = time.time() - hold_start
            if held >= HOLD_TIME:
                points = POINTS_TABLE.get(attempt, 0)
                score += points
                beep_right()
                # build result screen
                try:
                    rect = cv2.getWindowImageRect(WINDOW_NAME)
                    rw, rh = rect[2], rect[3]
                    if rw > 0 and rh > 0:
                        win_w, win_h = rw, rh
                except Exception:
                    pass
                canvas = cached_bg.copy() if cached_bg is not None else get_background(win_w, win_h)
                card_box = (int(win_w*0.06), int(win_h*0.26),
                            int(win_w*0.46), int(win_h*0.86))
                show_result(canvas, win_w, win_h, letter_img, card_box,
                            f"✓ {target} correct!",
                            C_GREEN, f"+{points} points")
                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(1200)
                current_idx += 1
                attempt = 1
                hold_start = None
                feedback = "Make the shape, then press SPACE"
                feedback_colour = C_MAGENTA
                continue
            else:
                feedback = f"Hold it... {held:.1f}s"
                feedback_colour = C_GREEN
        else:
            hold_start = None
            if not landmarks:
                feedback = "Show your hand!"
                feedback_colour = C_MAGENTA
            else:
                feedback = f"Attempt {attempt}/{MAX_ATTEMPTS} — press SPACE when ready"
                feedback_colour = C_MAGENTA

        # ---------- draw guessing screen ----------
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

        pill_text(canvas, f"Spell: {name}", int(win_h * 0.08),
                  win_w / 1100, C_TEXT_DARK, 3)

        draw_hand(canvas, int(win_w * 0.06), int(win_h * 0.08),
                  int(win_h * 0.09), math.sin(t * 3.0) * 12, t * 4.0)

        # Progress dots
        dot_r = int(win_h * 0.022)
        gap = dot_r * 3
        total_w = len(letters) * gap
        start_x = (win_w - total_w) // 2 + gap // 2
        for i in range(len(letters)):
            cx = start_x + i * gap
            cy = int(win_h * 0.16)
            if i < current_idx:
                colour = C_GREEN; label = "✓"
            elif i == current_idx:
                colour = C_MAGENTA; label = "?"
            else:
                colour = (220, 215, 225); label = "?"
            cv2.circle(canvas, (cx, cy), dot_r, colour, -1, cv2.LINE_AA)
            fs = win_w / 1600
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
            cv2.putText(canvas, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_DUPLEX, fs, C_TEXT_WHITE, 2, cv2.LINE_AA)

        # LEFT CARD — guessing state: just text, no picture
        card_x1 = int(win_w * 0.06)
        card_y1 = int(win_h * 0.26)
        card_x2 = int(win_w * 0.46)
        card_y2 = int(win_h * 0.86)
        rounded_rect(canvas, card_x1, card_y1, card_x2, card_y2, C_CARD, radius=22)

        line1 = f"Make the"
        line2 = f"{ordinal(current_idx + 1)} letter"
        line3 = f"of {name}"
        centre_card_x = (card_x1 + card_x2) // 2
        fs_l = win_w / 1400
        font = cv2.FONT_HERSHEY_DUPLEX
        for txt, off in [(line1, -60), (line2, 0), (line3, 60)]:
            (tw, th), _ = cv2.getTextSize(txt, font, fs_l, 2)
            cv2.putText(canvas, txt,
                        (centre_card_x - tw // 2,
                         (card_y1 + card_y2) // 2 + th // 2 + off),
                        font, fs_l, C_TEXT_DARK, 2, cv2.LINE_AA)

        # Camera card
        cam_x1 = int(win_w * 0.52)
        cam_y1 = int(win_h * 0.26)
        cam_x2 = int(win_w * 0.94)
        cam_y2 = int(win_h * 0.86)
        draw_camera_card(canvas, frame, cam_x1, cam_y1, cam_x2, cam_y2)

        # Feedback pill under camera
        (fw, fh), _ = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_DUPLEX, win_w/2100, 2)
        px1 = cam_x1 + (cam_x2 - cam_x1 - fw) // 2 - 15
        py1 = cam_y2 + 8
        rounded_rect(canvas, px1, py1, px1 + fw + 30, py1 + fh + 20,
                     feedback_colour, radius=12)
        cv2.putText(canvas, feedback, (px1 + 15, py1 + fh + 6),
                    cv2.FONT_HERSHEY_DUPLEX, win_w/2100, C_TEXT_WHITE, 2, cv2.LINE_AA)

        # Score pill
        sc_text = f"Score: {score}"
        (sw, sh), _ = cv2.getTextSize(sc_text, cv2.FONT_HERSHEY_DUPLEX, win_w/2000, 2)
        rounded_rect(canvas, win_w - sw - 60, win_h - 70,
                     win_w - 20, win_h - 25, C_CARD, radius=12)
        cv2.putText(canvas, sc_text, (win_w - sw - 40, win_h - 38),
                    cv2.FONT_HERSHEY_DUPLEX, win_w/2000, C_TEXT_DARK, 2, cv2.LINE_AA)

        # Attempt pill
        ap_text = f"Attempt {attempt}/{MAX_ATTEMPTS}"
        (aw, ah), _ = cv2.getTextSize(ap_text, cv2.FONT_HERSHEY_DUPLEX, win_w/2200, 2)
        rounded_rect(canvas, 20, win_h - 70, 20 + aw + 30, win_h - 25, C_CARD, radius=12)
        cv2.putText(canvas, ap_text, (35, win_h - 38),
                    cv2.FONT_HERSHEY_DUPLEX, win_w/2200, C_TEXT_DARK, 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            return None

        if key == ord(' ') and allow_space:
            allow_space = False
            if is_match:
                points = POINTS_TABLE.get(attempt, 0)
                score += points
                beep_right()
                canvas = cached_bg.copy() if cached_bg is not None else get_background(win_w, win_h)
                card_box = (card_x1, card_y1, card_x2, card_y2)
                show_result(canvas, win_w, win_h, letter_img, card_box,
                            f"✓ {target} correct!", C_GREEN,
                            f"+{points} points")
                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(1200)
                current_idx += 1
                attempt = 1
                feedback = "Make the shape, then press SPACE"
                feedback_colour = C_MAGENTA
                continue
            else:
                if attempt >= MAX_ATTEMPTS:
                    # Show the answer on the card, with "Moving on" text
                    beep_skip()
                    canvas = cached_bg.copy() if cached_bg is not None else get_background(win_w, win_h)
                    card_box = (card_x1, card_y1, card_x2, card_y2)
                    show_result(canvas, win_w, win_h, letter_img, card_box,
                                f"It was {target}", C_YELLOW,
                                f"Moving on — no points")
                    cv2.imshow(WINDOW_NAME, canvas)
                    cv2.waitKey(1400)
                    current_idx += 1
                    attempt = 1
                    feedback = "Make the shape, then press SPACE"
                    feedback_colour = C_MAGENTA
                    continue
                else:
                    attempt += 1
                    beep_wrong()
                    nxt = POINTS_TABLE.get(attempt, 0)
                    feedback = f"Attempt {attempt}/{MAX_ATTEMPTS} — {nxt} points available"
                    feedback_colour = C_RED
                    # quick red flash without picture
                    flash = canvas.copy()
                    overlay = np.full_like(flash, 0, dtype=np.uint8)
                    overlay[:] = C_RED
                    blended = cv2.addWeighted(flash, 0.6, overlay, 0.4, 0)
                    centred_text(blended, "Not quite", int(win_h * 0.5),
                                 win_w / 900, C_TEXT_WHITE, 6)
                    cv2.imshow(WINDOW_NAME, blended)
                    cv2.waitKey(500)

        if key != ord(' '):
            allow_space = True

    elapsed = time.time() - start_time_p
    return {"score": score, "time": elapsed}

# ============================================================
#   RUN TEST
# ============================================================

print("Testing play screen with name DOG...")
result = play_screen("DOG", ["D", "O", "G"])
if result:
    print(f"🎉 Done! Score: {result['score']}  Time: {result['time']:.1f}s")
cap.release()
cv2.destroyAllWindows()
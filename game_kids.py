import cv2
import mediapipe as mp
import numpy as np
import json
import os
import math
import winsound
import time
import random
import glob

# ============================================================
#   SIGN LANGUAGE ADVENTURE  -  full game
#   name -> teach -> play -> celebrate -> replay / new / fresh
# ============================================================

APP_DIR        = r"C:\Users\Elias\Documents\CVTS\SignLangaugeApp"
BACKROUND_PATH = r"C:\Users\Elias\Documents\CVTS\SignLangaugeApp\letters\backround.jpeg"
LETTER_IMG_DIR = r"C:\Users\Elias\Documents\CVTS\SignLangaugeApp\letters"
SESSION_DIR    = r"C:\Users\Elias\Documents\CVTS\SignLangaugeApp\sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

MATCH_THRESHOLD = 0.25
MATCH_MARGIN    = 0.05
HOLD_TIME       = 1.0
MAX_ATTEMPTS    = 5
POINTS_TABLE    = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2}

# --- PALETTE (BGR) ---
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
C_GREY_BTN    = (235, 235, 240)
C_HAND        = (60, 160, 240)

WINDOW_NAME = "Sign Language Adventure"
BASE_W, BASE_H = 1280, 720

# ============================================================
#   BACKGROUND
# ============================================================

_bg_img = cv2.imread(BACKROUND_PATH)
if _bg_img is None:
    _bg_img = np.full((720, 546, 3), (200, 40, 230), dtype=np.uint8)
    print("Background image not found - using magenta fallback")
else:
    print(f"Background loaded: {_bg_img.shape[1]}x{_bg_img.shape[0]}")

def get_background(w, h):
    bh, bw = _bg_img.shape[:2]
    scale = max(w / bw, h / bh)
    new_w, new_h = int(bw * scale), int(bh * scale)
    resized = cv2.resize(_bg_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    x_off = (new_w - w) // 2
    y_off = (new_h - h) // 2
    return resized[y_off:y_off + h, x_off:x_off + w].copy()

# ============================================================
#   DRAW HELPERS
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
    rounded_rect(img, cx - tw // 2 - 22, cy - th // 2 - 12,
                 cx + tw // 2 + 22, cy + th // 2 + 12, C_CARD, radius=14)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2),
                font, scale, colour, thickness, cv2.LINE_AA)

def draw_button(canvas, x1, y1, x2, y2, colour, label,
                font_scale=1.0, shadow=True):
    if shadow:
        rounded_rect(canvas, x1 + 5, y1 + 5, x2 + 5, y2 + 5,
                     (180, 175, 185), radius=18)
    rounded_rect(canvas, x1, y1, x2, y2, colour, radius=18)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
    cv2.putText(canvas, label,
                (int(x1 + (x2 - x1 - tw) / 2),
                 int(y1 + (y2 - y1 + th) / 2)),
                font, font_scale, C_TEXT_WHITE, 2, cv2.LINE_AA)

def draw_hand(canvas, cx, cy, size, angle, wave_phase):
    parts = []
    parts.append((0, 0, int(size * 0.42), C_HAND))
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

# ============================================================
#   SOUND
# ============================================================

def beep_click():
    try: winsound.Beep(880, 55)
    except Exception: pass

def beep_start():
    try:
        winsound.Beep(660, 70); winsound.Beep(880, 70); winsound.Beep(1100, 110)
    except Exception: pass

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

# ============================================================
#   MATH HELPERS
# ============================================================

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

def ordinal(n):
    if n == 1: return "1st"
    if n == 2: return "2nd"
    if n == 3: return "3rd"
    return f"{n}th"

# ============================================================
#   MEDIAPIPE + CAMERA
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
    print("Could not open camera")
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
    avail_w = x2 - x1 - 40
    avail_h = y2 - y1 - 40
    scale = min(avail_w / lw, avail_h / lh)
    nw, nh = int(lw * scale), int(lh * scale)
    rz = cv2.resize(letter_img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    px = x1 + (x2 - x1 - nw) // 2
    py = y1 + (y2 - y1 - nh) // 2
    canvas[py:py + nh, px:px + nw] = rz

def get_window_size(default=(BASE_W, BASE_H)):
    try:
        rect = cv2.getWindowImageRect(WINDOW_NAME)
        rw, rh = rect[2], rect[3]
        if rw > 0 and rh > 0:
            return rw, rh
    except Exception:
        pass
    return default

# ============================================================
#   SCREEN 1 - NAME INPUT
# ============================================================

def name_input_screen():
    entered = ""
    MAX_LEN = 12
    mouse_x, mouse_y = 0, 0
    clicked = False
    fullscreen = False

    rows = ["ABCDEFGHI", "JKLMNOPQR", "STUVWXYZ"]

    def mouse_cb(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y, clicked
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, BASE_W, BASE_H)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    win_w, win_h = BASE_W, BASE_H
    cached_bg, cached_size = None, None
    start_time = cv2.getTickCount() / cv2.getTickFrequency()

    while True:
        t = (cv2.getTickCount() / cv2.getTickFrequency()) - start_time
        win_w, win_h = get_window_size((win_w, win_h))

        if cached_size != (win_w, win_h):
            cached_bg = get_background(win_w, win_h)
            cached_size = (win_w, win_h)

        canvas = cached_bg.copy()
        font = cv2.FONT_HERSHEY_DUPLEX

        # Title
        title = "SIGN LANGUAGE ADVENTURE"
        ts = win_w / 1150
        (tw, th), _ = cv2.getTextSize(title, font, ts, 3)
        pill_x1 = (win_w - tw) // 2 - 40
        pill_x2 = (win_w + tw) // 2 + 40
        pill_y1 = int(win_h * 0.05)
        pill_y2 = pill_y1 + th + 30
        rounded_rect(canvas, pill_x1, pill_y1, pill_x2, pill_y2, C_CARD, radius=20)
        centred_text(canvas, title, pill_y2 - 15, ts, C_TEXT_DARK, 3)

        # Waving hand
        draw_hand(canvas, pill_x1 - int(win_w * 0.06),
                  (pill_y1 + pill_y2) // 2,
                  int(win_h * 0.11),
                  math.sin(t * 3.0) * 12, t * 4.0)

        # Subtitle
        sub = "Type your name to begin"
        ss = win_w / 1700
        (sw, sh), _ = cv2.getTextSize(sub, font, ss, 2)
        sub_y = int(win_h * 0.20)
        rounded_rect(canvas, (win_w - sw) // 2 - 25, sub_y - sh - 10,
                     (win_w + sw) // 2 + 25, sub_y + 10, C_CARD, radius=14)
        centred_text(canvas, sub, sub_y, ss, C_MAGENTA_D, 2)

        # Name card
        card_w = int(win_w * 0.55)
        card_h = int(win_h * 0.13)
        card_x = (win_w - card_w) // 2
        card_y = int(win_h * 0.25)
        rounded_rect(canvas, card_x + 5, card_y + 5,
                     card_x + card_w + 5, card_y + card_h + 5,
                     (180, 175, 185), radius=18)
        rounded_rect(canvas, card_x, card_y,
                     card_x + card_w, card_y + card_h, C_CARD, radius=18)
        rounded_rect(canvas, card_x + 20, card_y + card_h - 6,
                     card_x + card_w - 20, card_y + card_h, C_MAGENTA, radius=3)
        shown = entered if entered else "____"
        centred_text(canvas, shown, card_y + card_h // 2 + 14,
                     win_w / 900, C_TEXT_DARK if entered else C_TEXT_SOFT, 4)

        # Keyboard
        key_w = int(win_w * 0.062)
        key_h = int(win_h * 0.075)
        gap = int(win_w * 0.009)
        start_y = int(win_h * 0.44)
        for r, row in enumerate(rows):
            row_w = len(row) * key_w + (len(row) - 1) * gap
            start_x = (win_w - row_w) // 2
            for c, letter in enumerate(row):
                kx = start_x + c * (key_w + gap)
                ky = start_y + r * (key_h + gap)
                kx2, ky2 = kx + key_w, ky + key_h
                hover = kx <= mouse_x <= kx2 and ky <= mouse_y <= ky2
                colour = C_MAGENTA if not hover else C_TEAL
                rounded_rect(canvas, kx + 3, ky + 3, kx2 + 3, ky2 + 3,
                             (180, 175, 185), radius=10)
                rounded_rect(canvas, kx, ky, kx2, ky2, colour, radius=10)
                (tw2, th2), _ = cv2.getTextSize(letter, font, win_w / 1400, 2)
                cv2.putText(canvas, letter,
                            (int(kx + (key_w - tw2) / 2),
                             int(ky + (key_h + th2) / 2)),
                            font, win_w / 1400, C_TEXT_WHITE, 2, cv2.LINE_AA)
                if clicked and hover and len(entered) < MAX_LEN:
                    entered += letter
                    beep_click()

        # Buttons
        # Buttons
        btn_y = int(win_h * 0.84)
        btn_h = int(win_h * 0.09)

        # DELETE button - dark slate so it's always visible
        del_w = int(win_w * 0.20)
        del_x = int(win_w * 0.24)
        del_x2 = del_x + del_w
        del_y2 = btn_y + btn_h
        del_hover = del_x <= mouse_x <= del_x2 and btn_y <= mouse_y <= del_y2
        del_colour = (90, 85, 105) if not del_hover else (130, 120, 150)
        draw_button(canvas, del_x, btn_y, del_x2, del_y2,
                    del_colour, "DELETE", font_scale=win_w / 1400)
        if clicked and del_hover and entered:
            entered = entered[:-1]
            beep_click()

        # START button - green when ready, muted when not
        start_w = int(win_w * 0.26)
        start_x = int(win_w * 0.50)
        start_x2 = start_x + start_w
        start_hover = start_x <= mouse_x <= start_x2 and btn_y <= mouse_y <= del_y2
        if not entered:
            sc = (150, 145, 160)         # disabled - visible grey-purple
        elif start_hover:
            sc = (110, 220, 110)         # brighter green on hover
        else:
            sc = (70, 180, 70)           # ready - deeper green
        draw_button(canvas, start_x, btn_y, start_x2, del_y2,
                    sc, "START", font_scale=win_w / 1300)
        if clicked and start_hover and entered:
            beep_start()
            return entered

        cv2.putText(canvas, "F = fullscreen   -   ESC = quit",
                    (14, win_h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, C_TEXT_DARK, 1, cv2.LINE_AA)

        clicked = False
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return None
        elif key in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == 13:
            if entered:
                beep_start()
                return entered
        elif key == 8:
            entered = entered[:-1]
        elif 65 <= key <= 90:
            if len(entered) < MAX_LEN:
                entered += chr(key); beep_click()
        elif 97 <= key <= 122:
            if len(entered) < MAX_LEN:
                entered += chr(key).upper(); beep_click()

# ============================================================
#   SCREEN 2 - TEACH
# ============================================================

def teach_screen(name, letters):
    idx = 0
    win_w, win_h = BASE_W, BASE_H
    cached_bg, cached_size = None, None
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
            win_w, win_h = get_window_size((win_w, win_h))

            if cached_size != (win_w, win_h):
                cached_bg = get_background(win_w, win_h)
                cached_size = (win_w, win_h)

            canvas = cached_bg.copy()

            ret, frame = cap.read()
            if not ret:
                return False
            frame, landmarks = read_hand(frame)

            pill_text(canvas, f"Let's learn {target}", int(win_h * 0.08),
                      win_w / 1100, C_TEXT_DARK, 3)
            draw_hand(canvas, int(win_w * 0.06), int(win_h * 0.08),
                      int(win_h * 0.09), math.sin(t * 3.0) * 12, t * 4.0)

            # Progress dots
            dot_r = int(win_h * 0.022)
            gap = dot_r * 3
            total_w = len(letters) * gap
            start_x = (win_w - total_w) // 2 + gap // 2
            for i, L in enumerate(letters):
                cx = start_x + i * gap
                cy = int(win_h * 0.16)
                if i < idx:
                    colour, label = C_GREEN, "✓"
                elif i == idx:
                    colour, label = C_MAGENTA, L
                else:
                    colour, label = (220, 215, 225), L
                cv2.circle(canvas, (cx, cy), dot_r, colour, -1, cv2.LINE_AA)
                fs = win_w / 1800
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
                cv2.putText(canvas, label, (cx - tw // 2, cy + th // 2),
                            cv2.FONT_HERSHEY_DUPLEX, fs, C_TEXT_WHITE, 2, cv2.LINE_AA)

            card_x1 = int(win_w * 0.06)
            card_y1 = int(win_h * 0.26)
            card_x2 = int(win_w * 0.46)
            card_y2 = int(win_h * 0.86)
            draw_letter_card(canvas, letter_img, card_x1, card_y1, card_x2, card_y2)

            cam_x1 = int(win_w * 0.52)
            cam_y1 = int(win_h * 0.26)
            cam_x2 = int(win_w * 0.94)
            cam_y2 = int(win_h * 0.86)
            draw_camera_card(canvas, frame, cam_x1, cam_y1, cam_x2, cam_y2)

            status = "Hand detected!" if landmarks else "Show your hand!"
            scolour = C_GREEN if landmarks else C_MAGENTA
            (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, win_w/2000, 2)
            px1 = cam_x1 + (cam_x2 - cam_x1 - sw) // 2 - 15
            py1 = cam_y2 + 8
            rounded_rect(canvas, px1, py1, px1 + sw + 30, py1 + sh + 20, scolour, radius=12)
            cv2.putText(canvas, status, (px1 + 15, py1 + sh + 6),
                        cv2.FONT_HERSHEY_DUPLEX, win_w/2000, C_TEXT_WHITE, 2, cv2.LINE_AA)

            instr = "Copy the shape, then press SPACE"
            (iw, ih), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_DUPLEX, win_w/2000, 2)
            rounded_rect(canvas, (win_w - iw) // 2 - 25, win_h - 70,
                         (win_w + iw) // 2 + 25, win_h - 25, C_CARD, radius=14)
            cv2.putText(canvas, instr, ((win_w - iw) // 2, win_h - 38),
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
                beep_right()
                flash = canvas.copy()
                overlay = np.full_like(flash, 0, dtype=np.uint8)
                overlay[:] = C_GREEN
                blended = cv2.addWeighted(flash, 0.55, overlay, 0.45, 0)
                centred_text(blended, f"Got {target}!", int(win_h * 0.5),
                             win_w / 900, C_TEXT_WHITE, 5)
                cv2.imshow(WINDOW_NAME, blended)
                cv2.waitKey(400)
                saved = True
        idx += 1
    return True

# ============================================================
#   SCREEN 3 - PLAY
# ============================================================

def play_screen(name, letters):
    refs = {}
    for L in set(letters):
        p = os.path.join(SESSION_DIR, f"{name}_{L}.json")
        if not os.path.exists(p):
            print(f"Missing taught file for {L}")
            return None
        with open(p) as f:
            refs[L] = normalize(json.load(f)["landmarks"])

    win_w, win_h = BASE_W, BASE_H
    cached_bg, cached_size = None, None

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

        lp = os.path.join(LETTER_IMG_DIR, f"{target}.png")
        letter_img = cv2.imread(lp) if os.path.exists(lp) else None

        ret, frame = cap.read()
        if not ret:
            return None
        frame, landmarks = read_hand(frame)

        is_match = False
        if landmarks:
            live_norm = normalize(landmarks)
            distances = sorted([(L, distance(live_norm, refs[L])) for L in refs],
                               key=lambda x: x[1])
            cl_letter, cl_dist = distances[0]
            rn_dist = distances[1][1] if len(distances) > 1 else 999
            is_match = (cl_letter == target
                        and cl_dist < MATCH_THRESHOLD
                        and (rn_dist - cl_dist) > MATCH_MARGIN)

        if is_match:
            if hold_start is None:
                hold_start = time.time()
            held = time.time() - hold_start
            if held >= HOLD_TIME:
                points = POINTS_TABLE.get(attempt, 0)
                score += points
                beep_right()
                canvas = cached_bg.copy() if cached_bg is not None else get_background(win_w, win_h)
                card_box = (int(win_w*0.06), int(win_h*0.26),
                            int(win_w*0.46), int(win_h*0.86))
                draw_letter_card(canvas, letter_img, *card_box)
                centred_text(canvas, f"{target} correct!", int(win_h * 0.32),
                             win_w / 900, C_GREEN, 5)
                centred_text(canvas, f"+{points} points", int(win_h * 0.40),
                             win_w / 1400, C_TEXT_DARK, 3)
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
                feedback = f"Attempt {attempt}/{MAX_ATTEMPTS} - press SPACE when ready"
                feedback_colour = C_MAGENTA

        win_w, win_h = get_window_size((win_w, win_h))
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
                colour, label = C_GREEN, "v"
            elif i == current_idx:
                colour, label = C_MAGENTA, "?"
            else:
                colour, label = (220, 215, 225), "?"
            cv2.circle(canvas, (cx, cy), dot_r, colour, -1, cv2.LINE_AA)
            fs = win_w / 1600
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
            cv2.putText(canvas, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_DUPLEX, fs, C_TEXT_WHITE, 2, cv2.LINE_AA)

        # Left card
        card_x1 = int(win_w * 0.06)
        card_y1 = int(win_h * 0.26)
        card_x2 = int(win_w * 0.46)
        card_y2 = int(win_h * 0.86)
        rounded_rect(canvas, card_x1, card_y1, card_x2, card_y2, C_CARD, radius=22)

        cx_mid = (card_x1 + card_x2) // 2
        cy_mid = (card_y1 + card_y2) // 2
        fs_l = win_w / 1400
        for txt, off in [(f"Make the", -60),
                         (f"{ordinal(current_idx + 1)} letter", 0),
                         (f"of {name}", 60)]:
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs_l, 2)
            cv2.putText(canvas, txt, (cx_mid - tw // 2, cy_mid + th // 2 + off),
                        cv2.FONT_HERSHEY_DUPLEX, fs_l, C_TEXT_DARK, 2, cv2.LINE_AA)

        # Camera
        cam_x1 = int(win_w * 0.52)
        cam_y1 = int(win_h * 0.26)
        cam_x2 = int(win_w * 0.94)
        cam_y2 = int(win_h * 0.86)
        draw_camera_card(canvas, frame, cam_x1, cam_y1, cam_x2, cam_y2)

        # Feedback pill
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
                draw_letter_card(canvas, letter_img, *card_box)
                centred_text(canvas, f"{target} correct!", int(win_h * 0.32),
                             win_w / 900, C_GREEN, 5)
                centred_text(canvas, f"+{points} points", int(win_h * 0.40),
                             win_w / 1400, C_TEXT_DARK, 3)
                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(1200)
                current_idx += 1
                attempt = 1
                feedback = "Make the shape, then press SPACE"
                feedback_colour = C_MAGENTA
                continue
            else:
                if attempt >= MAX_ATTEMPTS:
                    beep_skip()
                    canvas = cached_bg.copy() if cached_bg is not None else get_background(win_w, win_h)
                    card_box = (card_x1, card_y1, card_x2, card_y2)
                    draw_letter_card(canvas, letter_img, *card_box)
                    centred_text(canvas, f"It was {target}", int(win_h * 0.32),
                                 win_w / 900, C_YELLOW, 5)
                    centred_text(canvas, "Moving on - no points", int(win_h * 0.40),
                                 win_w / 1400, C_TEXT_DARK, 3)
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
                    feedback = f"Attempt {attempt}/{MAX_ATTEMPTS} - {nxt} points available"
                    feedback_colour = C_RED
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
    return {"score": score, "time": elapsed, "name": name}

# ============================================================
#   SCREEN 4 - CELEBRATION  (3 working buttons)
# ============================================================

def celebration_screen(name, result):
    win_w, win_h = BASE_W, BASE_H
    cached_bg, cached_size = None, None
    mouse_x, mouse_y = 0, 0
    clicked = False

    def mouse_cb(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y, clicked
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, BASE_W, BASE_H)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    start = cv2.getTickCount() / cv2.getTickFrequency()

    while True:
        t = (cv2.getTickCount() / cv2.getTickFrequency()) - start
        win_w, win_h = get_window_size((win_w, win_h))

        if cached_size != (win_w, win_h):
            cached_bg = get_background(win_w, win_h)
            cached_size = (win_w, win_h)
        canvas = cached_bg.copy()

        # Confetti
        for i in range(80):
            rng = random.Random(i * 131 + int(t * 2))
            x = int(rng.uniform(0, win_w))
            y = int((rng.uniform(0, win_h) + t * rng.uniform(60, 180)) % win_h)
            colour = [C_MAGENTA, C_TEAL, C_GREEN, C_YELLOW, C_RED][i % 5]
            cv2.circle(canvas, (x, y), 6, colour, -1, cv2.LINE_AA)

        # Panel
        panel_x1 = int(win_w * 0.10)
        panel_y1 = int(win_h * 0.14)
        panel_x2 = int(win_w * 0.90)
        panel_y2 = int(win_h * 0.86)
        rounded_rect(canvas, panel_x1, panel_y1, panel_x2, panel_y2, C_CARD, radius=30)

        # Waving hand
        draw_hand(canvas, int(win_w * 0.5), int(win_h * 0.24),
                  int(win_h * 0.10), math.sin(t * 4.0) * 18, t * 6.0)

        centred_text(canvas, "YOU DID IT!", int(win_h * 0.38),
                     win_w / 600, C_MAGENTA, 6)
        centred_text(canvas, f"You spelled {name}!", int(win_h * 0.48),
                     win_w / 900, C_TEXT_DARK, 3)
        centred_text(canvas, f"Score: {result['score']}", int(win_h * 0.56),
                     win_w / 1100, C_GREEN, 3)
        centred_text(canvas, f"Time: {result['time']:.1f} seconds", int(win_h * 0.62),
                     win_w / 1500, C_TEXT_SOFT, 2)

        # Three buttons
        btn_h = int(win_h * 0.09)
        btn_y = int(win_h * 0.72)
        btn_gap = int(win_w * 0.02)

        # PLAY AGAIN
        again_w = int(win_w * 0.20)
        again_x = int(win_w * 0.14)
        again_x2 = again_x + again_w
        again_y2 = btn_y + btn_h
        again_hover = again_x <= mouse_x <= again_x2 and btn_y <= mouse_y <= again_y2
        draw_button(canvas, again_x, btn_y, again_x2, again_y2,
                    C_TEAL if not again_hover else (230, 210, 70),
                    "PLAY AGAIN", font_scale=win_w / 1500)

        # NEW NAME
        new_w = int(win_w * 0.20)
        new_x = int(win_w * 0.40)
        new_x2 = new_x + new_w
        new_hover = new_x <= mouse_x <= new_x2 and btn_y <= mouse_y <= again_y2
        draw_button(canvas, new_x, btn_y, new_x2, again_y2,
                    C_MAGENTA if not new_hover else (230, 100, 250),
                    "NEW NAME", font_scale=win_w / 1500)

        # START FRESH
        fresh_w = int(win_w * 0.22)
        fresh_x = int(win_w * 0.66)
        fresh_x2 = fresh_x + fresh_w
        fresh_hover = fresh_x <= mouse_x <= fresh_x2 and btn_y <= mouse_y <= again_y2
        draw_button(canvas, fresh_x, btn_y, fresh_x2, again_y2,
                    C_RED if not fresh_hover else (110, 110, 240),
                    "START FRESH", font_scale=win_w / 1600)

        cv2.putText(canvas, "R = play again   -   N = new name   -   F = fresh start   -   ESC = quit",
                    (14, win_h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, C_TEXT_DARK, 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        if clicked:
            if again_hover:
                clicked = False
                return "again"
            if new_hover:
                clicked = False
                return "new"
            if fresh_hover:
                clicked = False
                return "fresh"
        clicked = False

        if key == 27 or key == ord('q'):
            return "quit"
        elif key in (ord('r'), ord('R')):
            return "again"
        elif key in (ord('n'), ord('N')):
            return "new"
        elif key in (ord('f'), ord('F')):
            return "fresh"
        elif key == 13:
            return "quit"

# ============================================================
#   MAIN
# ============================================================

def main():
    while True:
        name = name_input_screen()
        if not name:
            break

        letters = []
        seen = set()
        for ch in name:
            if ch not in seen:
                seen.add(ch)
                letters.append(ch)

        missing = [L for L in letters
                   if not os.path.exists(os.path.join(LETTER_IMG_DIR, f"{L}.png"))]
        if missing:
            for _ in range(180):
                canvas = get_background(BASE_W, BASE_H)
                centred_text(canvas, "Oops!", int(BASE_H * 0.35),
                             BASE_W / 800, C_MAGENTA, 5)
                centred_text(canvas, f"No picture for: {', '.join(missing)}",
                             int(BASE_H * 0.50), BASE_W / 1400, C_TEXT_DARK, 2)
                centred_text(canvas, "Try another name", int(BASE_H * 0.60),
                             BASE_W / 1600, C_TEXT_SOFT, 2)
                cv2.imshow(WINDOW_NAME, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    return
            continue

        if not teach_screen(name, letters):
            break
        result = play_screen(name, letters)
        if result is None:
            break

        while True:
            choice = celebration_screen(name, result)
            if choice == "quit":
                return
            elif choice == "again":
                result = play_screen(name, letters)
                if result is None:
                    return
            elif choice == "new":
                break
            elif choice == "fresh":
                for f in glob.glob(os.path.join(SESSION_DIR, "*.json")):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                print("Sessions cleared.")
                break

if __name__ == "__main__":
    main()
    cap.release()
    cv2.destroyAllWindows()
    print("Thanks for playing!")
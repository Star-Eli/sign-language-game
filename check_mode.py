import cv2
import mediapipe as mp
import json
import os
import math
import time

# --- CONFIG ---
LETTER_IMG_DIR = r"C:\Users\Elias\letters"
SESSION_DIR    = r"C:\Users\Elias\sessions"

# Match must be BELOW this distance
MATCH_THRESHOLD = 0.25
# Target must be this much closer than the runner-up (margin)
MATCH_MARGIN    = 0.05
# Hold time in seconds
HOLD_TIME       = 1.0

# --- Ask for name ---
name = input("Enter the name to spell: ").strip().upper()
if not name:
    print("No name entered.")
    exit()

letters = [ch for ch in name if ch.isalpha()]

# Load ALL taught references for this name
refs = {}
missing = []
for L in set(letters):
    path = os.path.join(SESSION_DIR, f"{name}_{L}.json")
    if os.path.exists(path):
        with open(path) as f:
            refs[L] = json.load(f)["landmarks"]
    else:
        missing.append(L)

if missing:
    print(f"❌ Missing taught files for: {missing}")
    exit()

print(f"\nName: {name}")
print(f"Letters taught: {sorted(refs.keys())}")
print(f"Hold each shape for ~{HOLD_TIME:.1f}s\n")

# --- MediaPipe ---
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

# --- Helpers ---
def normalize(landmarks):
    wx, wy, wz = landmarks[0]
    pts = [(x - wx, y - wy, z - wz) for (x, y, z) in landmarks]
    max_d = max(math.hypot(p[0], p[1]) for p in pts)
    if max_d < 1e-6:
        max_d = 1.0
    return [(p[0] / max_d, p[1] / max_d, p[2] / max_d) for p in pts]

def distance(a, b):
    total = 0.0
    for (x1, y1, z1), (x2, y2, z2) in zip(a, b):
        total += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return total / len(a)

def load_letter_img(L):
    p = os.path.join(LETTER_IMG_DIR, f"{L}.png")
    img = cv2.imread(p)
    if img is None:
        return None
    h, w = img.shape[:2]
    panel_h = 300
    scale = panel_h / h
    return cv2.resize(img, (int(w * scale), panel_h))

# Pre-normalize all references
ref_norm = {L: normalize(lms) for L, lms in refs.items()}

current_idx = 0
hold_start  = None
score       = 0
feedback    = ""
closest_now = None   # (letter, dist)

while current_idx < len(letters):
    target = letters[current_idx]
    letter_img = load_letter_img(target)

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_present = False
    live_norm = None

    if results.multi_hand_landmarks:
        hand_present = True
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        live = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        live_norm = normalize(live)

    # --- Compute distances to all taught letters ---
    closest_now = None
    if live_norm is not None:
        all_d = sorted(
            [(L, distance(live_norm, ref_norm[L])) for L in ref_norm],
            key=lambda x: x[1]
        )
        closest_now = all_d[0]      # (letter, dist)
        runner_up   = all_d[1] if len(all_d) > 1 else (None, 999)

        closest_letter, closest_dist = closest_now
        runner_dist = runner_up[1]

        # Match conditions:
        #  1. closest letter IS the target
        #  2. closest distance is below threshold
        #  3. closest is meaningfully closer than the next best
        is_match = (
            closest_letter == target
            and closest_dist < MATCH_THRESHOLD
            and (runner_dist - closest_dist) > MATCH_MARGIN
        )
    else:
        is_match = False
        closest_dist = None
        closest_letter = None

    if is_match:
        if hold_start is None:
            hold_start = time.time()
        held = time.time() - hold_start
        if held >= HOLD_TIME:
            print(f"[{target}] MATCH (dist={closest_dist:.3f}) +10")
            score += 10
            current_idx += 1
            hold_start = None
            feedback = f"✅ {target} correct!"
            time.sleep(0.4)
            continue
        else:
            feedback = f"Holding {target}... {held:.1f}/{HOLD_TIME:.1f}s"
    else:
        hold_start = None
        if not hand_present:
            feedback = "No hand detected"
        else:
            feedback = (
                f"Seeing: {closest_letter} (d={closest_dist:.2f}) "
                f"— need {target}"
            )

    # --- Build display ---
    cam_h, cam_w = frame.shape[:2]
    panel_h = 300
    scale = panel_h / cam_h
    cam_small = cv2.resize(frame, (int(cam_w * scale), panel_h))

    if letter_img is not None:
        combined = cv2.hconcat([letter_img, cam_small])
    else:
        combined = cam_small

    cv2.rectangle(combined, (0, 0), (combined.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(combined,
                f"Spell: {name}   Make the letter: {target}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    progress = " ".join(
        f"[{L}]" if i < current_idx else f" {L} "
        for i, L in enumerate(letters)
    )
    cv2.putText(combined, f"Progress: {progress}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.rectangle(combined,
                  (0, combined.shape[0] - 50),
                  (combined.shape[1], combined.shape[0]),
                  (0, 0, 0), -1)
    cv2.putText(combined, f"{feedback}   Score: {score}   (Q to quit)",
                (20, combined.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Check Mode", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if current_idx >= len(letters):
    print(f"\n🎉 You spelled {name}!  Score: {score}")
else:
    print(f"\nStopped at letter {current_idx+1}/{len(letters)}. Score: {score}")
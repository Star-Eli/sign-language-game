import cv2
import mediapipe as mp
import json
import os

# --- CONFIG ---
LETTER_IMG_DIR = r"C:\Users\Elias\letters"
SESSION_DIR    = r"C:\Users\Elias\sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# --- Ask for name ---
name = input("Enter the child's name: ").strip().upper()
if not name:
    print("No name entered. Exiting.")
    exit()

# Unique letters from the name, in the order they appear
seen = set()
letters = []
for ch in name:
    if ch.isalpha() and ch not in seen:
        seen.add(ch)
        letters.append(ch)

# Check we have pictures for all letters
missing = [L for L in letters if not os.path.exists(os.path.join(LETTER_IMG_DIR, f"{L}.png"))]
if missing:
    print(f"❌ Missing letter images for: {missing}")
    print(f"   Please crop those letters first with crop_chart.py")
    exit()

print(f"\nName: {name}")
print(f"Letters to teach: {letters}")
print(f"Total: {len(letters)} letters\n")

# --- MediaPipe setup ---
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

idx = 0
saved_this_letter = False

while idx < len(letters):
    letter = letters[idx]
    letter_img = cv2.imread(os.path.join(LETTER_IMG_DIR, f"{letter}.png"))

    # Resize the letter image to a fixed panel height
    panel_h = 300
    if letter_img is not None:
        h, w = letter_img.shape[:2]
        scale = panel_h / h
        letter_img = cv2.resize(letter_img, (int(w * scale), panel_h))

    saved_this_letter = False
    print(f"\n--- Teach letter: {letter} ({idx+1}/{len(letters)}) ---")
    print("Make the shape with your hand, then press SPACE to save.")
    print("Press S to skip, Q to quit.")

    # Inner loop for this letter
    while not saved_this_letter:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_present = False
        current_landmarks = None

        if results.multi_hand_landmarks:
            hand_present = True
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

        # --- Build the display: letter image on the left, camera on the right ---
        cam_h, cam_w = frame.shape[:2]
        if letter_img is not None:
            # Scale camera to match panel height ratio
            scale = panel_h / cam_h
            cam_small = cv2.resize(frame, (int(cam_w * scale), panel_h))
            combined = cv2.hconcat([letter_img, cam_small])
        else:
            combined = frame

        # --- Overlay UI ---
        cv2.rectangle(combined, (0, 0), (combined.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(combined, f"Teach letter: {letter}   ({idx+1}/{len(letters)})",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        status = "HAND OK" if hand_present else "NO HAND"
        colour = (0, 255, 0) if hand_present else (0, 0, 255)
        cv2.putText(combined, status, (combined.shape[1] - 200, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

        cv2.putText(combined, "SPACE = save this shape   S = skip   Q = quit",
                    (20, combined.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        cv2.imshow("Teach Mode", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Quit.")
            exit()
        elif key == ord('s'):
            print(f"[{letter}] Skipped")
            saved_this_letter = True
        elif key == ord(' '):
            if current_landmarks is None:
                print(f"[{letter}] No hand detected - not saved")
            else:
                path = os.path.join(SESSION_DIR, f"{name}_{letter}.json")
                with open(path, "w") as f:
                    json.dump({
                        "name": name,
                        "letter": letter,
                        "landmarks": current_landmarks
                    }, f)
                print(f"[{letter}] Saved to {path}")
                saved_this_letter = True

    idx += 1

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ All letters taught for {name}!")
print(f"   Files saved in {SESSION_DIR}")
print(f"   Next: run check_mode.py to test recognition.")
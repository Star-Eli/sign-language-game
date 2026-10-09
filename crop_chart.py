import cv2
import os

CHART_PATH = r"C:\Users\Elias\Documents\CVTS\SignLangaugeApp\letters\chart.jpg"
OUT_DIR    = r"C:\Users\Elias\Documents\CVTS\SignLangaugeApp\letters"
LETTERS    = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(CHART_PATH)
if img is None:
    print(f"Could not load {CHART_PATH}")
    exit()

# Fit the WHOLE chart to your screen
h, w = img.shape[:2]
scale = min(1280 / w, 700 / h, 1.0)
if scale < 1.0:
    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_AREA)
print(f"Loaded chart, displayed at {img.shape[1]}x{img.shape[0]}")

cv2.namedWindow("Chart", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Chart", img.shape[1], img.shape[0])

display = img.copy()
ref_point = []
cropping = False
current_crop = None

def mouse_crop(event, x, y, flags, param):
    global ref_point, cropping, current_crop, display

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        display = img.copy()
        cv2.rectangle(display, ref_point[0], (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        ref_point.append((x, y))
        display = img.copy()
        cv2.rectangle(display, ref_point[0], ref_point[1], (0, 255, 0), 2)

        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        if (x2 - x1) < 6 or (y2 - y1) < 6:
            print("Selection too small, ignore")
            current_crop = None
            return

        current_crop = img[y1:y2, x1:x2].copy()
        if current_crop.size > 0:
            # show enlarged preview
            preview = cv2.resize(current_crop, (200, 200),
                                 interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Crop preview", preview)
            print("Crop captured. Press the letter key to save it.")

cv2.setMouseCallback("Chart", mouse_crop)

print("Instructions:")
print("1. Drag a box around a letter")
print("2. Press the letter key (A-Z) to save it")
print("3. ESC closes when you are done")

while True:
    cv2.imshow("Chart", display)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == ord('q'):
        # If a crop is pending, treat as letter Q
        if current_crop is not None and current_crop.size > 0:
            path = os.path.join(OUT_DIR, "Q.png")
            cv2.imwrite(path, current_crop)
            print(f"Saved Q -> {path}")
            current_crop = None
            continue
        else:
            break

    try:
        letter = chr(key).upper()
    except Exception:
        continue

    if letter in LETTERS:
        if current_crop is None or current_crop.size == 0:
            print(f"[{letter}] No crop selected yet - drag a box first")
            continue
        path = os.path.join(OUT_DIR, f"{letter}.png")
        cv2.imwrite(path, current_crop)
        print(f"Saved {letter} -> {path}")
        current_crop = None

cv2.destroyAllWindows()
import cv2
import time
import argparse
import sys
import os
import urllib.request
import numpy as np

parser = argparse.ArgumentParser(description='Finger tracking overlay using MediaPipe Tasks and OpenCV')
parser.add_argument('--camera', type=int, default=0, help='Camera device index')
parser.add_argument('--mirror', action='store_true', help='Mirror the camera preview (useful for selfie view)')
parser.add_argument('--model', type=str, default='hand_landmarker.task', help='Path to the hand_landmarker.task model')
args = parser.parse_args()

try:
    # Import the Tasks API
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult, HandLandmarksConnections
    from mediapipe.tasks.python.vision.core import image as mp_image_module
    from mediapipe.tasks.python.core.base_options import BaseOptions
except Exception:
    print("Error: mediapipe Tasks API not available. Ensure mediapipe (v0.10+) is installed in this environment.")
    sys.exit(1)

# Ensure model file exists; download official task model if not present
MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'
model_path = args.model
if not os.path.exists(model_path):
    print(f'Model not found at "{model_path}" — downloading...')
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print('Model downloaded.')
    except Exception as e:
        print('Failed to download the model:', e)
        print('You can provide a local model path with --model')
        sys.exit(1)

# Create the hand landmarker from model (image mode) with support for 2 hands
try:
    base_options = BaseOptions(model_asset_path=model_path)
    options = HandLandmarkerOptions(base_options=base_options, num_hands=2)
    landmarker = HandLandmarker.create_from_options(options)
except Exception as e:
    print('Failed to create HandLandmarker from model:', e)
    sys.exit(1)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f'Unable to open camera {args.camera}')
    landmarker.close()
    sys.exit(1)

prev_time = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Try to construct a MediaPipe Image in a few ways for compatibility
        mp_image = None
        try:
            mp_image = mp_image_module.Image.create_from_array(img_rgb)
        except Exception:
            try:
                mp_image = mp_image_module.Image(image_format=mp_image_module.ImageFormat.SRGB, data=img_rgb)
            except Exception:
                # As a last resort encode to JPEG bytes
                try:
                    success, enc = cv2.imencode('.jpg', img_rgb)
                    if success:
                        mp_image = mp_image_module.Image.create_from_bytes(enc.tobytes())
                except Exception:
                    mp_image = None

        if mp_image is None:
            cv2.putText(frame, 'MP Image creation failed', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            try:
                # Use image-mode detection (slower but simpler). For higher performance,
                # create a Video-mode landmarker and call detect_for_video with timestamps.
                result = landmarker.detect(mp_image)

                # Draw landmarks
                if result and result.hand_landmarks:
                    # We'll collect thumbnails for up to two hands to show in top-left
                    thumbs = []
                    for hand_landmarks in result.hand_landmarks:
                        # draw connections
                        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                            start = hand_landmarks[conn.start]
                            end = hand_landmarks[conn.end]
                            x1, y1 = int(start.x * w), int(start.y * h)
                            x2, y2 = int(end.x * w), int(end.y * h)
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # draw joints and compute bbox
                        xs = [lm.x for lm in hand_landmarks]
                        ys = [lm.y for lm in hand_landmarks]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        px1, py1 = max(int((min_x - 0.15) * w), 0), max(int((min_y - 0.15) * h), 0)
                        px2, py2 = min(int((max_x + 0.15) * w), w - 1), min(int((max_y + 0.15) * h), h - 1)

                        for lm in hand_landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                        # extract thumbnail
                        if px2 > px1 and py2 > py1:
                            thumb = frame[py1:py2, px1:px2]
                        else:
                            thumb = None
                        thumbs.append((thumb, (px1, py1, px2, py2)))

                    # Draw up to two thumbnails in top-left stacked
                    thumb_w, thumb_h = 160, 120
                    pad = 8
                    for i, (thumb, bbox) in enumerate(thumbs[:2]):
                        tx = pad
                        ty = pad + i * (thumb_h + pad)
                        if thumb is not None and thumb.size > 0:
                            thumb_resized = cv2.resize(thumb, (thumb_w, thumb_h))
                            frame[ty:ty+thumb_h, tx:tx+thumb_w] = thumb_resized
                            cv2.rectangle(frame, (tx, ty), (tx+thumb_w, ty+thumb_h), (255,255,255), 1)
                            label = 'Hand {}'.format(i+1)
                            cv2.putText(frame, label, (tx+4, ty+thumb_h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        else:
                            # placeholder
                            cv2.rectangle(frame, (tx, ty), (tx+thumb_w, ty+thumb_h), (100,100,100), 1)
                            cv2.putText(frame, 'No hand', (tx+4, ty+thumb_h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            except Exception:
                pass

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Finger Skeleton Overlay', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            args.mirror = not args.mirror

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        landmarker.close()
    except Exception:
        pass

import cv2
import mediapipe as mp
import os

# Khởi tạo Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

save_folder = "./data/mask_hand/A"
os.makedirs(save_folder, exist_ok=True) 
image_count = 1  

def save_hand_image(image, hand_landmarks, output_size=(128, 128)):
    global image_count
    # Lấy tọa độ bao quanh bàn tay
    h, w, _ = image.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min, y_min = min(x, x_min), min(y, y_min)
        x_max, y_max = max(x, x_max), max(y, y_max)

    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    hand_roi = image[y_min:y_max, x_min:x_max]
    hand_resized = cv2.resize(hand_roi, output_size)

    # Tạo tên tệp và lưu ảnh
    save_path = os.path.join(save_folder, f"hand_{image_count}.jpg")
    cv2.imwrite(save_path, hand_resized)
    print(f"Saved hand image to {save_path}")
    image_count += 1  # Tăng số đếm ảnh

# Vòng lặp chính để xử lý webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      )
            
            save_hand_image(frame, hand_landmarks)

    cv2.imshow("Hand Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
hands.close()

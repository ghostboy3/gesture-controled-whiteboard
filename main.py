import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Access the default camera (usually the first connected camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Perform hand tracking
    results = hands.process(image)

    # If hands are detected, iterate through each hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract bounding box coordinates
            h, w, c = frame.shape
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('Hand Tracking with Bounding Box', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
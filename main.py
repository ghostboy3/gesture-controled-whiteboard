import cv2
import mediapipe as mp
import math
import pygame

# Pygame initialization
pygame.init()
window_width, window_height = 640, 480
screen = pygame.display.set_mode((window_width, window_height))
clock = pygame.time.Clock()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Access the default camera (usually the first connected camera)
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # 1 for horizontal flipping

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Perform hand tracking
    results = hands.process(image)

    # If hands are detected, extract landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Get the landmark coordinates for the tip of the index finger (Landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            height, width, _ = frame.shape
            tip_x, tip_y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

            # Display the coordinates of the index finger tip
            cv2.putText(frame, f'Index Finger Tip: ({tip_x}, {tip_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Pygame drawing
            # screen.fill((0, 0, 0))  # Clear screen
            pygame.draw.circle(screen, (255, 0, 0), (tip_x, tip_y), 10)  # Draw a circle at finger tip
            pygame.display.flip()  # Update the display


    # Check for Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check for the 'q' key to exit the loop
    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        running = False

    # Limit frame rate
    # clock.tick(10)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
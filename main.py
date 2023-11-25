import cv2
import mediapipe as mp
import math
import pygame
import pickle
import numpy as np

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Pygame initialization
pygame.init()
window_width, window_height = 640, 480
screen = pygame.display.set_mode((window_width, window_height))
screen.fill((255, 255, 255))  # Clear screen 
pygame.display.update()

clock = pygame.time.Clock()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) #mp_hands.Hands()

# Access the default camera (usually the first connected camera)
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
while True:
    dataPoints = []
    
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
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                dataPoints.append(x)
                dataPoints.append(y)
            try:
                prediction = (model.predict([np.asarray(dataPoints)]))
            except Exception as e:
                prediction = ""
            # print(prediction)
            
            # Get the landmark coordinates for the tip of the index finger (Landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            height, width, _ = frame.shape
            tip_x, tip_y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

            # Display the coordinates of the index finger tip
            cv2.putText(frame, f'Index Finger Tip: ({tip_x}, {tip_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Pygame drawing
            if prediction == ['palm']:
                pass
            if prediction == ['index']:
                pygame.draw.circle(screen, (0, 0, 0), (tip_x, tip_y), 10)  # Draw a circle at finger tip
            if prediction == ['indexpinky']:      # Erase
                pygame.draw.circle(screen, (255, 255, 255), (tip_x, tip_y), 40)  # Draw a circle at finger tip
            if prediction == ['fist']:
                screen.fill((255, 255, 255))  # Clear screen
            
            #draw prediction on screen
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Draw the bounding box around the hand
            try:
                text = prediction[0]
            except Exception as e:
                text = ""
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
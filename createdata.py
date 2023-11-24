import cv2
import os
import mediapipe as mp
import pickle


rootDir = 'images'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

# Loop through all the images
for subfolder in os.listdir(rootDir):
    for file in os.listdir(os.path.join(rootDir, subfolder)):
        
        imgData = []
        
        path = os.path.join(rootDir, subfolder, file)
        img = cv2.imread(path)
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(imgRgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    imgData.append(x)
                    imgData.append(y)
                    
            data.append(imgData)
            labels.append(subfolder)
                    


                # mp_drawing.draw_landmarks(imgRgb, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
#save the data
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
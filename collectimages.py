# Press the 'e' key to start the code

import cv2
import os

# Create a directory to save images if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Initialize the camera
camera = cv2.VideoCapture(0)

# Variables to keep track of images and whether 'e' has been pressed
count = 0
save_images = False

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)  # 1 for horizontal flipping

    # Show the video stream
    cv2.imshow('Camera', frame)

    # Check for the 'e' key press
    key = cv2.waitKey(1)
    if key == ord('e'):
        save_images = True

    # Save images if 'e' is pressed
    if save_images:
        if count < 300:     # collect 300 images
            # Save the image to the folder
            cv2.imwrite(f'images/indexpinky/image_{count}.jpg', frame)
            count += 1
            print(f'Saved image {count}')
            # Wait for 100 milliseconds before capturing the next image
            cv2.waitKey(100)
        else:
            print('300 images captured. Exiting.')
            break

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
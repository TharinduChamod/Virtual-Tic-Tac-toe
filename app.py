# Importing libraries
import cv2
import numpy as np
import mediapipe as mp


# Class for tic tac toe interface
class tictactoe():
    def __init__(self,image) -> None:
        # self.x = x
        # self.y = y
        self.image = image
        self.selected_grids = []
        # Count for exiting the game
        self.count = 0
    
    def drawGrid(self):
        # Add tic-tac-toe interface
        grid_size = 3

        # Hieght and width definition
        h = 720
        w = 1280

        # Drawing the digram
        for i in range (0, grid_size):
            for j in range (0, grid_size):  
                cv2.rectangle(img=self.image, rec=[int(i * w / grid_size), 
                                              int(j*h / grid_size), 
                                              int((i + 1) * w/ grid_size), 
                                              int((j + 1) * h / grid_size)], 
                                              color=(0, 255, 0), 
                                              thickness=7)   

        # returning the image contain grid
        return self.image
    
    # Function to determine the winning state
    def isWon():
        pass


# Initializing video caputure element as webcam
capture = cv2.VideoCapture(0)

# Using HD resolution
capture.set(3, 1280)
capture.set(4, 720)

# From mediapipe initialize hand detecing instance
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence = 0.8)

# Defining the drawing utilities
mpDraw = mp.solutions.drawing_utils

# Capturing imag and processing it
# Convert color from BGR to RGB format
# Then process hansds on tha image
while True:
    # Capture the image from vido capturing
    success, image = capture.read()

    # Flipping the image
    image_flip = cv2.flip(image,1)

    # Converting the color format of flipped image
    imageRGB = cv2.cvtColor(image_flip, cv2.COLOR_BGR2RGB)
    # Get the hands results of image
    results = hands.process(imageRGB)

    # Add tic-tac-toe interface
    # Add the grid
    image_flip = tictactoe(image=image_flip).drawGrid()

    # Working with ecah hand
    # Chechking if there are hand object
    if results.multi_hand_landmarks:
        # Process each hand landmark one by one
        for hand_lm in results.multi_hand_landmarks:
            # Get the location of ecah hand landmark point (there are 20 points)
            for (id, lm) in enumerate(hand_lm.landmark):
                # Find height, width and channel for ecah image
                height, width, channel = image_flip.shape
                # Get the central points of the identified hands
                cx, cy = int(lm.x*width), int(lm.y*height)

                # Printing the location of finger point
                # print(f"X= {cx} Y ={cy}")

            
                # Drawing hand landmarks
                # Centered point 
                if id == 8:
                    cv2.circle(image_flip, center=(cx, cy), radius=10, color=(0, 255, 0), thickness=cv2.FILLED)
            
            # Drawing
            mpDraw.draw_landmarks(image_flip, hand_lm, mp_hands.HAND_CONNECTIONS)
        
        # Showing the image as output
        cv2.imshow("Output", image_flip)
        
        # Waitng for 1 s delay after showing
        # cv2.waitKey(10)
        # Check for the 'q' key to exit
        if cv2.waitKey(1) == ord('q'):
            break

    

    # Show the empty vidoe otherwise
    else:
        cv2.imshow("Output", image_flip)
        # cv2.waitKey(10)
        # Check for the 'q' key to exit
        if cv2.waitKey(1) == ord('q'):
            break


# Releaing vidoe capture
capture.release()

# Destroying windows from application
cv2.destroyAllWindows()
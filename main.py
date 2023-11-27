import cv2
import mediapipe as mp
import numpy as np
import time
import random


# Initialize webcam
from google.protobuf.json_format import MessageToDict 
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)


# Initialize hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
puckx = 640
pucky = 320
dx = 10
dy = -10

# Initialize paddle and puck positions
   if results.multi_hand_landmarks:
            #for num, hand in enumerate(results.multi_hand_landmarks):
                #mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            for handLMS in results.multi_hand_landmarks:
                for id , lm in enumerate(handLMS.landmark):
                    h , w , c = image.shape 
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 8 : # this is the finger tip (index)
                        cv2.rectangle(image , (cx-50,cy+15),(cx+50,cy-15) , (120,100,100), -1 )   
                        if puckx + 10 in range (cx-50,cx+50) and pucky + 10 in range (cy-15,cy):
                            dy = -dy
                        if puckx - 10 in range (cx-50,cx+50) and pucky - 10 in range (cy,cy+15):
                            dy = -dy
                        if pucky + 10 in range (cy-14,cy+14) and puckx + 10 in range (cx-50,cx):
                            dx = -dx
                        if pucky - 10 in range (cy-14,cy+14) and puckx - 10 in range (cx,cx+50):
                            dx = -dx
                    for coord in random_coordinates:
                        frame = overlay_image(frame, target, coord)                         
                        distance = calculate_distance((coord[1] + target.shape[1] // 2, coord[0] + target.shape[0] // 2), (puckx,pucky))
                        if distance < proximity_threshold:
                            random_coordinates.remove(coord)
                            print(random_coordinates)    
                          puckx += dx
        pucky += dy
        cv2.circle(image,(puckx,pucky),10,(255,0,0),-1)
        if puckx + 10 >= 1280:
            dx = -dx
        if puckx - 10 <= 0:
            dx = -dx
        if pucky - 10 >= 640:
            dy = -dy
        if pucky + 10 <= 0:
            dy = -dy
       



# Initial velocity
initial_puck_velocity = [10, 10] 
puck_velocity = initial_puck_velocity.copy()

# Load target image and resize it to 30,30
target = cv2.imread("C:\\Users\\sarth\\ERC-Assignment-4\\target.png",cv2.IMREAD_UNCHANGED)
target = cv2.resize(target,(30,30))

# Initialize 5 target positions randomly(remember assignment 2!!)
ret, frame = cap.read()
height, width, _ = frame.shape

num_coordinates = 5
random_coordinates = [(np.random.randint(0, height - target.shape[0]), np.random.randint(0, width - target.shape[1])) for _ in range(num_coordinates)]
print(random_coordinates)

proximity_threshold = 35

with mp_hands.Hands(min_detection_confidence=0.65,min_tracking_confidence=0.35, max_num_hands =1) as hands:
    while True:
        time_remaining= game_duration-int((time.time()-start_time))
        success, frame=cap.read()
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image = cv2.flip(image,1)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        ret, frame = cap.read()

        for coord in random_coordinates:
            image = overlay_image(image,target, coord) 

# Initialize score
#6

# Initialize timer variables
start_time = time.time()
game_duration = 30  # 1/2 minute in seconds
def overlay_image(background, foreground, position):
    y1, x1 = position
    h1, w1, _ = foreground.shape

    alpha = foreground[:, :, 3] / 255.0
    for c in range(0, 3):
        background[y1:y1 + h1, x1:x1 + w1, c] = (1 - alpha) * background[y1:y1 + h1, x1:x1 + w1, c] + alpha * foreground[:, :, c] * alpha

    return background

def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def ending(image):
    score=5-len(random_coordinates)
    cv2.putText(image,"GAME OVER",(550, 200), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2,cv2.LINE_AA)
    cv2.putText(image,str(score),(550, 280), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 0), 2,cv2.LINE_AA)
    return image

# Function to check if the puck is within a 5% acceptance region of a target
def is_within_acceptance(puck, target, acceptance_percent=5):
  #complete the function
  #7
    return (
        0#make changes here!! #8
    )

while True:
    # Calculate remaining time and elapsed time in minutes and seconds   
    #9
    
    # Read a frame from the webcam
    #10

    # Flip the frame horizontally for a later selfie-view display
    #11

    # Convert the BGR image to RGB
    #12

    # Process the frame with mediapipe hands
    #13

    # Update paddle position based on index finger tip
    #14

    # Update puck position based on its velocity
    #15

    # Check for collisions with the walls
    #16

    # Check for collisions with the paddle
    #17

    # Check for collisions with the targets(use is_within_acceptance)    
    #18
            # Increase the player's score
            # Remove the hit target
            # Increase puck velocity after each hit by 2(you will have to increase both x and y velocities

    # Draw paddle, puck, and targets on the frame and overlay target image on frame
    #19

   # FOR REFERENCE:
    # for target_position in target_positions:
    #     target_roi = frame[target_position[1]:target_position[1] + target_image.shape[0],
    #                       target_position[0]:target_position[0] + target_image.shape[1]]
    #     alpha = target_image[:, :, 3] / 255.0
    #     beta = 1.0 - alpha
    #     for c in range(0, 3):
    #         target_roi[:, :, c] = (alpha * target_image[:, :, c] +
    #                               beta * target_roi[:, :, c])

    # Display the player's score on the frame
    #20

    # Display the remaining time on the frame
    #21

    # Check if all targets are hit or time is up
    #22

    # Display the resulting frame
       cv2.imshow("Hand Detection",image)


    # Exit the game when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

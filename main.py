import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math


# Initialize webcam
cap=cv2.VideoCapture(0)

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


# Initialize paddle and puck positions
#3
paddle_coords = [0 , 0]
puck_coords = [width/2 , height]

# paddle dimensions
paddle_width=150
paddle_length=10

# Initial velocity
initial_puck_velocity = [10, 10] 
puck_velocity = initial_puck_velocity.copy()

# Load target image and resize it to 30,30
image_path = "C:\\Users\\Asus\\Desktop\\target.png"
original_image = cv2.imread(image_path)

# Check if the image is loaded successfully
if original_image is None:
    print(f"Error: Unable to load the image from {image_path}")
    # Add appropriate error handling or exit the program
    exit()

target_size = (30, 30)
resized_image = cv2.resize(original_image, target_size)

# Initialize 5 target positions randomly(remember assignment 2!!)
targets = [(random.randint(50, width - 50), random.randint(50, height - 50), 20) for _ in range(5)]


# Initialize score
#6
score = 0
targets_hit=0

# Initialize timer variables
start_time = time.time()
game_duration = 30  # 1/2 minute in seconds

# Function to check if the puck is within a 5% acceptance region of a target
def is_within_acceptance(puck, target, acceptance_percent=5):
    #complete the function
    #7
    global is_target_hit

    dist = (math.sqrt(abs((puck[0] - target[0]*[targets_hit])^2 + (puck[1] - target[1][targets_hit])^2)))

    if dist >= int((width/24 + 30)*(105/100)):
        is_target_hit = True
    else:
        is_target_hit = False

    return is_target_hit
while True:
    
    # Calculate remaining time and elapsed time in minutes and seconds   
    elapsed_time=time.time()-start_time
    remaining_time= game_duration-elapsed_time
    elapsed_time_minutes=(time.time()-start_time)/60
    remaining_time_minutes= (game_duration-elapsed_time)/60
    
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with mediapipe hands
    results = hands.process(frame_rgb)

    # Update paddle position based on index finger tip
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x, index_finger_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

            paddle_position = (index_finger_x, index_finger_y)

    # Update puck position based on its velocity
    puck_coords[0]+=puck_velocity[0]*elapsed_time
    puck_coords[1]+=puck_velocity[1]*elapsed_time

    # Check for collisions with the walls
    if puck_coords[0]<0 or puck_coords[0] > width:
        puck_velocity[0] *= -1

    if puck_coords[1]<0 or puck_coords[1] > height:
        puck_velocity[1] *= -1    

    # Check for collisions with the paddle
    if(
        paddle_coords[0]< puck_coords[0]< paddle_coords[0] + paddle_width and
        paddle_coords[1]< puck_coords[1]< paddle_coords[1] + paddle_length
    ):
        puck_velocity[1]*=-1

    # Check for collisions with the targets(use is_within_acceptance)    
    #18
         # Increase the player's score
         # Remove the hit target
        # Increase puck velocity after each hit by 2(you will have to increase both x and y velocities
    for target in targets:
        if is_within_acceptance(puck_coords, target):
        # Increase the player's score
                score += 1

        # Remove the hit target
                targets.remove(target)

        # Increase puck velocity after each hit by 2 (both x and y velocities)
                puck_velocity[0] += 2
                puck_velocity[1] += 2

    # Draw paddle, puck, and targets on the frame and overlay target image on frame
    cv2.rectangle(
        frame,
        (int(paddle_coords[0]), int(paddle_coords[1])),
        (int(paddle_coords[0] + paddle_width), int(paddle_coords[1] + paddle_length)),
        (0, 255, 0),
        -1,
        )

    cv2.circle(frame, (int(puck_coords[0]), int(puck_coords[1])), 10, (255, 0, 0), -1)

    for target in targets:
        cv2.circle(frame, (target[0], target[1]), target[2], (0, 0, 255), -1)

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
    cv2.putText(
        frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )


    # Display the remaining time on the frame
    cv2.putText(
        frame,
        f"Time: {int(remaining_time_minutes):02}:{int(remaining_time) % 60:02}",
        (width - 200, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # Check if all targets are hit or time is up
    if not targets or remaining_time <= 0:
        cv2.putText(
                frame,
                "Game Over",
                (int(width / 2) - 100, int(height / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4,
            )
        cv2.imshow('Game Frame', frame)
        cv2.waitKey(3000)  # Display "Game Over" for 3 seconds
        break

    # Display the resulting frame
    cv2.imshow('Game Frame', frame)

    # Exit the game when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

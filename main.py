import random
import time

import cv2
import mediapipe as mp
import numpy as np

width = 1920
height = 1080

# Initialize webcam
cap = cv2.VideoCapture(0)  # 1
cap.set(3, width)
cap.set(4, height)

# Initialize hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # 2

# Initialize paddle and puck positions
paddle_position = [width // 3, height // 3]
puck_position = [width // 2, height // 2]  # 3

# Initial velocity
initial_puck_velocity = [15, 15]
puck_velocity = initial_puck_velocity.copy()

# Load target image, convert it to RGB and resize it to 30,30
target_image = cv2.cvtColor(cv2.imread("target.png", -1), cv2.COLOR_BGRA2BGR)
target_image = cv2.resize(target_image, (50, 50))  # 4

# Initialize 20 target positions randomly
target_positions = [
    [
        random.randint(0, width - target_image.shape[1]),  # x-coordinate
        random.randint(0, height - target_image.shape[0]),  # y-coordinate
    ]
    for _ in range(20)
]  # 5

# Initialize score
score = 0  # 6

# Initialize timer variables
start_time = time.time()
game_duration = 120  # 8


# Function to check if the puck is within a 5% acceptance region of a target
def is_within_acceptance(puck, target, acceptance_percent=30):
    distance = np.sqrt((puck[0] - target[0]) ** 2 + (puck[1] - target[1]) ** 2)
    return distance <= acceptance_percent  # 7


while True:
    # Calculate remaining time and elapsed time in minutes and seconds
    elapsed_time = time.time() - start_time
    remaining_time = game_duration - elapsed_time  # 9

    # Read a frame from the webcam
    ret, frame = cap.read()  # 10

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)  # 11

    # Convert the BGR image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 12

    # Process the frame with mediapipe hands
    results = hands.process(rgb)  # 13

    # Update paddle position based on index finger tip
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:  # index finger tip
                    paddle_position = [cx, cy]  # 14

    # Update puck position based on its velocity
    puck_position[0] += puck_velocity[0]
    puck_position[1] += puck_velocity[1]  # 15

    # Check for collisions with the walls
    if puck_position[0] <= 0 or puck_position[0] >= width:
        puck_velocity[0] *= -1
    if puck_position[1] <= 0 or puck_position[1] >= height:
        puck_velocity[1] *= -1  # 16

    # Check for collisions with the paddle
    if is_within_acceptance(puck_position, paddle_position):
        puck_velocity[0] *= -1
        puck_velocity[1] *= -1  # 17

    # Check for collisions with the targets(use is_within_acceptance)
    for target_position in target_positions:
        if is_within_acceptance(puck_position, target_position):
            score += 1
            puck_velocity[0] += 10
            puck_velocity[1] += 10  # 18
            target_positions.remove(target_position)

    # Draw paddle, puck, and targets on the frame and overlay target image on frame
    cv2.circle(
        frame, tuple(paddle_position), 30, (255, 0, 0), -1
    )  # Increased paddle size
    cv2.circle(frame, tuple(puck_position), 30, (0, 255, 0), -1)
    for target_position in target_positions:
        frame[
            target_position[1] : target_position[1] + target_image.shape[0],
            target_position[0] : target_position[0] + target_image.shape[1],
        ] = target_image  # 19

    # Display the player's score on the frame
    cv2.putText(
        frame,
        "Score: " + str(score),
        (100, 100),  # Move text to the right
        cv2.FONT_HERSHEY_SIMPLEX,
        2,  # Increase font scale
        (0, 0, 0),  # Set color to white
        2,
        cv2.LINE_AA,
    )  # 20

    # Display the remaining time on the frame
    cv2.putText(
        frame,
        "Time: " + str(int(remaining_time)),
        (100, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )  # 21

    # Check if all targets are hit or time is up
    if not target_positions or remaining_time <= 0:
        break  # 22

    # Display the resulting frame
    cv2.imshow("Air Hockey", frame)  # 23

    # Exit the game when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()  # 24

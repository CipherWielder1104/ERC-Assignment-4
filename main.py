# Import necessary libraries
import mediapipe as mp  # for hand tracking
import numpy as np  # for numerical operations
import time  # for time-related functions
import random  # for generating random target positions
import cv2  # for computer vision and image processing

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize paddle and puck positions
paddle_x, paddle_y = 320, 480
puck_x, puck_y = 320, 240

# Initial velocity
initial_puck_velocity = [15, 15]
puck_velocity = initial_puck_velocity.copy()

# Load target image and resize it to 30,30
target_image = cv2.imread("ERC-Assignment-4/target.png", cv2.IMREAD_UNCHANGED)
target_image = cv2.resize(target_image, (30, 30))

# Initialize 5 target positions randomly
target_positions = [(random.randint(50, 590), random.randint(50, 430)) for _ in range(5)]

# Initialize score
score = 0

# Initialize timer variables
start_time = time.time()
game_duration = 30

# Function to check if the puck is within a 5% acceptance region of a target
def is_within_acceptance(puck, target, acceptance_percent=5):
    distance = np.sqrt((puck[0] - target[0]) ** 2 + (puck[1] - target[1]) ** 2)
    return distance <= (acceptance_percent / 100) * max(target_image.shape[:2])

# Main game loop
while True:
    # Calculate remaining time and elapsed time in minutes and seconds
    elapsed_time = time.time() - start_time
    remaining_time = max(0, game_duration - elapsed_time)
    minutes, seconds = divmod(int(remaining_time), 60)

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
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            paddle_x, paddle_y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])

    # Update puck position based on its velocity
    puck_x += puck_velocity[0]
    puck_y += puck_velocity[1]

    # Check for collisions with the walls
    if puck_x <= 0 or puck_x >= frame.shape[1]:
        puck_velocity[0] *= -1
    if puck_y <= 0 or puck_y >= frame.shape[0]:
        puck_velocity[1] *= -1

    # Check for collisions with the paddle
    if paddle_x < puck_x < paddle_x + 100 and paddle_y < puck_y < paddle_y + 20:
        puck_velocity[1] *= -1

    # Check for collisions with the targets
    for target_position in target_positions:
        if is_within_acceptance([puck_x, puck_y], target_position):
            # Increase the player's score
            score += 1

            # Remove the hit target
            target_positions.remove(target_position)

            # Increase puck velocity after each hit by 2 (both x and y velocities)
            puck_velocity[0] += 2
            puck_velocity[1] += 2

    # Draw paddle, puck, and targets on the frame and overlay target image on frame
    frame = cv2.rectangle(frame, (paddle_x, paddle_y), (paddle_x + 100, paddle_y + 20), (255, 0, 0), -1)
    frame = cv2.circle(frame, (int(puck_x), int(puck_y)), 10, (255, 255, 255), -1)

    for target_position in target_positions:
        frame_roi = frame[
            target_position[1] : target_position[1] + target_image.shape[0],
            target_position[0] : target_position[0] + target_image.shape[1],
        ]
        alpha = target_image[:, :, 3] / 255.0
        beta = 1.0 - alpha
        for c in range(0, 3):
            frame_roi[:, :, c] = (alpha * target_image[:, :, c] + beta * frame_roi[:, :, c])

    # Display the player's score on the frame
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the remaining time on the frame
    cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Check if all targets are hit or time is up
    if not target_positions or remaining_time == 0:
        break

    # Display the resulting frame
    cv2.imshow("Virtual Air Hockey", frame)

    # Exit the game when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import time


# Initialize webcam
#1

# Initialize hand tracking
#2

# Initialize paddle and puck positions
#3

# Initial velocity
initial_puck_velocity = [10, 10] 
puck_velocity = initial_puck_velocity.copy()

# Load target image and resize it to 30,30
#4

# Initialize 5 target positions randomly(remember assignment 2!!)
#5

# Initialize score
#6

# Initialize timer variables
start_time = time.time()
game_duration = 30  # 1/2 minute in seconds

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
    #23

    # Exit the game when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
#24

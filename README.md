# ERC-Assignment-4

## Steps to get you started

- Fork the repo
- clone it on your system
- make the required changes in the python file (main.py)
- push your changes on github
- make a pull request

### installation 

Navigate to the folder and run the following command to install the required python libraries


    pip install -r requirements.txt
**Problem Statement: Virtual Air Hockey Game**

You are tasked with developing a virtual air hockey game using computer vision and hand tracking. The game should be played using a webcam, where the player controls the paddle with their hand to hit a virtual puck and score points by hitting targets that appear on the screen. The game has a time limit, and the player's objective is to score as many points as possible within the given time.

**Features and Requirements:**

1. **Hand Tracking:**
   - Utilize the `mediapipe` library for hand tracking.
   - The position of the player's hand, specifically the index finger tip, should control the paddle's position on the screen.

2. **Game Elements:**
   - Display a virtual air hockey table on the screen.
   - Implement a puck that moves around the screen with an initial velocity.
   - Create a paddle controlled by the player's hand.
   - Generate multiple target positions randomly on the screen.The target image has been provided.

3. **Game Logic:**
   - The puck should bounce off the walls of the screen.
   - If the puck collides with the paddle, its vertical velocity should reverse.
   - If the puck enters the region of a target, the player scores a point, and the target disappears.
   - After hitting a target, increase the puck's velocity to make the game more challenging.

4. **Score and Timer:**
   - Display the player's score on the screen.
   - Implement a timer indicating the remaining time to play.

5. **Game Over Conditions:**
   - The game should end when either all targets are hit or the time limit is reached.
   - Display a game over message with the player's final score.

6. **User Interface:**
   - Provide clear and visible text displaying the player's score and the remaining time.

7. **User Interaction:**
   - Allow the player to quit the game by pressing the 'q' key.

8. **Visual Effects:**
   - Add visual effects to represent the puck, paddle, and targets on the screen.

9. **Congratulations Message:**
   - If the player hits all the targets within the time limit, display a congratulations message.

10. **Optimizations:**
   - Optimize the code for efficiency and smooth gameplay.
   - Implement any additional features or improvements to enhance the gaming experience.

**Note:** Ensure that the code is well-documented, making it easy for others to understand.Also feel free to add more features to the game to make it more interesting. 


**All the Best!!**

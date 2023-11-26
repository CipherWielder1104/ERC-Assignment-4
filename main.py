import cv2
import mediapipe as mp
import random
import time

# Importing required libraries

# Initializing hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Now we are setting up score and time variables and their initial values
score = 0
time_limit = 30
start_time = time.time()

# Screen dimensions
swidth = 640
sheight = 480

# Initialize puck variables
puck_radius = 15
puck_x = swidth // 2
puck_y = sheight // 2
puck_speed = 6
''' This is the speed with which it will move, later on we will increase the speed as the game proceeds
                    to make it more challenging.'''
puck_velocity = [random.choice([-1, 1]) * puck_speed, random.choice([-1, 1]) * puck_speed]

# Paddle dimensions
paddle_width = 150
paddle_height = 10
paddle_x = swidth // 2
paddle_y = sheight - 50
'''This is for the initial position of the paddle on the screen '''

# Load the target image with error checking
target_image = cv2.imread(
    "C:\\Users\\Hi\\Downloads\\winlibs-x86_64-posix-seh-gcc-13.2.0-mingw-w64ucrt-11.0.0-r1\\mingw64\\lib\\python3.9\\venv\\scripts\\target.png")

if target_image is None:
    print("Error: Unable to read the image file.")
else:
    # Resize the target image to 30x30
    target_image = cv2.resize(target_image, (30, 30))
    target_height, target_width, _ = target_image.shape

    # Opening the camera
    cap = cv2.VideoCapture(0)

    # Initialize smoothing parameters
    smoothing_factor = 0.99
    ''' I have included this to make tracking of the finger tip faster, the greater the smoothing factor the greater is the speed with which the hand is tracked'''

    # Initialize hand position for smoothing
    smoothed_paddle_x = paddle_x
    smoothed_paddle_y = paddle_y

    # Initialize target variables
    target_radius = 20
    targets = [(random.randint(50, swidth - 50), random.randint(50, sheight - 200)) for i in range(5)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            new_paddle_x = int(index_finger_tip.x * swidth)
            new_paddle_y = int(index_finger_tip.y * sheight)

            # Apply smoothing
            smoothed_paddle_x = int(smoothed_paddle_x * (1 - smoothing_factor) + new_paddle_x * smoothing_factor)
            smoothed_paddle_y = int(smoothed_paddle_y * (1 - smoothing_factor) + new_paddle_y * smoothing_factor)

            paddle_x = smoothed_paddle_x
            paddle_y = smoothed_paddle_y

        # Update puck position
        puck_x += puck_velocity[0]
        puck_y += puck_velocity[1]

        # This is to check collision with walls and if there is any collision then change the direction of movement of puck
        if puck_x - puck_radius <= 0 or puck_x + puck_radius >= swidth:
            puck_velocity[0] = -puck_velocity[0]

        if puck_y - puck_radius <= 0 or puck_y + puck_radius >= sheight:
            puck_velocity[1] = -puck_velocity[1]

        # Check collision with the paddle
        if (
                paddle_x - paddle_width // 2 <= puck_x <= paddle_x + paddle_width // 2
                and paddle_y - paddle_height // 2 <= puck_y <= paddle_y + paddle_height // 2
        ):
            puck_velocity[1] = -puck_velocity[1]

        # Check collision with targets
        hit_targets = []
        for target in targets:
            target_x, target_y = target
            # Calculate distance from the puck to the center of the target
            distance = ((puck_x - target_x) ** 2 + (puck_y - target_y) ** 2) ** 0.5

            '''Here, I am considering that a target is hit only when it hits the donut(circle)
            . That's why I have used the radius of the circle inscribed in the square.'''
            inscribed_circle_radius = min(target_width, target_height) / 2

            # Check if the distance is within the inscribed circle
            if distance <= inscribed_circle_radius:
                score += 1
                hit_targets.append(target)
                # This is to decrease the paddle size by 20 with each target hit so as to make it more challenging
                paddle_width = max(10, paddle_width - 20)
                # As I have mentioned before, this line of code is to increase the speed of puck to make the game more challenging
                puck_speed += 2
                puck_velocity = [random.choice([-1, 1]) * puck_speed, random.choice([-1, 1]) * puck_speed]

        # Removing hit targets
        for hit_target in hit_targets:
            targets.remove(hit_target)
        '''This piece of code is to remove the targets that are hit, first we sort the targets that are hit in a list and then remove them from the targets list'''
        '''This is for drawing the puck and paddle on the screen'''
        cv2.circle(frame, (puck_x, puck_y), puck_radius, (255, 0, 0), -1)
        cv2.rectangle(frame, (paddle_x - paddle_width // 2, paddle_y - paddle_height // 2),
                      (paddle_x + paddle_width // 2, paddle_y + paddle_height // 2), (0, 255, 0), -1)

        # This is for drawing the targets
        for target in targets:
            target_x, target_y = target
            target_region = frame[target_y - 15:target_y + 15, target_x - 15:target_x + 15]

            # Check if the target region has the correct shape (30x30)
            if target_region.shape == (30, 30, 3):
                frame[target_y - 15:target_y + 15, target_x - 15:target_x + 15] = target_image
            else:
                print("Error: Incorrect target region shape")

        # Horizontally flip the frame for the selfie camera view
        frame = cv2.flip(frame, 1)
        '''I have done this to give a mirror-like feeling while playing the game, which is more convenient'''

        # Drawing score and timer on the screen
        elapsed_time = int(time.time() - start_time)
        remaining_time = max(0, time_limit - elapsed_time)
        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {remaining_time}s", (swidth - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Air Hockey Game", frame)

        ''' This is to check for game over conditions; there are two conditions either the time is over or all targets are hit'''
        if not targets or remaining_time == 0:
            if not targets:
                cv2.putText(frame, "Congratulations! All targets hit!", (swidth // 5, sheight // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Game Over. Final Score: {score}", (swidth // 4, sheight // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Air Hockey Game", frame)
            cv2.waitKey(0)
            break

        # For quitting the game (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # To release resources
    cap.release()
    cv2.destroyAllWindows()

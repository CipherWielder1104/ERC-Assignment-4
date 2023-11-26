import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# sets the width and height
video_width=640
video_height=480
cap.set(3,video_width)  
cap.set(4,video_height)

x_r , y_r= (30,610 ), (30, 450) #create a range after which the target cant be loaded

def rem_arr(arr,i ): # This function is used to remove a row 'i' from the 2d array . array->list->remove_element->array
    arr=arr.tolist()
    x=arr.index(i.tolist())
    arr.pop(x)
    arr= np.array(arr)
    return arr

def gameover(): #Function to check for game over 
    if elapsed_t >= game_duration or no_of_hits == tcords:
        return True
    else:
        return False
    
def c_cords(text,font,font_scale,font_thickness): # defining centre cords 
   text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
   cen_x = (frame.shape[1] - text_size[0]) // 2
   cen_y = (frame.shape[0] + text_size[1]) // 2
   return (cen_x, cen_y)

# Draw the text

def draw_puck(frame, puck_p): #puck draw 
    puck_radius = 10
    center = (int(puck_p[0]), int(puck_p[1]))
    cv2.circle(frame, center, puck_radius,(255,0,0), cv2.FILLED)

# Initialize hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,min_detection_confidence=0.75,min_tracking_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

# Initialize paddle and puck positions

paddle_position = [0,0] 
def pos_puck():
    random_puck_pos_x=np.random.randint(50, 600,1)
    random_puck_pos_y=np.random.randint(50, 400,1)
    puck_pos=[random_puck_pos_x[0],random_puck_pos_y[0] ]
    return puck_pos
puck_pos=pos_puck()

# Initial velocity
initial_puck_velocity = [10, 10] 
puck_velocity = initial_puck_velocity.copy()

# Load target image and resize it to 30,30
target=r"Targent.png location" 
point_img= cv2.imread(target , cv2.IMREAD_UNCHANGED)
target_i=cv2.resize(point_img , (30,30))

# Initialize 5 target positions randomly(remember assignment 2!!)
tcords = 5
target_positions = np.random.randint(low=(x_r[0], y_r[0]), high=(x_r[1], y_r[1]), size=(tcords, 2)) # Generated random target positions using NumPy

# Initialize score
score = 0
no_of_hits = 0

# Initialize timer variables
start_time = time.time()
game_duration = 30  # 1/2 minute in second

# Function to check if the puck is within a 5% acceptance region of a target

def is_within_acceptance(puck, target, acceptance_percent=5):
    # Calculate the distance between the puck and the target
    for i in range (0,tcords):
        distance = (((puck[0] - target[0]) ** 2) + ((puck[1] - target[1]) ** 2)) ** 0.5
        # Calculate the acceptance distance
        acceptance_distance = 30 + 30* (acceptance_percent / 100.0)
        # Check if the puck is within the acceptance area
    if distance <= acceptance_distance:
            return True
    else:
            return False


while True:
    pressedKey = cv2.waitKey(1) & 0xFF

    elapsed_t= time.time() - start_time
    remaining_t= game_duration - elapsed_t
    rem_time_min=int(remaining_t//60)
    rem_time_sec=int(remaining_t%60)

    success, frame = cap.read()  # Read a frame from the webcam 
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB
    results = hands.process(frameRGB)    # Process the frame with mediapipe hands

    if results.multi_hand_landmarks:
        # Extract index finger tip position
        index_tip = results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        
        # Scale the coordinates based on video dimensions
        x, y = int(index_tip.x * video_width), int(index_tip.y * video_height)

        # Update paddle position based on index finger tip
        paddle_position[0] = x  # Update x-axis position
        paddle_position[1] = y  # Update y-axis position

    # Update puck position based on its velocity
    if not gameover():
        draw_puck(frame, puck_pos)
        puck_pos[0] += puck_velocity[0]
        puck_pos[1] += puck_velocity[1]

        # Check for collisions with the walls
        if puck_pos[0] <= 0 or puck_pos[0] >= video_width:
            puck_velocity[0] *= -1  # Reflect the puck if it hits the left or right wall
        elif puck_pos[1] <= 0 or  puck_pos[1]>=video_height:
            puck_velocity[1] *= -1  # Reflect the puck if it hits the top wall

        # Check for collisions with the paddle
        if (
        paddle_position[1] - 20 <= puck_pos[1] <= paddle_position[1] + 20
        and paddle_position[0] - 20 <= puck_pos[0] <= paddle_position[0] + 20):
            puck_velocity[1] *= -1
    
        # Check for collisions with the targets(use is_within_acceptance)

        for target_pos in target_positions:
            if is_within_acceptance(puck_pos, target_pos , acceptance_percent=5)==True:
                # Puck hit the target
                score += 1
                no_of_hits += 1
                puck_velocity[0] += 2
                puck_velocity[1] += 2
                target_positions = rem_arr(target_positions,target_pos ) #removes the cords of the hit target using the function defined earlier 

                # Increase the player's score
                # Remove the hit target
                # Increase puck velocity after each hit by 2(you will have to increase both x and y velocities

        # Draw paddle, targets on the frame and overlay target image on frame .
        
        #Darwing the paddle (circle)
        if results.multi_hand_landmarks and results.multi_hand_landmarks[0].landmark:
            paddle_center = (paddle_position[0], paddle_position[1])
            cv2.circle(frame, paddle_center , 20 ,(0,255,0), cv2.FILLED)

        #Tried to draw a rectangular paddle . But faced some issue .
        '''paddle_width, paddle_height = 70, 20  # paddle size as 70 * 30  
        paddle_top_left = (int(paddle_position[0] - paddle_width / 2), int(paddle_position[1] - paddle_height / 2))
        paddle_bottom_right = (int(paddle_position[0] + paddle_width / 2), int(paddle_position[1] + paddle_height / 2))
        cv2.rectangle(frame, paddle_top_left, paddle_bottom_right, (0, 255, 0), cv2.FILLED)'''
        
        #target frame and target positioning 
        for target_position in target_positions :
            target_roi = frame[target_position[1]:target_position[1] + target_i.shape[0],
                            target_position[0]:target_position[0] + target_i.shape[1]]
            alpha = target_i[:, :, 3] / 255.0
            beta = 1.0 - alpha
            for c in range(0, 3):
                target_roi[:, :, c] = (alpha * target_i[:, :, c] +
                                    beta * target_roi[:, :, c])

        cv2.putText(frame, f"Score: {score}", (25 , 40 ), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255), 1) # Display the player's score on the frame

        cv2.putText(frame, f"Time Remaining : {rem_time_min} minutes {rem_time_sec} seconds ", (25 , 25 ), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255), 1)  # Display the remaining time on the frame
        
        if pressedKey == ord('r'):  #Press R to Reload 
            # Reset the games ( variables to start aganin )
            start_time = time.time()
            score = 0
            no_of_hits = 0
            puck_pos=pos_puck()
            puck_velocity = initial_puck_velocity.copy()
            target_positions = np.random.randint(low=(x_r[0], y_r[0]), high=(x_r[1], y_r[1]), size=(tcords, 2))

    if gameover():   #when game is over
        font=cv2.FONT_HERSHEY_DUPLEX
        if no_of_hits == tcords:
            won_text="WON"
            c_cord_1=c_cords(won_text,font,3,3)
            cv2.putText(frame, "WON", c_cord_1, font, 3, (41,134,204), 3)
            cv2.putText(frame, "Press 'q' to quit and 'r' to restart ", (20,20 ), font, 1, (30,144,255), 1)
        else:
            game_over_text= "Game Over .Time Up !!"
            c_cord_2=c_cords(game_over_text,font,3,3)
            cv2.putText(frame,"Game Over .Time Up !!", (120,240), font, 1, (250,48,51), 1)
            cv2.putText(frame, "Press 'q' to quit and 'r' to restart ", (20,20 ), font, 1, (30,144,255), 1)
        if pressedKey == ord('r'): #Play game again . Press 'r'
            # Reset the games ( variables to start aganin )
            start_time = time.time()
            score = 0
            no_of_hits = 0
            puck_pos=pos_puck()
            puck_velocity = initial_puck_velocity.copy()
            target_positions = np.random.randint(low=(x_r[0], y_r[0]), high=(x_r[1], y_r[1]), size=(tcords, 2))
            
    cv2.imshow('Hand Detection', frame)  # Display the resulting frame

    if  pressedKey == ord('q'): # Exit the game when 'q' is pressed
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
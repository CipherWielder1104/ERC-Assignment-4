import numpy as np 
import cv2 
import time
import mediapipe as mp 
import math 
from math import ceil 
import random 

class AirHockey():
    def __init__(self, mode = False, maxHands=2, modelComplexity = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.modelComplex = modelComplexity
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands= mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw = False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)


        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmark, self.mpHands.HAND_CONNECTIONS)

        return img   

    def findPositon(self, img, handNo=0, draw=True):
        lmList = np.zeros(2, dtype='int32') 
        if self.results.multi_hand_landmarks:
            theHand = self.results.multi_hand_landmarks[handNo]
        
            for id, lm in enumerate(theHand.landmark):
                h,w,c = img.shape
                if id == 8:
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList[0] = cx
                    lmList[1] = cy 
                    cv2.rectangle(img, (cx-50,cy-15), (cx+50, cy+15), (0,250,250), cv2.FILLED)
                        
        return lmList
    

class Puck():
    frame_h = 720
    frame_w = 1280
    reg1 = True
    reg2 = True
    reg3 = True
    reg4 = True
    reg5 = True

    def __init__(p):
        p.vel = np.array([90,90])
        p.dirX = 1
        p.dirY = 1
        p.dv = np.array([[10,10], [10,-10], [-10,10], [-10,-10]], dtype='int32')
        p.var = np.array([True, False, False, False])

    def velocity(p, img, x,y, target, mask):
        cv2.circle(img, (p.vel[0], p.vel[1]), 10, (0,0,250), -1)
        
        #Bounce Hit Top
        if(y<p.vel[1]<y+40 and x-50<p.vel[0]<x+50):
            if(p.dirX == 1):
                for i in range(4):
                    if i == 1:
                        p.var[i] = True
                        p.dirY = -p.dirY
                    else: 
                        p.var[i] = False
            elif(p.dirX == -1):
                for i in range(4):
                    if i == 3:
                        p.var[i] = True
                        p.dirY = -p.dirY
                    else: 
                        p.var[i] = False
    
             

        #Bottom 
        if((p.vel[1]+10) == p.frame_h):
            if(p.dirX ==1):
                for i in range(4):
                    if i == 1:
                        p.var[i] = True
                        p.dirY = -p.dirY
                    else: 
                        p.var[i] = False
            elif(p.dirX == -1):
                for i in range(4):
                    if i == 3:
                        p.var[i] = True
                        p.dirY = -p.dirY
                    else: 
                        p.var[i] = False


            #Right
        if((p.vel[0]+10) == p.frame_w):
            if(p.dirY == -1):
                for i in range(4):
                    if i == 3:
                        p.var[i] = True
                        p.dirX = -p.dirX
                    else: 
                        p.var[i] = False
            elif(p.dirY == 1):
                for i in range(4):
                    if i == 2:
                        p.var[i] = True
                        p.dirX = -p.dirX
                    else: 
                        p.var[i] = False


            #Top
        if((p.vel[1]-10) == 0):
            if(p.dirX == -1):
                for i in range(4):
                    if i == 2:
                        p.var[i] = True
                        p.dirY = -p.dirY
                    else: 
                        p.var[i] = False
            elif(p.dirX == 1):
                for i in range(4):
                    if i == 0:
                        p.var[i] = True
                        p.dirY = -p.dirY
                    else: 
                        p.var[i] = False

            #Left
        if((p.vel[0]-10) == 0):
            if(p.dirY == -1):
                for i in range(4):
                    if i == 1:
                        p.var[i] = True
                        p.dirX = -p.dirX
                    else: 
                        p.var[i] = False
            elif(p.dirY == 1):
                for i in range(4):
                    if i == 0:
                        p.var[i] = True
                        p.dirX = -p.dirX
                    else: 
                        p.var[i] = False


        for i in  range(4): 
            if p.var[i] == True:
                p.vel = p.vel + p.dv[i]
                break 
        return p.vel
        


def main():
    cap = cv2.VideoCapture(0)

    start_time = time.time() + 30
    delta = math.ceil(start_time - time.time())

    px = random.randint(100,700)
    py = random.randint(100,700)
    # Read logo and resize 
    target = cv2.imread('/Users/sarthak/Desktop/Python AI:ML/git/ERC-Assignment-4_Sarthak/target.png') 
    target = cv2.resize(target, (30,30)) 
    game_over = False

    # Create a mask of logo 
    img2gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) 
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY) 

    puck = Puck()
    bounce = AirHockey()
    lmList = [5,5]

    
    reg1 = True
    reg2 = True
    reg3 = True
    reg4 = True
    reg5 = True
    Score = 0
    count1 = True
    count2 = True
    count3 = True
    count4 = True
    count5 = True
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)


        
        #Timer
        if(delta >=0):
            if(delta>=0):
                cv2.putText(frame, "Time: "+str(delta), (30,50),1, cv2.FONT_HERSHEY_COMPLEX, (0,100,250),2)
                if(Score == 5):
                    cv2.putText(frame, "GAME OVER", (300,400),2, cv2.FONT_HERSHEY_COMPLEX, (0,100,250),3)
        elif(delta<0): 
            if (delta>-6 or game_over == True):
                cv2.putText(frame, "GAME OVER", (300,400),2, cv2.FONT_HERSHEY_COMPLEX, (0,100,250),3)

            else:
                break
        
        delta = math.ceil(start_time - time.time())

        bounce.findHands(frame)
        lmList = bounce.findPositon(frame)
        cx = lmList[0]-5
        cy = lmList[1]-5
        vel = puck.velocity(frame, cx, cy, target, mask)
        

        roi1 = frame[290:320, 550:580]
        roi2 = frame[400:430, 600:630]
        roi3 = frame[100:130, 460:490]
        roi4 = frame[600:630, 600:630]
        roi5 = frame[500:530, 310:340]
        
        
        if(550-20<=vel[0]<=580+20 and 290-20<vel[1]<320+20):
            if(count1 == True):
                reg1 = False
                Score = Score +1 
                count1 = False   
        elif(reg1 == True):
            roi1[np.where(mask)] = 0
            roi1 += target

        if(600-20<=vel[0]<=630+20 and 400-20<vel[1]<430+20):
            if(count2 == True):
                reg2 = False
                Score = Score +1
                count2 = False
        elif(reg2 == True):
            roi2[np.where(mask)] = 0
            roi2 += target
        
        if(460-20<=vel[0]<=490+20 and 100-20<vel[1]<130+20):

            if(count3 == True):
                reg3 = False
                Score = Score + 1
                count3 = False
        elif(reg3 == True):
            roi3[np.where(mask)] = 0
            roi3 += target

        if(600-20<=vel[0]<=630+20 and 600-10<=vel[1]<=630+20):
            if(count4 == True):
                reg4 = False
                Score = Score+1
                count4 = False
        elif(reg4 == True):
            roi4[np.where(mask)] = 0
            roi4 += target

        if(310-20<=vel[0]<=340+20 and 500-20<vel[1]<530+20):
            if(count5 == True):
                count5 = False
                reg5 = False
                Score = Score+1
        elif(reg5 == True):
            roi5[np.where(mask)] = 0
            roi5 += target


        cv2.putText(frame, "Score: "+str(Score), (800,50),1, cv2.FONT_HERSHEY_COMPLEX, (100,100,100),2)
        

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'): 
            break
    
        
    cap.release() 
    cv2.destroyAllWindows() 
    
if __name__ == "__main__":
    main()


import cv2
import numpy as np

yellow_team_count = 0
red_team_count = 0

def identify_players(frame):
    
    global yellow_team_count, red_team_count
     
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #limits
    yellow_lower = np.array([5, 0, 100])
    yellow_upper = np.array([67, 255, 255])
    red_lower = np.array([175, 137, 100])
    red_upper = np.array([178, 255, 255])
    
    #mask without morphing
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    #masks with morphing
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow("Post-Morph Red Mask", red_mask)
    
    #finding contours
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    yellow_count = 0
    red_count = 0
    
    for cnt in yellow_contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            #find the centre of the contours with yellow and area > 500
            center = (x + w // 2, y + h // 2)
            cv2.circle(frame, center, 5, (0, 255, 255), -1)
            yellow_count += 1
    
    for cnt in red_contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            #find the centre of the contours with red and area > 500
            center = (x + w // 2, y + h // 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            red_count += 1
    
    yellow_team_count += yellow_count
    red_team_count += red_count
    
    return yellow_count, red_count

def edited(frame):

    bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #the template to match - ball.png is a same-sized image of the volleyball. its been used as grayscale
    template = cv2.imread("ball.png", 0)

    (h, w) = template.shape

    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    bw2 = bw.copy()
    
    #choosing a method of template matching, from methods.
    method = methods[1]

    result = cv2.matchTemplate(bw2, template, method)

    minv, maxv, minl, maxl = cv2.minMaxLoc(result)
    
    y, r = identify_players(frame)
    
    text = f"Red : {r}, Yellow : {y}"
    cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    #sqdiff requires us to use the lower, the others require us to use the upper
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = minl
    else:
        location = maxl
        
    #threshold of 0.8 
    if maxv >= 0.9:
        btm_rt = (location[0] + w, location[1] + h)
        cv2.rectangle(frame, location, btm_rt, 255, 5)

    return frame
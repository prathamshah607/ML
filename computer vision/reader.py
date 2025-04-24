import cv2
from modifier import edited

#capturing the volleyball video
cap = cv2.VideoCapture("volleyball_match.mp4")

if not cap.isOpened():
    print("CRITICAL ERROR: Cannot open file/camera system.")
    exit()

# vid deets - fps, w, h
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# to write the edited frame
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('VOLLEYBALL_OUTPUT.avi', fourcc, fps, (width, height))

paused = False
last_frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            
        last_frame = frame
        
    elif last_frame is not None:
        frame = last_frame
    
    #applies ALL opencv methods and returns edited(frame)
    edited_frame = edited(frame)
    
    
    #live video showing
    cv2.imshow('frame', edited_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    #quits on q, pauses on space
    if key == ord('q'):
        break
    elif key == 32:
        paused = not paused
    
    if not paused and last_frame is not None:
        out.write(edited_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
import cv2
from PIL import Image

for i in range(12):
     try:
         cap = cv2.VideoCapture(i)  # 0 is the default webcam
         ret, frame = cap.read()
         if ret:
            print(i)
     except Exception as e:
         continue

def preprocess_frame(frame, cam_id):
    
        cam_id = cam_id -1 

        webcam_crop = [[720,720,280,0],[640,640,310,80]]
        
        cropped = frame[webcam_crop[cam_id][3]:webcam_crop[cam_id][3] + webcam_crop[cam_id][1], webcam_crop[cam_id][2]:webcam_crop[cam_id][2] + webcam_crop[cam_id][0]]
        print(f"cropped:{cropped.shape} frame{frame.shape}")
        frame = cv2.resize(cropped, (480, 480), interpolation=cv2.INTER_LINEAR)
        return frame   
        
cap = cv2.VideoCapture(10) 
frame_size=(1280, 800)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = preprocess_frame(frame, 2)    
    cv2.imshow("Camera Feed", frame)
    ret, frame = cap.read()
  
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
cap.release()
cv2.destroyAllWindows()

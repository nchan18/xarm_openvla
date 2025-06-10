import cv2
from PIL import Image

cap = cv2.VideoCapture(6)  # 0 is the default webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera Feed", frame[125:550, 132:472])
    #ret, frame = cap.read()
    #bgr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #image: Image.Image = Image.fromarray(frame)
    #image.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
cap.release()
cv2.destroyAllWindows()

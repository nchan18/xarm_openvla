import apriltag
import cv2
cap = cv2.VideoCapture(7)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
detector = apriltag.Detector()
result = detector.detect(gray_frame)
print(result)
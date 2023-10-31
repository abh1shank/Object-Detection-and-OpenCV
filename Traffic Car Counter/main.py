import cvzone
from ultralytics import YOLO
import cv2
from sort import *
import math
path=r'C:\Users\saran\Desktop\MLDL\pythonProject\Object-Detection-101\Videos\cars.mp4'
mask=cv2.imread(r'C:\Users\saran\Desktop\MLDL\pythonProject\Object-Detection-101\Project 1 - Car Counter\mask.png')
mask=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
cap = cv2.VideoCapture(path)
model = YOLO('yolov8n.pt')
limits = [400, 297, 673, 297]
totalCount=[]
cap.set(3, 1000)
cap.set(4, 1000)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
d = {0: '__background__', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'}
while True:
    ret, frame = cap.read()
    masked_frame=frame&mask;
    results = model(masked_frame, stream=True)
    detections=np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for b in boxes:
            x, y, xx, yy = b.xyxy[0]
            x, y, xx, yy = int(x), int(y), int(xx), int(yy)
            w,h=abs(xx-x),abs(yy-y)
            typee = d[int(b.cls[0]+1)]
            conf = int(b.conf[0]*100)/100
            if conf>0.1 and typee=='car':
               currentArray=np.array([x,y,xx,yy,conf])
               detections=np.vstack((detections,currentArray))

    trackresult=tracker.update(detections)

    for result in trackresult:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx,cy= x+w//2,y+h//2
        if limits[0] < cx < limits[2] and limits[1] - 70 < cy < limits[1] + 70:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        cv2.putText(frame, f'Count: {str(len(totalCount))}', (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 255), 8)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

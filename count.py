import cv2
import numpy as np
from centroidtracker import CentroidTracker


cap = cv2.VideoCapture('video/vehicles_1.mp4')
confthershold = 0.2
videosave = cv2.VideoWriter('./count.avi', cv2.VideoWriter_fourcc(*'MJPG'), 3, (960, 540))

model_cfg = ('model/yolov3-320.cfg')
model_weights = ('model/yolov3-320.weights')
classes = open("model/coco.names","r").read().strip().split('\n')
# print(classes)
tracker = CentroidTracker(maxDisappeared=0, maxDistance=90)

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
count = []


def findObjects(outputs, img):
    global count
    h, w, c = img.shape
    boxes = []
    class_name = []
    conf = []
    rects = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_ids = np.argmax(scores)
            confidence = scores[class_ids]
            if confidence > confthershold:
                W = int(detection[2]*w)
                H = int(detection[3]*h)
                x = int(detection[0]*w-W/2)
                y = int(detection[1]*h-H/2)
                boxes.append([int(i)for i in (x, y, W, H)])
                class_name.append(class_ids)
                conf.append(float(confidence))
                # print(conf)
                # print(len(boxes))

    indices = cv2.dnn.NMSBoxes(boxes, conf, confthershold, nms_threshold=0.3)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, W, H = box
        rects.append(box)
        cv2.rectangle(img, (x, y), (x+W, y+H), (0, 255, 255), 1)
        cv2.putText(img, f'{classes[class_name[i]]}', (x, y-10), 0, 1, (0, 0, 255), 1)

    obj = tracker.update(rects)
    for (objectid,centroid) in obj.items():
        count.append(objectid)
        text = 'ID: {}'.format(objectid)
        # print(text)

    cv2.putText(img, "vehicle_count:"+str(objectid), (0, 30), 0, 1, (0, 0, 0), 2)


while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    if not ret:
        break


    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), 1, crop=False)
    net.setInput(blob)

    layernames = net.getLayerNames()
    # print(layernames)
    output_layernames = [layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(ouput_layernames)

    outputs = net.forward(output_layernames)
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    #cv2.putText(frame, "FPS:" + str(int(fps)), (0, 80), 0, 1, (0, 255, 0), 2)

    findObjects(outputs, frame)
    #videosave.write(frame)
    #result.write(frame)

    cv2.imshow('frame', frame)
    videosave.write(frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
videosave.release()
cv2.destroyAllWindows()
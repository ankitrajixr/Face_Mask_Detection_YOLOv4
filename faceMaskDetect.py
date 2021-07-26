# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
from datetime import datetime 
import pytz 
import cv2
import os




ap = argparse.ArgumentParser()

ap.add_argument("-o", "--output", type=str, default="",help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,help="whether or not output frame should be displayed")
ap.add_argument("-u", "--use-gpu", type=bool, default=0,help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())


# load the class labels 
labelsPath = "yolov4-mask-detector\obj.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
COLORS = [[0,0,255],[0,255,0]]

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolov4-mask-detector\yolov4_face_mask.weights"
configPath = "yolov4-mask-detector\yolov4-obj.cfg"


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# check if we are going to use GPU
if args["use_gpu"]:
    # set CUDA 
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# initialize the width and height of the frames in the video file
W = None
H = None

# initialize the video stream and pointer to output video file, then

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(0)
writer = None
fps = FPS().start()
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (864, 864),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence >0.42:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply NMS to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.45,0.3)

    #Add top-border to frame to display stats
    border_size=100
    border_text_color=[255,255,255]
    frame = cv2.copyMakeBorder(frame, border_size,0,0,0, cv2.BORDER_CONSTANT)
    #calculate count values
    filtered_classids=np.take(classIDs,idxs)
    mask_count=(filtered_classids==1).sum()
    nomask_count=(filtered_classids==0).sum()
    #display count
    text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
    cv2.putText(frame,text, (0, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,border_text_color, 2)
    #display status
    text = "Status:"
    cv2.putText(frame,text, (W-200, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,border_text_color, 2)
    ratio=nomask_count/(mask_count+nomask_count+0.000001)

    
    if ratio>=0.1 and nomask_count>=3:
        text = "Danger !"
        cv2.putText(frame,text, (W-100, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[26,13,247], 2)
        
    elif ratio!=0 and np.isnan(ratio)!=True:
        text = "No Mask"
        cv2.putText(frame,text, (W-100, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[0,255,255], 2)

    else:
        text = "Mask "
        cv2.putText(frame,text, (W-100, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[0,255,0], 2)
    

 
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1]+border_size)
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)


    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 24,(frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)
    # update the FPS counter
    fps.update()

fps.stop()
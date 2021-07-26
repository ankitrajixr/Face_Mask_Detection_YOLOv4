from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import time

app = Flask(__name__)

def maskImage(image_path):
        # load the class labels 
    labelsPath = "yolov4-mask-detector\obj.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible label
    COLORS = [[0,0,255],[0,255,0]]

    # derive the paths to the YOLO weights and model configuration
    weightsPath = "yolov4-mask-detector\yolov4_face_mask.weights"
    configPath = "yolov4-mask-detector\yolov4-obj.cfg"

    # load our YOLO object detector 
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and get it height and width
    imagePath = image_path
    image = cv2.imread(imagePath)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()

    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (832, 832),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln) #list of 3 arrays, for each output layer.
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))


    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:

        # loop over each of the detections
        for detection in output:

            scores = detection[5:] 
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.45:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply NMS to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.45,0.3)
    border_size=100
    border_text_color=[255,255,255]
    #Add top-border to image to display stats
    image = cv2.copyMakeBorder(image, border_size,0,0,0, cv2.BORDER_CONSTANT)
    #calculate count values
    filtered_classids=np.take(classIDs,idxs)
    mask_count=(filtered_classids==1).sum()
    nomask_count=(filtered_classids==0).sum()
    #display count
    text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
    cv2.putText(image,text, (0, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,border_text_color, 2)
    #display status
    text = "Status:"
    cv2.putText(image,text, (W-300, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,border_text_color, 2)
    ratio=nomask_count/(mask_count+nomask_count)

    if ratio>=0.1 and nomask_count>=3:
        text = "No Mask !"
        cv2.putText(image,text, (W-200, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,[26,13,247], 2)
        
    elif ratio!=0 and np.isnan(ratio)!=True:
        text = "Warning !"
        cv2.putText(image,text, (W-200, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,[0,255,255], 2)

    else:
        text = "Mask is there "
        cv2.putText(image,text, (W-200, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.8,[0,255,0], 2)
    return(text)

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = maskImage(img_path)

	return render_template("aboutproj.html", prediction = p, img_path = img_path)

@app.route('/exec2')
def parse(name=None):
    import faceMaskDetect
    return render_template('about.html',name=name)

# routes
@app.route("/exec", methods=['GET', 'POST'])
def parse1(name=None):
    return render_template('aboutproj.html',name=name)


if __name__ == '__main__':
    app.run()
    app.debug = True
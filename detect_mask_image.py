import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import load_img , img_to_array
import numpy as np
import os
import matplotlib.pyplot as plt
def mask_detection_prediction(frame, faceNet, maskNet):

    # find the dimension of frame and construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # create a empty list which'll store list of faces,face location and prediction
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        
        # find the confidence or probability associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter the strong detection [confidence > min confidence(let 0.5)]
        if confidence > 0.5:

            # find starting and ending coordinates of boundry box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # make sure bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # append the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding prediction
    return (locs, preds)






from os.path import dirname, join

prototxtPath = join("face_detector", "deploy.prototxt")
weightsPath = join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("fmd.h5")
inpimage = input(""" Enter the image filename 
      Note : Input full image path : """)
image = cv2.imread(inpimage)
blob = cv2.dnn.blobFromImage(image,1.0,(224,224),(104.0,177.0,123.0))
(locs, preds) = mask_detection_prediction(image, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
    (startX, startY, endX, endY) = box
    (mask, withoutMask) = pred
#detecting faces

    if mask>withoutMask:
        class_label = "Mask"
        color = (0,255,0)
            
    else:
        class_label = "No Mask"
        color = (0,0,255)
        

        #display the label and bounding boxes

    cv2.putText(image,class_label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
    cv2.rectangle(image,(startX,startY),(endX,endY),color,2)


cv2.imshow("OutPut",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
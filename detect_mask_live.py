
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from tkinter import messagebox
import tkinter 
root = tkinter.Tk()
root.withdraw()
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage

from pygame import mixer
mixer.init()
sound = mixer.Sound('mixkit-alarm-tone-996.wav')


# ### Image Pre-Processing

# In[5]:


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


# ### Load Caffe Model
# Caffe (Convolutional Architecture for Fast Feature Embedding) is a deep learning framework that allows users to create image classification and image segmentation models. It is a Caffe model which is based on the Single Shot-Multibox Detector (SSD) and uses ResNet-10 architecture as its backbone. It was introduced post OpenCV 3.3 in its deep neural network module.

# In[6]:


# load our serialized face detector model from disk
from os.path import dirname, join

prototxtPath = join("face_detector", "deploy.prototxt")
weightsPath = join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("fmd.h5")


# ### Face Detection on Live Camera

# In[10]:


# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = mask_detection_prediction(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        if mask>withoutMask:

            label = "Mask"
            color = (0, 255, 0)
            print("Normal")
        else:
            label = "No Mask"
            color = (0, 0, 255)
            sound.play()
            print("Alert!!!")
            messagebox.showwarning("Warning","Please Wear mask to get access .By Pressing OK ,You are acknowledging that you have violated a Covid 19 Protocol")
            #ret, fram = vs.read()
            ts = datetime.datetime.now()
            img_name = "withoutmask.png"
            print(img_name)
            cv2.imwrite(img_name, frame)

        # e-mail the captured image to the respective department
        
        

        # The mail addresses and password
            sender_address = 'facemaskauthority_jit@outlook.com'
            sender_pass = 'facejit123'
            receiver_address = 'sricharanbellur@gmail.com'

        # Setup the MIME
            message = MIMEMultipart()
            message['From'] = sender_address
            message['To'] = receiver_address
            message['Subject'] = 'Person without a mask.'
        
    
            msgText = MIMEText('''Hello,
        
    Identified a person without a mask.
    Please check the attachment.
        
    Thank You
    ''')
        
            message.attach(msgText)
            fp = open('withoutmask.png', 'rb') #Read image 
            msgImage = MIMEImage(fp.read())
            fp.close()
            msgImage.add_header('001', 'withoutmask.png')
            message.attach(msgImage)
        # Create SMTP session for sending the mail
            session = smtplib.SMTP('smtp.office365.com', 587)  # use gmail with port
            session.ehlo()
            session.starttls()
            session.login(sender_address, sender_pass)  # login with mail_id and password
            text = message.as_string()
            session.sendmail(sender_address, receiver_address, text)
            session.quit()
            messagebox.showwarning("IMPORTANT","Respected authority have been informed about Covid -19 Protocol violation")
            print('Mail Sent')
            
            

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
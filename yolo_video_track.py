# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from pyimagesearch.trackableobject2 import TrackableObject2
from pyimagesearch.trackableobject3 import TrackableObject3
from pyimagesearch.trackableobject4 import TrackableObject4
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import dlib
import pandas as pd
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,    help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
ap.add_argument("-s", "--skip-frames", type=int, default=5, help="# of skip frames between detections")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])

writer = None
(W, H) = (None, None)

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=30)
trackers = []
trackableObjects = {}
trackableObjects2 = {}
trackableObjects3 = {}
trackableObjects4 = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown_in = 0
totalDown_out = 0
totalUp_out = 0
totalUp_in = 0
frameCount = 0
Road_Length = 10
df = pd.DataFrame(columns = ["TotalCount_in","CummD_totalDown_in","TotalCount_out","CummD_totalDown_out","Density_Down"])
df.loc[len(df)] = [0,0,0,0,0]
z = 1

mem_flow_rate_down_in = 0
mem_flow_rate_down_out = 0
mem_Density_down = 0

print(df)

# start the frames per second throughput estimator
fps = FPS().start()

# try to determine the total number of frames in the video file
try:
    if imutils.is_cv2():
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
    else:
        prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
    
#determine FPS of Video Stream
frames_PS= int(vs.get(cv2.CAP_PROP_FPS))

print (frames_PS)

plt.ion()

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    frame = imutils.resize(frame, width=1366)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    
    #cv2.imshow('frame', frame)

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    frameCount +=1
    
        
    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []
    
    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False) 
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
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
                
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"] and classID == 2:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])
        
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x+w, y+h)
                tracker.start_track(rgb, rect)
                
                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)
                
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"
            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))
    
    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(frame, (0, int(H * 0.40)), (W, int(H * 0.40)), (0, 255, 255), 2)
    cv2.line(frame, (0, int(H * 0.50)), (W, int(H * 0.50)), (0, 255, 255), 2)
    
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)
    
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        Up_out = trackableObjects.get(objectID, None)
        Down_in = trackableObjects2.get(objectID, None)
        Up_in = trackableObjects3.get(objectID, None)
        Down_out = trackableObjects4.get(objectID, None)
        
        
        # if there is no existing trackable object, create one
        if Up_out is None:
            Up_out = TrackableObject(objectID, centroid)
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in Up_out.centroids]
            direction = centroid[1] - np.mean(y)
            Up_out.centroids.append(centroid)
            # check to see if the object has been counted or not
            if not Up_out.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < int(H * 0.40):
                    totalUp_out += 1
                    Up_out.counted = True
        # store the trackable object in our dictionary
        trackableObjects[objectID] = Up_out
        
        
        if Down_in is None:
            Down_in = TrackableObject2(objectID, centroid)
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in Down_in.centroids]
            direction = centroid[1] - np.mean(y)
            Down_in.centroids.append(centroid)
            # check to see if the object has been counted or not
            if not Down_in.counted:
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                if direction > 0 and centroid[1] > int(H * 0.40):
                    totalDown_in += 1
                    Down_in.counted = True
        # store the trackable object in our dictionary
        trackableObjects2[objectID] = Down_in
        
        
        if Up_in is None:
            Up_in = TrackableObject3(objectID, centroid)
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in Up_in.centroids]
            direction = centroid[1] - np.mean(y)
            Up_in.centroids.append(centroid)
            # check to see if the object has been counted or not
            if not Up_in.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < int(H * 0.50):
                    totalUp_in += 1
                    Up_in.counted = True
        # store the trackable object in our dictionary
        trackableObjects3[objectID] = Up_in
        
        
        if Down_out is None:
            Down_out = TrackableObject4(objectID, centroid)
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in Down_out.centroids]
            direction = centroid[1] - np.mean(y)
            Down_out.centroids.append(centroid)
            # check to see if the object has been counted or not
            if not Down_out.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction > 0 and centroid[1] > int(H * 0.50):
                    totalDown_out += 1
                    Down_out.counted = True
        # store the trackable object in our dictionary
        trackableObjects4[objectID] = Down_out
        
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
    # construct a tuple of information we will be displaying on the
    # frame
    info_line1 = [("Up", totalUp_out), ("Down", totalDown_in), ("Status", status)]
    info_line2 = [("Up", totalUp_in), ("Down", totalDown_out), ("Status", status)]
    
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info_line1):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info_line2):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (W // 2, H - ((i * 20) + 20)),    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    #calculate flow rate and display value in frame
    if frameCount % frames_PS == 0:
        df.loc[z] = [totalDown_in, (totalDown_in - df.iloc[(z-1, 0)]), totalDown_out, (totalDown_out - df.iloc[(z-1, 2)]), (totalDown_in - totalDown_out)]
        print (df)
        cv2.putText(frame, "Flow_Rate_in (Veh/S): " + str(df.iloc[(z, 1)]), (10, H - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Flow_Rate_out (Veh/S): " + str(df.iloc[(z, 3)]), (W // 2, H - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Density_Down (Veh/Km): " + str(df.iloc[(z, 4)]), (W // 2, H // 2 - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        mem_flow_rate_down_in = df.iloc[(z, 1)]
        mem_flow_rate_down_out = df.iloc[(z, 3)]
        mem_Density_down = df.iloc[(z, 4)]
        z +=1
        #show a plot of flow against density
        plt.scatter(df["CummD_totalDown_out"],df["Density_Down"], label = "My Test")
        plt.title("Flow Curve")
        plt.xlabel("Density(Veh/Km)")
        plt.ylabel("Flow(Veh/s)")
        plt.draw()
        plt.pause(0.001)
    else:
        cv2.putText(frame, "Flow_Rate_in (Veh/S): " + str(mem_flow_rate_down_in), (10, H - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Flow_Rate_out (Veh/S): " + str(mem_flow_rate_down_out), (W // 2, H - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Density_Down (Veh/Km): " + str(mem_Density_down), (W // 2, H // 2 - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        
    
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write the output frame to disk
    writer.write(frame)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()
    
plt.show()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# release the file pointers
print("[INFO] cleaning up...")
df.to_csv("Flow_Data.csv")
writer.release()
vs.release()
cv2.destroyAllWindows()
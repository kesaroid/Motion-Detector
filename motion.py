# import necessary packages
import cv2
import time
import pandas
from datetime import datetime

# define parameters for future use
first_frame = None
status_list = [None, None]  # Add 2 items so that it finds nothing when it checks the [-2] index
times = []

# create 2 columns csv file and append date and time
df = pandas.DataFrame(columns=["Start", "End"])

# Video is being captured
video = cv2.VideoCapture(0)

# infinite loop
while True:
    check, frame = video.read()  # Check = true if webcam is capturing image, frame = numpy array of the first frame
    status = 0  # whether or not there is motion

    # Gaussian blur the image after greyscaling it
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (21, 21), 0)

    # Obtain the first frame where there is no movement
    if first_frame is None:
        first_frame = grey
        continue

    # Delta frame and Threshold frames are created from CV2 methods
    delta_frame = cv2.absdiff(first_frame, grey)
    thershold_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thershold_frame = cv2.dilate(thershold_frame, None, iterations=2)

    # Contouring of the image is necessary to smoothen the image (from CV2 documentation). It uses a
    # copy of the threshold frame and approximation is done
    (cnts, _) = cv2.findContours(thershold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filtering out the contours that are greater than 1000
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        # If movement, change status
        status = 1

        # Bonding box rectangle parameter creation from the contours and creating a box
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # append status onto list
    status_list.append(status)

    status_list = status_list[-2:]

    # recoding the start and stop time when the status changes (previous 2 statuses)
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # Output each filter Dialog box
    cv2.imshow("grey Frame", grey)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thershold_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

for i in range(0, len(times), 2):
    # print out start and end time without the indexes by skipping 2 of the start and end times
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows

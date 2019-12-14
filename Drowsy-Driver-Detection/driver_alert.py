from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import winsound


def eyeAspectRatio(eye):
	x1 = dist.euclidean(eye[1], eye[5])
	x2 = dist.euclidean(eye[2], eye[4])
	y = dist.euclidean(eye[0], eye[3])
	eye_aspect_ratio = (x1 + x2) / (2 * y)
	return eye_aspect_ratio

 
max_eye_aspect_ratio = 0.25  # defines the max eye aspect ratio
maxFrame_eye_aspect_ratio = 15 # threshold which defines number of consecutive farmes the eye must be close to set alarm

counter = 0
ALARM_ON = False

# initialize objects
detector = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"C:\Users\Shivangi\Desktop\mini_project\shape_predictor_68_face_landmarks.dat")

# returns list of starting and ending index of
#left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#print((lStart, lEnd))
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#print((rStart, rEnd))

cap = cv2.VideoCapture(0)


while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	driver = detector(gray, 0) # detect faces in the grayscale frame
	for face in driver:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predict(gray, face)
		shape = face_utils.shape_to_np(shape)
	
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		
		leftEAR = eyeAspectRatio(leftEye)
		rightEAR = eyeAspectRatio(rightEye)

		# average the eye aspect ratio together for both eyes
		eye_aspect_ratio_average = (leftEAR + rightEAR) / 2.0

		# check if the average eye aspect ratio is below the threshold
		# if true, increment the blink frame counter
		if eye_aspect_ratio_average < max_eye_aspect_ratio:
			counter += 1

			# if the eyes were closed more than the threshold frame
			if counter >= maxFrame_eye_aspect_ratio:
				# if true play the alarm sound
				if ~ALARM_ON:
					ALARM_ON = True
					winsound.PlaySound("alarm",winsound.SND_FILENAME)
				cv2.putText(frame, "ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# if false, reset the counter and alarm
		else:
			counter = 0
			ALARM_ON = False
			
		cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(eye_aspect_ratio_average), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 

        
	
cap.release()
cv2.destroyAllWindows()


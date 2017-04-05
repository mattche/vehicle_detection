import cv2
import numpy as np

videoFileName = '../videos/video1.mp4'

def detectCars(filename):
	subtractor = cv2.BackgroundSubtractorMOG2(120, 70, True)
	vc = cv2.VideoCapture(filename)
	if (vc.isOpened()):
		res, frame = vc.read()
	else:
		res = False

	while res:
		res, frame = vc.read()
		frameS = cv2.resize(frame, (600,375))
		fgmask = subtractor.apply(frameS)
		
		kernel = np.ones((2,2),np.uint8)
		erode = cv2.erode(fgmask, None, iterations = 2)  
		dilate = cv2.dilate(erode, None, iterations = 10)
		
		(contours, _) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		for c in contours:
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frameS, (x, y), (x+w, y+h), (0, 0, 255), 2)
		
		cv2.imshow('Origin', frameS)
		cv2.imshow('BG', fgmask)
		cv2.imshow('Erode', erode)
		cv2.imshow('Dilate', dilate)
		
		if cv2.waitKey(33) == 27:
			break
	vc.release()
	cv2.destroyAllWindows()

detectCars(videoFileName)

import cv2.cv as cv
from datetime import datetime
import time
import sys

from MotionDetector import MotionDetector

class MotionDetectorInstantaneous(MotionDetector):
    
    def __init__(self, source, threshold=8, doRecord=True, showWindows=True):
        MotionDetector.__init__(self, source=source, threshold=threshold,
                                doRecord=doRecord, showWindows=showWindows)

        self.frame1gray = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t-1
        cv.CvtColor(self.frame, self.frame1gray, cv.CV_RGB2GRAY)
        
        #Will hold the thresholded result
        self.res = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U)
        
        self.frame2gray = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t
        
        self.width = self.frame.width
        self.height = self.frame.height
        self.nb_pixels = self.width * self.height
        
    def processImage(self, frame):
        cv.CvtColor(frame, self.frame2gray, cv.CV_RGB2GRAY)
        
        #Absdiff to get the difference between to the frames
        cv.AbsDiff(self.frame1gray, self.frame2gray, self.res)
        
        #Remove the noise and do the threshold
        cv.Smooth(self.res, self.res, cv.CV_BLUR, 5,5)
        cv.MorphologyEx(self.res, self.res, None, None, cv.CV_MOP_OPEN)
        cv.MorphologyEx(self.res, self.res, None, None, cv.CV_MOP_CLOSE)
        cv.Threshold(self.res, self.res, 10, 255, cv.CV_THRESH_BINARY_INV)

        self.motion_level = self.somethingHasMoved()

    def somethingHasMoved(self):
        nb = self.nb_pixels - cv.CountNonZero(self.res)        
        return 1.0 * nb / self.nb_pixels
        
    def visualize(self):
        return self.res


if __name__=="__main__":
    fname = sys.argv[-1]
    if fname=='cam':
        source = cv.CaptureFromCAM(0)
    else:
        source = cv.CaptureFromFile(sys.argv[-1])

    detector = MotionDetectorInstantaneous(source, doRecord=False)
    detector.run()

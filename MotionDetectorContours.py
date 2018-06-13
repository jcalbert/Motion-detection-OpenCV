import cv2.cv as cv
from datetime import datetime
import time
import sys

from MotionDetector import MotionDetector

class MotionDetectorAdaptative(MotionDetector):

    def __init__(self, source, threshold=25, doRecord=True, showWindows=True):
        MotionDetector.__init__(self, source=source, threshold=threshold,
                                doRecord=doRecord, showWindows=showWindows)
    


        self.gray_frame = cv.CreateImage(cv.GetSize(self.frame), cv.IPL_DEPTH_8U, 1)
        self.average_frame = cv.CreateImage(cv.GetSize(self.frame), cv.IPL_DEPTH_32F, 3)
        self.absdiff_frame = None
        self.previous_frame = None
        
        self.surface = self.frame.width * self.frame.height
        self.currentsurface = 0
        self.currentcontours = None

    def processImage(self, curframe):
        cv.Smooth(curframe, curframe) #Remove false positives
        
        if not self.absdiff_frame: #For the first time put values in difference, temp and moving_average
            self.absdiff_frame = cv.CloneImage(curframe)
            self.previous_frame = cv.CloneImage(curframe)
            cv.Convert(curframe, self.average_frame) #Should convert because after runningavg take 32F pictures
        else:
            cv.RunningAvg(curframe, self.average_frame, 0.05) #Compute the average
        
        cv.Convert(self.average_frame, self.previous_frame) #Convert back to 8U frame
        
        cv.AbsDiff(curframe, self.previous_frame, self.absdiff_frame) # moving_average - curframe
        
        cv.CvtColor(self.absdiff_frame, self.gray_frame, cv.CV_RGB2GRAY) #Convert to gray otherwise can't do threshold
        cv.Threshold(self.gray_frame, self.gray_frame, 50, 255, cv.CV_THRESH_BINARY)

        cv.Dilate(self.gray_frame, self.gray_frame, None, 15) #to get object blobs
        cv.Erode(self.gray_frame, self.gray_frame, None, 10)

        self.motion_level = self.somethingHasMoved()
            
    def somethingHasMoved(self):
        
        # Find contours
        storage = cv.CreateMemStorage(0)
        contours = cv.FindContours(self.gray_frame, storage, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)

        self.currentcontours = contours #Save contours
        
        while contours: #For all contours compute the area
            self.currentsurface += cv.ContourArea(contours)
            contours = contours.h_next()
        
        avg = 1.0 * self.currentsurface / self.surface #Calculate the average of contour area on the total size
        self.currentsurface = 0 #Put back the current surface to 0
        
        return avg

    def visualize(self):
        vis_frame = cv.CloneImage(self.frame)
        cv.DrawContours(vis_frame, self.currentcontours, (0, 0, 255), (0, 255, 0), 1, 2, cv.CV_FILLED)
        return vis_frame         

        
if __name__=="__main__":
    fname = sys.argv[-1]
    if fname=='cam':
        source = cv.CaptureFromCAM(0)
    else:
        source = cv.CaptureFromFile(sys.argv[-1])

    detector = MotionDetectorAdaptative(source, doRecord=False)
    detector.run()

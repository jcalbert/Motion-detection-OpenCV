try:
    import cv2.cv as cv
except:
    pass
import cv2
from datetime import datetime
import time
import sys

from MotionDetector import MotionDetector, MotionQuantifier
import numpy as np

from scipy.ndimage import maximum_filter1d

CV2 = True

EWMA_ALPHA = 0.05 #Factor for exponentially accumulated motion blur

DILATE_ITERS = 15 #Number of 3x3 dilation iters
ERODE_ITERS = 10 #Number of 3x3 dilation iters


CONTOUR_COLOR = (0,0,255)
HOLE_COLOR = (0, 255, 0)

class ContourQuantifier(MotionQuantifier):
 
    def __init__(self, init_frame):
        MotionQuantifier.__init__(self, init_frame)

        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.average_frame = self.frame * 1.0
        self.absdiff_frame = self.frame * 1.0
        self.previous_frame = self.frame * 1.0
        self.area = self.gray_frame.size
        self.currentsurface = 0
        self.currentcontours = None
    
    def step(self, curframe):
        cv2.GaussianBlur(curframe, (3,3), sigmaX=0, sigmaY=0, dst=curframe) #Remove false positives
        cv2.accumulateWeighted(curframe, self.average_frame, EWMA_ALPHA) #Compute the average
        
        self.previous_frame = self.average_frame.astype('u1') #Convert back to 8U frame
        
        self.absdiff_frame = cv2.absdiff(curframe, self.previous_frame) # moving_average - curframe

        self.gray_frame = cv2.cvtColor(self.absdiff_frame, cv2.COLOR_BGR2GRAY) #Convert to gray
        _,self.gray_frame = cv2.threshold(self.gray_frame, 50, 255, cv2.THRESH_BINARY)

        self.gray_frame = cv2.dilate(self.gray_frame,
                                     np.ones((3,3),np.uint8),
                                     iterations = DILATE_ITERS) #to get object blobs

        self.gray_frame = cv2.erode(self.gray_frame,
                                     np.ones((3,3),np.uint8),
                                     iterations = ERODE_ITERS) #to get object blobs

        _,contours, tree = cv2.findContours(self.gray_frame,
                                          mode = cv2.RETR_EXTERNAL,
                                          method = cv2.CHAIN_APPROX_SIMPLE)
        moving_area = sum(map(cv2.contourArea, contours))

        self.contours = contours
        self.contour_tree = tree #Save contours
        avg = 1.0 * moving_area / self.area #Calculate the average of contour area on the total size

        self.motion_level = avg
        return self.motion_level

class MotionDetectorAdaptative(MotionDetector):

    def __init__(self, source, threshold=0.2, doRecord=True, showWindows=True):
        MotionDetector.__init__(self, source=source, threshold=threshold,
                                doRecord=doRecord, showWindows=showWindows)
    
        if CV2:
            self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.average_frame = self.frame * 1.0
            self.absdiff_frame = self.frame * 1.0
            self.previous_frame = self.frame * 1.0
            self.area = self.gray_frame.size
        else:
            self.gray_frame = cv.CreateImage(cv.GetSize(self.frame), cv.IPL_DEPTH_8U, 1)
            self.average_frame = cv.CreateImage(cv.GetSize(self.frame), cv.IPL_DEPTH_32F, 3)
            self.absdiff_frame = cv.CloneImage(self.frame)
            self.previous_frame = cv.CloneImage(self.frame)
            self.area = self.frame.width * self.frame.height

        self.currentsurface = 0
        self.currentcontours = None

    def processImage(self, curframe):
        if CV2:
            cv2.GaussianBlur(curframe, (3,3), sigmaX=0, sigmaY=0, dst=curframe) #Remove false positives
            cv2.accumulateWeighted(curframe, self.average_frame, EWMA_ALPHA) #Compute the average
            
            self.previous_frame = self.average_frame.astype('u1') #Convert back to 8U frame
            
            self.absdiff_frame = cv2.absdiff(curframe, self.previous_frame) # moving_average - curframe

            self.gray_frame = cv2.cvtColor(self.absdiff_frame, cv2.COLOR_BGR2GRAY) #Convert to gray
            _,self.gray_frame = cv2.threshold(self.gray_frame, 50, 255, cv2.THRESH_BINARY)

            self.gray_frame = cv2.dilate(self.gray_frame,
                                         np.ones((3,3),np.uint8),
                                         iterations = DILATE_ITERS) #to get object blobs

            self.gray_frame = cv2.erode(self.gray_frame,
                                         np.ones((3,3),np.uint8),
                                         iterations = ERODE_ITERS) #to get object blobs

        else:
            cv.Smooth(curframe, curframe) #Remove false positives
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
        if CV2:
            _, contours, tree = cv2.findContours(self.gray_frame,
                                              mode = cv2.RETR_EXTERNAL,
                                              method = cv2.CHAIN_APPROX_SIMPLE)
            
            moving_area = sum(map(cv2.contourArea, contours))

        else:
            storage = cv.CreateMemStorage(0)
            contours = cv.FindContours(self.gray_frame, storage,
                                       cv.CV_RETR_EXTERNAL,
                                       cv.CV_CHAIN_APPROX_SIMPLE)

            moving_area = 0.0        
            while contours: #For all contours compute the area
                moving_area += cv.ContourArea(contours)
                contours = contours.h_next()

        self.contours = contours
        self.contour_tree = tree #Save contours
        avg = 1.0 * moving_area / self.area #Calculate the average of contour area on the total size

        return avg

    def visualize(self):
        if CV2:
            vis_frame = self.frame.copy()

            cv2.drawContours(vis_frame, self.contours,
                            hierarchy=self.contour_tree,
                            contourIdx=-1,
                            color=CONTOUR_COLOR, thickness=2)

        else:
            vis_frame = cv.CloneImage(self.frame)
            if self.contours:
                cv.DrawContours(vis_frame, self.contours, (0, 0, 255), (0, 255, 0), 1, 2, cv.CV_FILLED)#ME

        return vis_frame


def get_motion_series(capture):
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    motion_series = np.zeros(n_frames)
    
    quant = ContourQuantifier(capture.read()[1])
    motion_series[1:] = [quant.step(capture.read()[1]) for i in range(1,n_frames)]
#    for i in range(1, n_frames):
#        motion_series[i] = quant.step(capture.read()[1])
#        print 100 * i // n_frames
    return motion_series


def get_clips(series, threshold, warmup = 30, cooldown = 30):    
    maximum_filter1d(series, warmup+cooldown+1, origin=(-warmup + cooldown)/2)
    moving_frames = (series > threshold).astype('float')
    on_off = np.diff(moving_frames)
    starts = np.where(on_off == 1)[0]
    stops  = np.where(on_off == -1)[0]

    return zip(starts, stops)
    
if __name__=="__main__":
    fname = sys.argv[-1]
    if CV2:
        if fname=='cam':
            source = cv2.VideoCapture()
        else:
            source = cv2.VideoCapture(sys.argv[-1])
    else:
        if fname=='cam':
            source = cv.CaptureFromCAM(0)
        else:
            source = cv.CaptureFromFile(sys.argv[-1])

    detector = MotionDetectorAdaptative(source, threshold = 0.02,
                                        doRecord=False, showWindows=True)
    detector.run()

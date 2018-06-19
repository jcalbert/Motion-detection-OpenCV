import cv2
from datetime import datetime
import time
import sys

from MotionDetector import MotionDetector, MotionQuantifier
import numpy as np

from scipy.ndimage import maximum_filter1d

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
    
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.average_frame = self.frame * 1.0
        self.absdiff_frame = self.frame * 1.0
        self.previous_frame = self.frame * 1.0
        self.area = self.gray_frame.size

        self.currentsurface = 0
        self.currentcontours = None

    def processImage(self, curframe):

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


        self.motion_level = self.somethingHasMoved()
            
    def somethingHasMoved(self):
        # Find contours
        _, contours, tree = cv2.findContours(self.gray_frame,
                                          mode = cv2.RETR_EXTERNAL,
                                          method = cv2.CHAIN_APPROX_SIMPLE)
        
        moving_area = sum(map(cv2.contourArea, contours))

        self.contours = contours
        self.contour_tree = tree #Save contours
        avg = 1.0 * moving_area / self.area #Calculate the average of contour area on the total size

        return avg

    def visualize(self):
        vis_frame = self.frame.copy()

        cv2.drawContours(vis_frame, self.contours,
                        hierarchy=self.contour_tree,
                        contourIdx=-1,
                        color=CONTOUR_COLOR, thickness=2)

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

    if fname=='cam':
        source = cv2.VideoCapture()
    else:
        source = cv2.VideoCapture(sys.argv[-1])

    detector = MotionDetectorAdaptative(source, threshold = 0.02,
                                        doRecord=False, showWindows=True)
    detector.run()
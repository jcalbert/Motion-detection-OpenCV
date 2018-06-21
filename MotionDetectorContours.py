import cv2
from datetime import datetime
import time
import sys

from MotionDetector import MotionDetector, MotionDetectorOLD, MotionQuantifier
import numpy as np

from scipy.ndimage import maximum_filter1d


CONTOUR_COLOR = (0,0,255)
HOLE_COLOR = (0, 255, 0)

class ContourQuantifier(MotionQuantifier):
 
    def __init__(self, init_frame, alpha=0.05,
                 dilate_amt = .1, erode_amt = 0.05):
        """
        alpha - decay constant for ewma image
        
        """
        MotionQuantifier.__init__(self)

        self._scale = int((init_frame.shape[0] * init_frame.shape[1])**.5)

        self.alpha = alpha
        self.dilate_iters = int(dilate_amt * self._scale)
        self.erode_iters = int(erode_amt * self._scale)
        
        self.slow_avg = init_frame * 1.0
        self.step(init_frame)
        
    def step(self, curframe):
        #Not necessary with downsampled data
        #cv2.GaussianBlur(curframe, (3,3), sigmaX=0, sigmaY=0, dst=curframe) #Remove false positives
        
        
#        CURRENT:
        #incorporate new input to EWMA
        #put ewma into previous_frame
        #put difference between input and prev into absdiff
        #convert diff to gray, put into gray_frame
        #apply threshold to gray image
        #dilate gray
        #erode gray
        
        #find countours
        #sum area of contours
        #normalize area, put into motion_level
        #return motion_level
        
        
#        New:
        #incorporate new input to EWMA
        #put ewma into previous_frame
        #put difference between input and prev into absdiff
        #convert diff to gray, put into gray_frame
        #apply threshold to gray image
        #dilate gray
        #erode gray
        
        #find countours
        #sum area of contours
        #normalize area, put into motion_level
        #return motion_level
        cv2.accumulateWeighted(curframe, self.slow_avg, self.alpha) #Compute the average
        
        abs_diff = cv2.absdiff(curframe, self.slow_avg.astype('u1')) # moving_average - curframe

        abs_diff_gray = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY) #Convert to gray

        cv2.threshold(abs_diff_gray, 50, 255, cv2.THRESH_BINARY,
                      dst = abs_diff_gray)

        cv2.dilate(abs_diff_gray, kernel=None, iterations=self.dilate_iters,
                   dst = abs_diff_gray)
        
        cv2.erode(abs_diff_gray, kernel=None, iterations=self.erode_iters,
                  dst = abs_diff_gray)

        _, contours, tree = cv2.findContours(abs_diff_gray,
                                             mode=cv2.RETR_EXTERNAL,
                                             method=cv2.CHAIN_APPROX_SIMPLE)

        moving_area = sum(map(cv2.contourArea, contours))

        #For visualiztaion
        self._contours = contours #useful only for visualization
        self._contour_tree = tree #Save contours
        self._lastframe = curframe
        
        self.motion_level = 1.0 * moving_area / abs_diff_gray.size

        return self.motion_level
    
    def visualize(self):
        vis_frame = self.slow_avg.astype('u1')

        mask = vis_frame[..., 0].copy()
        cv2.drawContours(mask, self._contours, -1, 255, cv2.FILLED);
        mask = (mask == 255)
        vis_frame[mask] = self._lastframe[mask]
        
        cv2.drawContours(vis_frame, self._contours,
                hierarchy=self._contour_tree,
                contourIdx=-1,
                color=CONTOUR_COLOR, thickness=2)
        
        return vis_frame


def get_motion_series(capture, max_width = 256, frame_skip = 1):
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    motion_series = np.zeros(n_frames)
    
    frame = capture.read()[1]
    if max_width:
        downsample_factor = np.log2(max(frame.shape[0:2]) / max_width)
        downsample_factor = max(0,int(downsample_factor))
    
    def downsample(f):
        for n in range(downsample_factor):
            f = cv2.pyrDown(f)
        return f
    
    quant = ContourQuantifier(downsample(frame))
    #motion_series[1:] = [quant.step(downsample(capture.read()[1])) for i in range(1,n_frames)]
    for i in range(1, n_frames):
        new_frame = capture.read()[1]
        if i%frame_skip == 0:
            motion_series[i] = quant.step(downsample(new_frame))
        else:
            motion_series[i] = motion_series[i-1]

        if i%(n_frames // 100) == 0:
            print i
    return motion_series


def get_clips(series, threshold, warmup = 30, cooldown = 30):    
    maxed_series = maximum_filter1d(series,
                                    warmup+cooldown+1,
                                    origin=(-warmup + cooldown)/2)
    moving_frames = (maxed_series > threshold).astype('float')
    on_off = np.diff(moving_frames)
    starts = np.where(on_off == 1)[0]
    stops  = np.where(on_off == -1)[0]

    return zip(starts, stops)

from subprocess import Popen
import os
import tempfile
def split_vid(fname, clip_bounds):
    tmpdir = tempfile.mkdtemp()
    cmd_base = ['mencoder',
                fname,
                '-oac','pcm',
                '-ovc','copy']
    for i, cb in enumerate(clip_bounds):
        this_file = 'clip_' + str(i).zfill(5) +'.'+ fname.split('.')[-1]
        this_out = os.path.join(tmpdir, this_file)
        this_cmd = ['-ss', str(cb[0]),
                    '-endpos', str(cb[1]),
                    '-o', this_out]
        
        proc = Popen(cmd_base + this_cmd)
        proc.wait()


def make_edit(fname):
    #Get FPS
    #read infile @ 15FPS, low res
    #get motion time series
    #convert to clips
    #use mencoder to extract segments
    #use mencoder to stitch together segments
    pass
if __name__=="__main__":
    fname = sys.argv[-1]

    if fname=='cam':
        source = cv2.VideoCapture()
    else:
        source = cv2.VideoCapture(sys.argv[-1])

    #detector = MotionDetectorAdaptative(source, threshold = 0.02,
    #                                    doRecord=False, showWindows=True)
    #detector.run()
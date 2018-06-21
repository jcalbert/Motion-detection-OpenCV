import cv2
from datetime import datetime
import time
import sys

from MotionDetector import MotionDetector, MotionQuantifier
import numpy as np



from subprocess import Popen
import os
import tempfile
import time


from scipy.ndimage import maximum_filter1d


CONTOUR_COLOR = (0,0,255)
HOLE_COLOR = (0, 255, 0)

class ContourQuantifier(MotionQuantifier):
 
    def __init__(self, init_frame, alpha=0.05,
                 dilate_amt = .03, erode_amt = 0.02):
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
        vis_frame = self.slow_avg.astype('u1').copy()

        mask = vis_frame * 0
        cv2.drawContours(mask, self._contours, -1, (255,255,255), cv2.FILLED);
        vis_frame[mask==255] = self._lastframe[mask==255]
        
        cv2.drawContours(vis_frame, self._contours,
                hierarchy=self._contour_tree,
                contourIdx=-1,
                color=CONTOUR_COLOR, thickness=2)
        
        return vis_frame


def get_motion_series(capture, n_max = None):
    
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    motion_series = np.zeros(n_frames)
    
    if n_max:
        n_frames = min(n_max, n_frames)
        
    detector = MotionDetector(capture, 0.025,
                              ContourQuantifier, quant_args={'alpha':0.2})
    cv2.namedWindow("Preview")
    dt = [0,time.time()]
    for i in range(1,n_frames):
        detector.step()
        motion_series[i] = detector.quant.motion_level
        
        dt[1] = time.time()
        if dt[1]-dt[0] > 2.0:
            dt[0] = dt[1]
        #if i%(n_frames // 100) == 0:
            print "Frame {}/{}".format(i, n_frames)
            if True:
                cv2.imshow("Preview",detector.visualize())
                _=cv2.waitKey(1) % 0x100
            
    return motion_series

def get_clips(series, threshold, warmup = 30, cooldown = 30):    
    maxed_series = maximum_filter1d(series,
                                    warmup+cooldown+1,
                                    origin=(-warmup + cooldown)/2)
    moving_frames = (maxed_series > threshold).astype('float')
    moving_frames[0] = 0
    moving_frames[-1] = 0
    on_off = np.diff(moving_frames)
    starts = np.where(on_off == 1)[0]
    stops  = np.where(on_off == -1)[0]

    return zip(starts, stops)

def split_vid(fpath, clip_bounds):
    cap = cv2.VideoCapture(fpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fdir, fname = os.path.split(fpath)
    
    dst = os.path.join(fdir, fname.split('.')[0] + '_CLIPS')

    os.mkdir(dst)
    
    cmd_base = ['mencoder',
                fpath,
                '-oac','pcm',
                '-ovc','copy']

    if '.mts' in fpath.lower():
        cmd_base.append('-demuxer')
        cmd_base.append('lavf')
        #cmd_base.append('-of')
        #cmd_base.append('lavf=mp4')
        
        
    for i, cb in enumerate(clip_bounds):

        this_file = 'clip_' + str(i).zfill(5) +'.'+ fname.split('.')[-1]
        this_out = os.path.join(dst, this_file)
        
        start_time = round(cb[0] / fps, 2)
        end_time = round( (cb[1]-cb[0]) / fps, 2)
        
        this_cmd = ['-ss', str(start_time),
                    '-endpos', str(end_time),
                    '-o', this_out]
        print "Writing: {}".format(this_out)
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


from imutils.video import FileVideoStream
from queue import Queue
class FVS_CAP(FileVideoStream):
    def __init__(self, name, queueSize=128, pre_filter=None):
        FileVideoStream.__init__(self, name, queueSize=queueSize)

        self._cap = cv2.VideoCapture(name)
        time.sleep(1.0)
        
    def get(self, prop):
        return self._cap.get(prop)
    
    def read(self):
        return (self.more(), FileVideoStream.read(self))

if __name__=="__main__":
    fname = sys.argv[-1]

    if fname=='cam':
        source = cv2.VideoCapture()
    else:
        #source = cv2.VideoCapture(fname)
        source = FVS_CAP(fname, 16).start()
        y = get_motion_series(source)

    #detector = MotionDetectorAdaptative(source, threshold = 0.02,
    #                                    doRecord=False, showWindows=True)
    #detector.run()
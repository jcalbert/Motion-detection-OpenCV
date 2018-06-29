import cv2
import time

import logging
import sys

import numpy  as np



from subprocess import Popen
import os
import tempfile

from scipy.ndimage import maximum_filter1d

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()

COOLDOWN_FRAMES = 20 # Num of non-moving frames before recording stops


#For now, hard code the size of image used for detection
DOWNSAMPLE = True
SCALEDOWN = True
MIN_WIDTH = 256

if DOWNSAMPLE:
    if SCALEDOWN:
        def sampdown(im):
            big_edge = max(im.shape[0], im.shape[1])
            shrink_factor = 2**int(np.log2(big_edge / MIN_WIDTH))
                fac = 1.0 / shrink_factor
                im = cv2.resize(im, dsize=None, fx=fac, fy=fac, 
                                interpolation=cv2.INTER_AREA)
            return im

    else:
        def sampdown(im):
            big_edge = max(im.shape[0], im.shape[1])
            shrink_factor = 2**int(np.log2(big_edge / MIN_WIDTH))
            for _ in range(int(np.log2(shrink_factor))):
                im = cv2.pyrDown(im)

            return im

else:
    def sampdown(im):
        return im


class MotionQuantifier():    
    """
    Class takes in a sequence of frames and tracks a per-frame level of movement.

    Subclasses should maintain internal state variables necessary to track
    movement.  They may optionally implement a visualize() method which
    returns a frame of the same type as the input data which represents the
    quantifier's state (e.g. highlighting fast-changing pixels).

    ...
    
    Attributes
    ----------
    motoin_level : float
        Number between 0 and 1 representing the current level of movement.

    Methods
    -------
    step(frame)
        Update the quantifier by `frame`.
    visualize()
        Returns a visualization of the quantifier's state.

    """
    def __init__(self):
        self.motion_level = 0.0 #All detectors quantify level of motion from 0 to 1

    def step(self, frame):
        raise NotImplementedError("""Subclasses should process the frame,
                                  update internal state and return a motion 
                                  level""")
        return self.motion_level

    def visualize(self):
        raise NotImplementedError("""If implemented, this should return a
                                  visual representation of the qunatifier's
                                  state.""")


class MotionDetector():
    """
    This base class combines a video capture source and a quantifier.  It
    exposes a method step() which consumes one frame and updates the
    quantifier's internal state and level of motion.  It also incluse 
    a run() funciton which repeats step() until all frames are consumed.

    For now it contains an internal mechanism for detecting when motion
    has started or stopped.  Eventually this will be abstracted out.

    ...
    

    Methods
    -------
    step(frame)
        Update the quantifier by `frame`.
    visualize()
        Returns a visualization of the quantifier's state.

    """
    def __init__(self, source, threshold, quantifier, quant_args={}):

        self.capture = source
        self.running = True

        #Internal state common to all detectors
        #Used for motion detection itself
        self.frame_no = 1 #Frame counter

        _,init_frame = self.capture.read() #Take a frame to init recorder 
        init_frame = sampdown(init_frame)
        
        self.quant = quantifier(init_frame, **quant_args)

        self.cooldown = 0

        self.trigger_time = 0 #Hold timestamp of the last detection
        
        self.threshold = threshold


    def onInterrupt(self, sig, frame):
        self.running = False


    #Not sure I even want to use this
    def initRecorder(self): #Create the recorder
        raise NotImplementedError("This relies on the old interface.")
        codec = cv.CV_FOURCC('M', 'J', 'P', 'G') #('W', 'M', 'V', '2')
        writer = cv.CreateVideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S")+".wmv",
                                         codec, 5, cv.GetSize(self.frame), 1)
        #FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font

        return writer

    def step(self):
        log.debug("Frame: {}".format(self.frame_no))
        log.debug("Movement: {}".format(self.quant.motion_level))

        read_ok, this_frame = self.capture.read()
        self.frame_no += 1

        if not read_ok:
            log.debug("Out of frames, breaking.")
            self.running = False
            return
            

        this_frame = sampdown(this_frame)        

        self.quant.step(this_frame)
        
    def run(self):
        #raise NotImplementedError("Subclasses must implement this method")
        while self.running:
            self.step()
            
            if self.quant.motion_level > self.threshold:
                if self.cooldown == 0:
                    log.info("Start: {}".format(self.frame_no))#start recording
                self.cooldown = COOLDOWN_FRAMES

            if self.cooldown > 0:
                pass#write a frame
                if self.cooldown == 1:
                    log.info("Stop: {}".format(self.frame_no))
                self.cooldown -= 1                                
 
    #Embelling the qunatifier's visualization if necessary
    def visualize(self):
        vis_frame = self.quant.visualize()
        if self.cooldown > 0:
            width = vis_frame.shape[0]
            radius = int(width * .04)
            center = (int(radius*1.5), ) * 2
            cv2.circle(vis_frame, center, radius, (0,0,255), thickness=-1)
        return vis_frame
    

CONTOUR_COLOR = (0,0,255)
HOLE_COLOR = (0, 255, 0)

class ContourQuantifier(MotionQuantifier):
    """
    Quantifier compares current frame to an Exponentially Weighted Moving
    Average (EWMA).  The absolute difference is thresholded, dilated,
    eroded.  Contour lines are drawn for this image, and the motion level
    is the fractional area covered by top-level contours.

    Credit to github users RobinDavid and mattwilliamson for the aglorithm.
    """
    def __init__(self, init_frame, alpha=0.05,
                 dilate_amt = .03, erode_amt = 0.02):
        """
        alpha - decay constant for ewma image, between 0 and 1.
        dilate_amt - relative size of motion area dilation, between 0 and 1. 
        erode_amt - relative size of motion area dilation, between 0 and 1.
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
        """
        Draw contours around moving area.  Non-moving area is EWMA frame.
        """
        vis_frame = self.slow_avg.astype('u1').copy()

        mask = vis_frame * 0
        cv2.drawContours(mask, self._contours, -1, (255,255,255), cv2.FILLED);
        vis_frame[mask==255] = self._lastframe[mask==255]
        
        cv2.drawContours(vis_frame, self._contours,
                hierarchy=self._contour_tree,
                contourIdx=-1,
                color=CONTOUR_COLOR, thickness=2)
        
        return vis_frame


class BGSubQuantifier(MotionQuantifier):
    """
    Quantifier based on openCV's BackgroundSubtractor class.  These 
    already identify foreground and background objects through
    various methods.  The fractoinal area identified as 'foreground'
    is used for the motion level.
    """

    def __init__(self, init_frame, bgsub=None, 
                 dilate_amt = .01, erode_amt = 0.01):
        """
        bgsub - An initialized cv2.BackgroundSubtractor.
        dilate_amt - relative size of motion area dilation, between 0 and 1. 
        erode_amt - relative size of motion area dilation, between 0 and 1.
        """
        assert isinstance(bgsub, cv2.BackgroundSubtractor)
        
        MotionQuantifier.__init__(self)

        self.bgsub = bgsub


        self._scale = int((init_frame.shape[0] * init_frame.shape[1])**.5)

        self.dilate_iters = int(dilate_amt * self._scale)
        self.erode_iters = int(erode_amt * self._scale)
        
        self._fgmask = self.bgsub.apply(init_frame)
        
        self.step(init_frame)


                
    def step(self, curframe):
        self._lastframe = curframe
        self.bgsub.apply(curframe, self._fgmask)

        cv2.dilate(self._fgmask, kernel=None, iterations=self.dilate_iters,
                   dst = self._fgmask)
        
        cv2.erode(self._fgmask, kernel=None, iterations=self.erode_iters,
                  dst = self._fgmask)
        
        self.motion_level = (self._fgmask == 255).mean()

        return self.motion_level
    
    def visualize(self):
        """
        Draw foreground normally and background at halved RGB values.
        """
        fg = self._fgmask == 255
        vis_frame = self._lastframe.copy()
        vis_frame[~fg] /= 2
        #vis_frame = self.bgsub.getBackgroundImage().copy()
        
 #       vis_frame[self._fgmask] = self._lastframe[self._fgmask]
 #       vis_frame[~self._fgmask] /= 2

        return vis_frame




import cv2

from datetime import datetime
import time

import signal

import logging
import sys

import numpy  as np
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()
COOLDOWN_FRAMES = 20 # Num of non-moving frames before recording stops

DOWNSAMPLE = True
MIN_WIDTH = 256
SCALEINSTEAD = True

def sampdown(im):
    big_edge = max(im.shape[0], im.shape[1])
    shrink_factor = 2**int(np.log2(big_edge / MIN_WIDTH))
    
    
    if DOWNSAMPLE:
        if SCALEINSTEAD:
            fac = 1.0 / shrink_factor
            im = cv2.resize(im, dsize=None, fx=fac, fy=fac, 
                            interpolation=cv2.INTER_AREA)

        else:
            for _ in range(int(np.log2(shrink_factor))):
                im = cv2.pyrDown(im)
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

    def __init__(self, source, threshold, quantifier, quant_args={},
                 doRecord=False, showWindows=True):

        self.capture = source
        self.running = True

        #Internal state common to all detectors
        #Used for motion detection itself
        self.frame_no = 1 #Frame counter

        _,init_frame = self.capture.read() #Take a frame to init recorder 
        init_frame = sampdown(init_frame)
        

        self.quant = quantifier(init_frame, **quant_args)

        self.cooldown = 0

        self.doRecord = doRecord #Either or not record the moving object
        self.show = showWindows #Either or not show the 2 windows
        

        if doRecord:
            self.writer = self.initRecorder()
        
        self.trigger_time = 0 #Hold timestamp of the last detection
        
        self.threshold = threshold
        if showWindows:
            cv2.namedWindow("Image")
            cv2.createTrackbar("Detection treshold: ",
                              "Image",
                              int(self.threshold * 100),
                              100,
                              lambda v: self.onChange({'threshold': v / 100.}))
    
    
    #This is only really used for interactivity.
    def onChange(self, keyvals):
        for k,v in keyvals.iteritems():
            setattr(self, k, v)

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
            
            if self.show:
                vis_frame = self.visualize()
                cv2.imshow("Image", vis_frame)

                c=cv2.waitKey(1) % 0x100
                if c==27 or c == 10: #Break if user enters 'Esc'.
                    self.running = False
 
    #Embelling the qunatifier's visualization if necessary
    def visualize(self):
        vis_frame = self.quant.visualize()
        if self.cooldown > 0:
            width = vis_frame.shape[0]
            radius = int(width * .04)
            center = (int(radius*1.5), ) * 2
            cv2.circle(vis_frame, center, radius, (0,0,255), thickness=-1)
        return vis_frame
        
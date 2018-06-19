import cv2

from datetime import datetime
import time

import signal

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()
COOLDOWN_FRAMES = 20 # Num of non-moving frames before recording stops

DOWNSAMPLE = True
MIN_WIDTH = 256

class MotionQuantifier():
    
    def __init__(self, init_frame):
        self.frame = init_frame
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

    def __init__(self, source, threshold, doRecord=True, showWindows=True):

        self.running = True
        self.capture = source

        #Internal state common to all detectors
        #Used for motion detection itself
        self.frame_no = 1   #Frame counter, used instead of time()

        _,self.frame = self.capture.read() #Take a frame to init recorder 
        if DOWNSAMPLE:
            while max(self.frame.shape[0:2]) > MIN_WIDTH * 2:
                self.frame = cv2.pyrDown(self.frame)
        

        self.motion_level = 0.0 #All detectors quantify level of motion from 0 to 1
        self.cooldown = 0

        self.doRecord=doRecord #Either or not record the moving object
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

    def run(self):
        #raise NotImplementedError("Subclasses must implement this method")
        while self.running:
            log.debug("Frame: {}".format(self.frame_no))
            log.debug("Movement: {}".format(self.motion_level))            

            self.running, this_frame = self.capture.read()
            if not self.running:
                break

            if DOWNSAMPLE:
                while max(this_frame.shape[0:2]) > MIN_WIDTH * 2:
                    this_frame = cv2.pyrDown(this_frame)
            
            self.frame = this_frame

            self.processImage(this_frame) #Process the image and update internal state
            if self.motion_level > self.threshold:
                if self.cooldown == 0:
                    log.info("Start: {}".format(self.frame_no))#start recording
                self.cooldown = COOLDOWN_FRAMES

            if self.cooldown > 0:
                pass#write a frame
                if self.cooldown == 1:
                    log.info("Stop: {}".format(self.frame_no))
                self.cooldown -= 1                                
            
            if self.show or self.cooldown > 0:
                vis_frame = self.visualize()
                cv2.imshow("Image", vis_frame)

                c=cv2.waitKey(1) % 0x100
                if c==27 or c == 10: #Break if user enters 'Esc'.
                    self.running = False
            self.frame_no += 1
    #Return an image representing the internal state of the processor
    def visualize(self):
        return self.frame
        
    def processImage(self, this_frame):
        raise NotImplementedError("Subclasses must implement this method")



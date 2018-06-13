import cv2.cv as cv
from datetime import datetime
import time

import signal

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()
COOLDOWN_FRAMES = 20 # Num of non-moving frames before recording stops

class MotionDetector():

    def __init__(self, source, threshold, doRecord=True, showWindows=True):

        self.running = True
        self.capture = source

        #Internal state common to all detectors
        #Used for motion detection itself
        self.frame_no = 0   #Frame counter, used instead of time()
        self.frame = cv.QueryFrame(self.capture) #Take a frame to init recorder
        self.motion_level = 0.0 #All detectors quantify level of motion from 0 to 1
        self.record = True #Toggle whether activity is detected
        self.cooldown = 0

        self.font = None
        self.doRecord=doRecord #Either or not record the moving object
        self.show = showWindows #Either or not show the 2 windows
        

        if doRecord:
            self.writer = self.initRecorder()
        
        self.trigger_time = 0 #Hold timestamp of the last detection
        
        self.threshold = threshold
        if showWindows:
            cv.NamedWindow("Image")
            cv.CreateTrackbar("Detection treshold: ",
                              "Image",
                              self.threshold,
                              100,
                              lambda v: self.onChange({'threshold': v / 100.}))
    
    #This is only really used for interactivity.
    def onChange(self, keyvals):
        for k,v in keyvals.iteritems():
            setattr(self, k, v)

    def onInterrupt(self, signal, frame):
        self.running = False


    #Not sure I even want to use this
    def initRecorder(self): #Create the recorder
        codec = cv.CV_FOURCC('M', 'J', 'P', 'G') #('W', 'M', 'V', '2')
        writer = cv.CreateVideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S")+".wmv",
                                         codec, 5, cv.GetSize(self.frame), 1)
        #FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font

        return writer


    def run(self):
        #raise NotImplementedError("Subclasses must implement this method")
        while self.running:
            this_frame = cv.QueryFrame(self.capture)
            
            self.processImage(this_frame) #Process the image and update internal state
            if self.motion_level > self.threshold:
                if self.cooldown == 0:
                    log.info("Start Recording")#start recording
                self.cooldown = COOLDOWN_FRAMES

            if self.cooldown > 0:
                pass#write a frame
                if self.cooldown == 1:
                    log.info("Stop Recording")
                self.cooldown -= 1                                
            
            if self.show:
                vis_frame = self.visualize()
                cv.ShowImage("Image", vis_frame)
#                cv.ShowImage("Res", self.res)
                c=cv.WaitKey(1) % 0x100
                if c==27 or c == 10: #Break if user enters 'Esc'.
                    self.running = False
    
    #Return an image representing the internal state of the processor
    def visualize(self):
        return self.frame
        
    def processImage(self, this_frame):
        raise NotImplementedError("Subclasses must implement this method")



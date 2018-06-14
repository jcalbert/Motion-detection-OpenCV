import cv2.cv as cv
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

CV2 = True
class MotionDetector():

    def __init__(self, source, threshold, doRecord=True, showWindows=True):

        self.running = True
        self.capture = source

        #Internal state common to all detectors
        #Used for motion detection itself
        self.frame_no = 1   #Frame counter, used instead of time()

        if CV2:
           _,self.frame = self.capture.read() #Take a frame to init recorder 
        else:
           self.frame = cv.QueryFrame(self.capture) #Take a frame to init recorder
        

        self.motion_level = 0.0 #All detectors quantify level of motion from 0 to 1
        self.cooldown = 0

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
            log.debug("Frame: {}".format(self.frame_no))
            if CV2:
                self.running, this_frame = self.capture.read()
            else:
                this_frame = cv.QueryFrame(self.capture)

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
            
            if self.show:
                vis_frame = self.visualize()
                if CV2:
                    cv2.imshow("Image", vis_frame)
                else:
                    cv.ShowImage("Image", vis_frame)

                c=cv.WaitKey(1) % 0x100
                if c==27 or c == 10: #Break if user enters 'Esc'.
                    self.running = False
            self.frame_no += 1
    #Return an image representing the internal state of the processor
    def visualize(self):
        return self.frame
        
    def processImage(self, this_frame):
        raise NotImplementedError("Subclasses must implement this method")



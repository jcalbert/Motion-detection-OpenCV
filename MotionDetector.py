import cv2.cv as cv
from datetime import datetime
import time

class MotionDetector():

    def __init__(self, source, threshold, doRecord=True, showWindows=True):
        self.font = None
        self.doRecord=doRecord #Either or not record the moving object
        self.show = showWindows #Either or not show the 2 windows
        self.frame = None
        
        self.capture = source
        self.frame = cv.QueryFrame(self.capture) #Take a frame to init recorder

        if doRecord:
            self.writer = None
            self.initRecorder()
        
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0 #Hold timestamp of the last detection
        
        if showWindows:
            cv.NamedWindow("Image")
            cv.CreateTrackbar("Detection treshold: ",
                              "Image",
                              self.threshold,
                              100,
                              lambda v: self.onChange({'threshold':v}))
    
    #This is only really used for interactivity.
    def onChange(self, keyvals):
        for k,v in keyvals.iteritems():
            setattr(self, k, v)

    #Not sure I even want to use this
    def initRecorder(self): #Create the recorder
        codec = cv.CV_FOURCC('M', 'J', 'P', 'G') #('W', 'M', 'V', '2')
        self.writer=cv.CreateVideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S")+".wmv",
                                         codec, 5, cv.GetSize(self.frame), 1)
        #FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font

    def run(self):
        raise NotImplementedError("Subclasses must implement this method")

    def processImage(self, this_frame):
        raise NotImplementedError("Subclasses must implement this method")



import cv2.cv as cvs
from datetime import datetime
import time

from MotionDetector import MotionDetector

class MotionDetectorInstantaneous(MotionDetector):
    
    def __init__(self, source, threshold=8, doRecord=True, showWindows=True):
        MotionDetector.__init__(self, source=source, threshold=threshold,
                                doRecord=doRecord, showWindows=showWindows)

        self.frame1gray = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t-1
        cv.CvtColor(self.frame, self.frame1gray, cv.CV_RGB2GRAY)
        
        #Will hold the thresholded result
        self.res = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U)
        
        self.frame2gray = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t
        
        self.width = self.frame.width
        self.height = self.frame.height
        self.nb_pixels = self.width * self.height
        
    def run(self):
        started = time.time()
        while True:
            
            curframe = cv.QueryFrame(self.capture)
            instant = time.time() #Get timestamp o the frame
            
            self.processImage(curframe) #Process the image
            
            if not self.isRecording:
                if self.somethingHasMoved():
                    self.trigger_time = instant #Update the trigger_time
                    if instant > started +5:#Wait 5 second after the webcam start for luminosity adjusting etc..
                        print datetime.now().strftime("%b %d, %H:%M:%S"), "Something is moving !"
                        if self.doRecord: #set isRecording=True only if we record a video
                            self.isRecording = True
            else:
                if instant >= self.trigger_time +10: #Record during 10 seconds
                    print datetime.now().strftime("%b %d, %H:%M:%S"), "Stop recording"
                    self.isRecording = False
                else:
                    cv.PutText(curframe,datetime.now().strftime("%b %d, %H:%M:%S"), (25,30),self.font, 0) #Put date on the frame
                    cv.WriteFrame(self.writer, curframe) #Write the frame
            
            if self.show:
                cv.ShowImage("Image", curframe)
                cv.ShowImage("Res", self.res)
                
            cv.Copy(self.frame2gray, self.frame1gray)
            c=cv.WaitKey(1) % 0x100
            if c==27 or c == 10: #Break if user enters 'Esc'.
                break            
    
    def processImage(self, frame):
        cv.CvtColor(frame, self.frame2gray, cv.CV_RGB2GRAY)
        
        #Absdiff to get the difference between to the frames
        cv.AbsDiff(self.frame1gray, self.frame2gray, self.res)
        
        #Remove the noise and do the threshold
        cv.Smooth(self.res, self.res, cv.CV_BLUR, 5,5)
        cv.MorphologyEx(self.res, self.res, None, None, cv.CV_MOP_OPEN)
        cv.MorphologyEx(self.res, self.res, None, None, cv.CV_MOP_CLOSE)
        cv.Threshold(self.res, self.res, 10, 255, cv.CV_THRESH_BINARY_INV)

    def somethingHasMoved(self):
        nb=0 #Will hold the number of black pixels
        min_threshold = (self.nb_pixels/100) * self.threshold #Number of pixels for current threshold
        nb = self.nb_pixels - cv.CountNonZero(self.res)
        if (nb) > min_threshold:
           return True
        else:
           return False
        
if __name__=="__main__":
    fname = sys.argv[-1]
    if fname=='cam':
        source = cv.CaptureFromCAM(0)
    else:
        source = cv.CaptureFromFile(sys.argv[-1])

    detector = MotionDetectorInstantaneous(source, doRecord=False)

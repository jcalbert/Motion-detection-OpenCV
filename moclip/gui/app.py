import sys

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication

import os
import cv2

this_dir = os.path.split(__file__)[0]

qtCreatorFile = "Main_Window.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(os.path.join(this_dir, qtCreatorFile))

from .. import detectors


DEFAULTS = {'threshold' : 0.02}

class ClipperMainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        #Ui_MainWindow.__init__(self)

        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)

        #self.input_fname = '' Just read the input box instead

        self.ui.btn_infile.clicked.connect(self.input_dialogue)

        self.ui.btn_start.clicked.connect(self.run)
        self.ui.btn_pause.clicked.connect(self.interrupt)

        self.running = False

        self.cap = None

    def set_to_running(self):
        #Set all buttons except pause to disabled
        pass

    def set_to_stopped(self):
        #restore all buttons to regular state.
        pass

    def input_dialogue(self):

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                        QtWidgets.QFileDialog(),
                        "Select File Containing Projections",
                        filter="Video Files (*.mpg *.mp4 *.mts);; All Files (*)")

        if filename:
            #self.input_fname = filename Just read the input box instead
            self.ui.le_infile.setText(filename)

    def load_file(self):
        fname = self.ui.le_infile.text()
        if not os.path.exists(fname):
            err_msg = QtWidgets.QErrorMessage()
            err_msg.showMessage('File {} does not exist.'.format(fname))
            return False

        self.cap = cv2.VideoCapture(fname)

        #Set progress bar
        n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setMaximum(n_frames)

        #Set video preview
        _, frame = self.cap.read()
        frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qim = QtGui.QImage(frame_cvt,
                       frame.shape[1],
                       frame.shape[0],
                       QtGui.QImage.Format_RGB888)
        self.ui.lab_video.setPixmap(QtGui.QPixmap.fromImage(qim))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return True

    def initialize(self):
        #check if the filename is valid
        if not self.load_file():
            return False

        # Choosing detector method
        if self.ui.rb_contours.isChecked():
            quantifier = detectors.ContourQuantifier
            quant_args = {'alpha':0.2}

        elif self.ui.rb_bgsub.isChecked():
            bgsub = detectors.cv2.createBackgroundSubtractorMOG2(history=500,
                                                                 detectShadows=True)
            quantifier = detectors.BGSubQuantifier
            quant_args = {'bgsub':bgsub}

        elif self.ui.rb_ewma.isChecked():
            raise NotImplementedError("I got rid of this")

        else:
            raise Exception('No radio button checked, somehow')


        self.detector = detectors.MotionDetector(self.cap, DEFAULTS['threshold'],
                                                 quantifier = quantifier,
                                                 quant_args = quant_args)

        #lock out stuff

        self.running = True
        return
        raise NotImplementedError('Uh yeah this needs to happen.')


    def step(self):
        self.detector.step()
        self.ui.progressBar.setValue(self.detector.frame_no)
        frames = [self.detector.quant._lastframe, self.detector.visualize()]
        labs = [self.ui.lab_video, self.ui.lab_visualize]
        for frame, lab in zip(frames, labs):
            frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qim = QtGui.QImage(frame_cvt,
                           frame.shape[1],
                           frame.shape[0],
                           QtGui.QImage.Format_RGB888)
#            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)


            lab.setPixmap(QtGui.QPixmap.fromImage(qim))

    def run(self):
        self.initialize()
        while self.running:
            self.step()
            if self.detector.frame_no >= self.ui.progressBar.maximum():
                self.running = False
            QApplication.processEvents()
        #If complete, move to next tab


        raise NotImplementedError('Uh yeah this needs to happen.')

    def interrupt(self):
        self.running = False
        raise NotImplementedError('Uh yeah this needs to happen.')


    def log(self, msg):
        do_some_logging_to(self.te_log)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClipperMainWindow()
    window.show()
    #sys.exit(app.exec_())

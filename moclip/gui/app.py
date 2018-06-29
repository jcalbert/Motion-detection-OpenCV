import sys


from PyQt5 import QtCore, QtGui, uic, QtWidgets

from PyQt5.QtWidgets import QMainWindow, QApplication
import sys, random




qtCreatorFile = "gui/Main_Window.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

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



    def input_dialogue(self):
        
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                        QtWidgets.QFileDialog(),
                        "Select File Containing Projections",
                        filter="Video Files (*.mpg *.mp4 *.mts);; All Files (*)")

        if filename:
            #self.input_fname = filename Just read the input box instead

            self.ui.le_infile.setText(filename)

    def initialize():
        raise NotImplementedError('Uh yeah this needs to happen.')
        #check if the filename is valid
        #open the file with opencv
        #read number of frames
        #set max number for progress bar
        #read detector type from radio buttons
        #create a motion detector object  
        #lock out radio buttons



    def run(self):
        raise NotImplementedError('Uh yeah this needs to happen.')
        self.initialize()       
        while self.running:
            do_one_step()
            #update progress bar
            #if it's time, update image previews

        #If complete, move to next tab

    def interrupt(self):
        raise NotImplementedError('Uh yeah this needs to happen.')


    def log(self, msg):
        do_some_logging_to(self.te_log)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClipperMainWindow()
    window.show()
    #sys.exit(app.exec_())

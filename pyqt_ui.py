import sys
import os
import numpy as np
import cv2
import shutil
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QStackedWidget, QHBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from detection.model import getModel, getPrediction
from detection.utils import drawBoxes, getCTime


# Stacked Widget Base Class
class BaseWidget(QWidget):
    def __init__(self, modelPretrainedWeight=None):
        super().__init__()
        self.displayLabel = QLabel(self)
        self.displayLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.displayLabel.setGeometry(60, 40, 640, 480)
        self.displayLabel.setFrameShape(QLabel.Shape.Box)
        if modelPretrainedWeight is not None:
            self.setModelWeight(modelPretrainedWeight)
            self.warmup()
    
    def setModelWeight(self, modelPretrainedWeight):
        self.model = getModel(weightsPath=modelPretrainedWeight)
        self.warmup()
    
    def warmup(self):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
    
    def predictImg(self, img):
        prediction = getPrediction(self.model, img)
        predictedImg = drawBoxes(img, bndboxes=prediction, withConfScore=True, isRgb=True)
        return predictedImg
    
    @pyqtSlot(np.ndarray)
    def updateImg(self, img):
        h, w, ch = img.shape
        bytesPerLine = ch * w
        qImg = QImage(img.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
        
        # Center the image in the label
        x = self.displayLabel.x() + (self.displayLabel.width() - w) // 2
        y = self.displayLabel.y() + (self.displayLabel.height() - h) // 2
        self.displayLabel.setGeometry(x, y, w, h)
        self.displayLabel.setPixmap(QPixmap.fromImage(qImg))
    
    def stop(self):
        self.displayLabel.setGeometry(60, 40, 640, 480)


# Image Widget
class ImageWidget(BaseWidget):
    def __init__(self, modelPretrainedWeight):
        super().__init__(modelPretrainedWeight)
        self.browseButton = QPushButton("Browse", self)
        self.browseButton.setGeometry(340, 560, 100, 28)
        self.browseButton.clicked.connect(self.browse)
        self.displayLabel.setText("Browse an image to start prediction.")

    def browse(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        img = cv2.imread(fileName)
        self.showPrediction(img)
    
    def showPrediction(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prediction = self.predictImg(img)
        self.updateImg(prediction)
    
    def stop(self):
        super().stop()
        self.displayLabel.setText("Browse an image to start prediction.")


# Camera Widget
class CameraWidget(BaseWidget):
    def __init__(self, modelPretrainedWeight):
        super().__init__(modelPretrainedWeight)
        self.displayLabel.setText("Camera frame")
        self.startBtn = QPushButton("Start", self)
        self.startBtn.setGeometry(160, 560, 90, 40)
        self.stopBtn = QPushButton("Stop", self)
        self.stopBtn.setGeometry(520, 560, 90, 40)
        
        self.camera = CameraThread()
        self.camera.frameSignal.connect(self.updateFrame)
        self.startBtn.clicked.connect(self.startCamera)
        self.stopBtn.clicked.connect(self.stopCamera)
    
    @pyqtSlot(np.ndarray)
    def updateFrame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.predictImg(frame)
        self.updateImg(frame)
    
    def startCamera(self):
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.camera.start()
        
    def stopCamera(self):
        self.camera.stopCamera()
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self.displayLabel.setText("Camera stopped")
    
    def stop(self):
        self.camera.stopThread()
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self.displayLabel.setText("Camera frame")


class CameraThread(QThread):
    frameSignal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.cap = None
        
    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.frameSignal.emit(frame)
            self.msleep(30)
            
    def stopCamera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def stopThread(self):
        self.stopCamera()
        self.terminate()
        self.wait()


# Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Mask Detection")
        self.setFixedSize(760, 760)
        self.initWeightInputLayout()
        self.initInputSelectionLayout()
        self.initWidgetStack()
    
    def checkWeightFile(self):
        if not os.path.exists(self.modelPretrainedWeight):
            QMessageBox.warning(self, "Warning", f"Please upload an weights to continue!")
            self.uploadWeight()
        else:
            self.setWeightFile()
            
    def setWeightFile(self):
        for index in range(self.stack.count()):
                self.stack.widget(index).setModelWeight(self.modelPretrainedWeight)
        self.weightLabel.setText(f"Pretrained Weights: {self.modelPretrainedWeight}\
                                    \nTime: {getCTime(self.modelPretrainedWeight)}")
        
    def uploadWeight(self):
        while True:
            fileName, _ = QFileDialog.getOpenFileName(self, "Select model weights", "", "(*.pt)")
            if fileName:
                shutil.copy(fileName, self.modelPretrainedWeight)
                self.setWeightFile()
                QMessageBox.information(self, "Success", "Weights uploaded successfully!")
                break
            
    def initWeightInputLayout(self):
        os.makedirs('uploaded-weights', exist_ok=True)
        self.modelPretrainedWeight = 'uploaded-weights/weights.pt'
        self.uploadBtn = QPushButton("Upload", self)
        self.uploadBtn.setGeometry(100, 700, 80, 30)
        self.uploadBtn.clicked.connect(self.uploadWeight)
        
        self.weightLabel = QLabel("Pretrained Weights: Please upload an weights to continue!", self)
        self.weightLabel.setGeometry(220, 700, 500, 30)
    
    def initInputSelectionLayout(self):
        self.inputSelectionWidget = QWidget(self)
        self.inputSelectionLayout = QHBoxLayout(self.inputSelectionWidget)
        
        self.webcamButton = QPushButton("Webcam")
        self.imageButton = QPushButton("Image")
        
        self.inputSelectionLayout.addWidget(self.imageButton)
        self.inputSelectionLayout.addWidget(self.webcamButton)
        
        self.inputSelectionWidget.setLayout(self.inputSelectionLayout)
        self.inputSelectionWidget.setGeometry(60, 0, 640, 50)
        
    def initWidgetStack(self, modelPretrainedWeight=None):
        self.stack = QStackedWidget(self)
        self.stack.setGeometry(0, 50, 900, 650)
        
        self.webcamWidget = CameraWidget(modelPretrainedWeight)
        self.imageWidget = ImageWidget(modelPretrainedWeight)
        
        self.stack.addWidget(self.imageWidget)
        self.stack.addWidget(self.webcamWidget)
        
        self.connectButton()
        self.imageButton.isCheckable = True
        self.switchWidget(0)
        
    def connectButton(self):
        self.imageButton.clicked.connect(lambda: self.switchWidget(0))
        self.webcamButton.clicked.connect(lambda: self.switchWidget(1))
    
    def switchWidget(self, index):
        self.stack.currentWidget().stop()
        self.stack.setCurrentIndex(index)
    
    def closeEvent(self, event):
        event.accept()
        
    def centerWindow(self):    
        screen = self.screen().availableGeometry()
        widget = self.geometry()
        x = (screen.width() - widget.width()) // 2
        y = (screen.height() - widget.height()) // 2
        self.move(x, y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.centerWindow()
    window.show()
    window.checkWeightFile()
    sys.exit(app.exec())
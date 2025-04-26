import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound

class Panorama(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("Panorama")
    self.setGeometry(300, 300, 800, 400)

    collectButton = QPushButton("영상 수집", self)
    self.showButton = QPushButton("영상 보기", self)
    self.stitchButton = QPushButton("영상 합성", self)
    self.saveButton = QPushButton("영상 저장", self)
    quitButton = QPushButton("종료", self)
    self.label = QLabel("포스가 함꼐하길...", self)

    collectButton.setGeometry(10, 40, 85, 50)
    self.showButton.setGeometry(180, 40, 85, 50)
    self.stitchButton.setGeometry(300, 40, 85, 50)
    self.saveButton.setGeometry(380, 40, 85, 50)
    quitButton.setGeometry(480, 40, 85, 50)
    self.label.setGeometry(10, 80, 700, 200)

    self.showButton.setEnabled(False)
    self.stitchButton.setEnabled(False)
    self.saveButton.setEnabled(False)

    collectButton.clicked.connect(self.collectFunction)
    self.showButton.clicked.connect(self.showFunction)
    self.stitchButton.clicked.connect(self.stitchFunction)
    self.saveButton.clicked.connect(self.saveFunction)
    quitButton.clicked.connect(self.quitFunction)

  def collectFunction(self):
    self.showButton.setEnabled(False)
    self.stitchButton.setEnabled(False)
    self.saveButton.setEnabled(False)
    self.label.setText('c를 여러번 눌러 수집, 끝나면 q로 비디오 종료')
  
    self.cap = cv.VideoCapture(1, cv.CAP_DSHOW) # 1번 카메라 사용으로 수정
    if not self.cap.isOpened(): sys.exit('카메라 연결 실패')
  
    self.imgs = []
    while True:
      ret, frame = self.cap.read()
      if not ret: break
  
      cv.imshow('video display', frame)
  
      key = cv.waitKey(1)
      if key == ord('c'):
        self.imgs.append(frame) #저장
      elif key == ord('q'):
        self.cap.release()
        cv.destroyWindow('video display')
        break
      
    if len(self.imgs) >= 2:
      self.showButton.setEnabled(True)
      self.stitchButton.setEnabled(True)
      self.saveButton.setEnabled(True)
  
  def showFunction(self):
    self.label.setText('수집 영상은' + str(len(self.imgs)) + '장')
    stack = cv.resize(self.imgs[0], dsize = (0, 0), fx = 0.3, fy= 0.3)
    for i in range(1, len(self.imgs)):
      stack = np.hstack((stack, cv.resize(self.imgs[i], dsize = (0, 0), fx = 0.3, fy = 0.3)))
    cv.imshow('Img collection', stack)
  
  def stitchFunction(self):
    stitcher = cv.Stitcher_create()
    status, self.img_stitched = stitcher.stitch(self.imgs)
    if status == cv.STITCHER_OK:
      cv.imshow('Img stitched panorama', self.img_stitched) 
    else:
      winsound.Beep(2000, 600)
      self.label.setText('파노라마 제작 실패')
  
  def saveFunction(self):
    fname = QFileDialog.getSaveFileName(self, '파일저장', './')
    cv.imwrite(fname[0], self.img_stitched)
  
  def quitFunction(self):
    if hasattr(self, 'cap'):  # cap이 있을 때만 release
        self.cap.release()
    cv.destroyAllWindows()
    self.close()
    
app = QApplication(sys.argv)
win = Panorama()
win.show()
app.exec_()

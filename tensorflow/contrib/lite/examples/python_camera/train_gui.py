import sys
import shutil
import signal
import os
import math
from PyQt4 import QtGui
from PyQt4 import QtCore
from capturer import Capturer
import Image
import re

# example_dir = ""
widget = None

cameraicon_xpm = [
"32 32 5 1",
"   c None",
".  c #000000",
"+  c #B5B5B5",
"@  c #FFFFFF",
"#  c #323232",
"                                ",
"                                ",
"                                ",
"                                ",
"                                ",
"                                ",
"           .........            ",
"     ..   .+++++++++.  ....     ",
"    ....  .+++++++++.  ....     ",
"  .........+++++++++.........   ",
" .++++++++++++@@@++++++++++++.  ",
" .++++++++++@@...@@+++++@..++.  ",
" .+++++++++@.......@++++...++.  ",
" .++++++++@.........@+++...++.  ",
" .+++++++@...........@+++++++.  ",
" .+++++++@...........@+++++++.  ",
" .+++++++@...........@+++++++.  ",
" .+++++++@...........@+++++++.  ",
" .++++++++@.........@++++++++.  ",
" .+++++++++@.......@+++++++++.  ",
" .++++++++++@@...@@++++++++++.  ",
" .++++++++++++@@@++++++++++++.  ",
" .+++++++++++++++++++++++++++.  ",
" ..#.........................   ",
"                                ",
"                                ",
"                                ",
"                                ",
"                                ",
"                                ",
"                                ",
"                                "]

capturer = Capturer()
capturer.Start()
def handle_ctrlc(*args):
    capturer.Shutdown()
    sys.exit(0)



class TrainModel:
  def __init__(self, basedir):
    self.basedir = basedir
    if not os.path.isdir(self.basedir):
      os.makedirs(self.basedir)
    self._classes = [x for x in os.listdir(self.basedir)
                     if os.path.isdir(os.path.join(self.basedir, x))]

  def classPath(self, classname):
    """Returns where images for the given class `classname` are stored."""
    return os.path.join(self.basedir, classname)

  def addClass(self, classname):
    """Add a class named `classname`. This creates the appropriate directory."""
    os.mkdir(self.classPath(classname))
    self._classes.append(classname)

  def images(self, classname):
    return os.listdir(self.classPath(classname))

  def saveClassImage(self, classname, image):
    image = Image.fromarray(image)
    classpath = self.classPath(classname)
    num = len(os.listdir(classpath))
    image.save(os.path.join(classpath, "%05d.jpg" % num))


  def train(self):
    fp = os.popen(
        "python tensorflow/examples/image_retraining/retrain.py"
        " --image_dir=\"{imagedir}\" --architecture={model}"
        " --how_many_training_steps={steps} 2>&1".format(
            imagedir=self.basedir, steps=1000, model="mobilenet_1.0_224"),
        "r")
    while 1:
      line = fp.readline()
      if not line: break
      m = re.match("^.+Validation accuracy = (\d+\.\d+).+$", line)
      if m:
        # self. float(m.group(1))
        print "Validation %f%%"% float(m.group(1))


  @property
  def classes(self):
    return self._classes

class LossGraph(QtGui.QWidget):
  def __init__(self, train_model, parent=None):
    QtGui.QWidget.__init__(self, parent)
    self.setMinimumWidth(512)
    self.setMinimumHeight(512)
    self.points = []
    self.xmin = 0
    self.xmax = 1000
    self.ymin = 0
    self.ymax = 100



  def paintEvent(self, event):
    painter = QtGui.QPainter(self)
    painter.setPen(self.palette().foreground().color());
    old = None
    for x,y in self.points:
      xx = (x - self.xmin) / (self.xmax-self.xmin) * self.width()
      yy = (y - self.ymin) / (self.ymax-self.ymin) * self.height()
      print xx,yy
      old = xx,yy
      if old:
        print "HI!"
        painter.drawLine(old[0], old[1], xx, yy)




class TrainInterface(QtGui.QWidget):
  def __init__(self, train_model, parent=None):
    QtGui.QWidget.__init__(self, parent)

    self.image = None
    self.train_model = train_model

    toplayout = QtGui.QHBoxLayout()
    self.setLayout(toplayout)


    layout = QtGui.QVBoxLayout()
    toplayout.addLayout(layout)


    layout.addWidget(QtGui.QLabel("<b>Classes</b>"))

    self.classAddItemBox = QtGui.QLineEdit()
    self.classAddItemBox.setPlaceholderText("new class")
    layout.addWidget(self.classAddItemBox)
    self.classWidget = QtGui.QListWidget()
    layout.addWidget(self.classWidget)

    hlayout = QtGui.QHBoxLayout()
    hlayout.addWidget(QtGui.QLabel("<b>Images</b>"))
    self.captureButton = QtGui.QPushButton("Capture")
    self.captureButton.setIcon(QtGui.QIcon(QtGui.QPixmap(cameraicon_xpm)))


    hlayout.addWidget(self.captureButton)
    layout.addLayout(hlayout)
    self.imageWidget = QtGui.QListWidget()
    layout.addWidget(self.imageWidget)

    self.trainButton = QtGui.QPushButton("Train")
    layout.addWidget(self.trainButton)

    rightlayout = QtGui.QVBoxLayout()
    rightlayout.addWidget(QtGui.QLabel("<b>Live Camera</b>"))
    self.image_label = QtGui.QLabel()
    rightlayout.addWidget(self.image_label)
    self.graph = LossGraph(None)
    rightlayout.addWidget(self.graph)
    rightlayout.addItem(QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))

    toplayout.addLayout(rightlayout)



    self.repopulateClasses()

    self.classAddItemBox.connect(self.classAddItemBox, QtCore.SIGNAL("returnPressed()"), self.addNewClass)
    self.classWidget.connect(self.classWidget, QtCore.SIGNAL("currentRowChanged(int)"), self.setActiveClass)
    self.captureButton.connect(self.captureButton, QtCore.SIGNAL("pressed()"), self.captureImageForClass)
    self.trainButton.connect(self.trainButton, QtCore.SIGNAL("pressed()"), self.train)

    capturer.Subscribe(self.getImage, (224,224))

    self.timer = QtCore.QTimer()
    self.timer.setInterval(10)
    self.timer.connect(self.timer,QtCore.SIGNAL("timeout()"), self.updateImage)
    self.timer.start()


  def train(self):
    self.train_model.train()

  def updateImage(self):
    if not hasattr(self,"image"): return
    qimage = QtGui.QImage(self.image, 224, 224, QtGui.QImage.Format_RGB888)
    qpixmap = QtGui.QPixmap.fromImage(qimage)
    self.image_label.setPixmap(qpixmap)


  def getImage(self, image):
    self.image = image

  def addNewClass(self):
    self.train_model.addClass(str(self.classAddItemBox.text()))
    self.classAddItemBox.setText("")
    self.repopulateClasses()
    self.classWidget.setCurrentRow(self.classWidget.count()-1)

  def setActiveClass(self):
    self.repopulateClassImages()

  def repopulateClasses(self):
    old_selection_row = self.classWidget.currentRow()
    if old_selection_row != -1:
      old_selection_text = str(self.classWidget.item(old_selection_row).text())
    else:
      old_selection_text = None
    self.classWidget.clear()
    for idx, i in enumerate(self.train_model.classes):
      self.classWidget.addItem(i)
      if i == old_selection_text:
        self.classWidget.setCurrentRow(idx)

  def captureImageForClass(self, ):
    old_selection_row = self.classWidget.currentRow()
    if old_selection_row != -1:
      old_selection_text = str(self.classWidget.item(old_selection_row).text())
      self.train_model.saveClassImage(old_selection_text, self.image)
      self.repopulateClassImages()


  def repopulateClassImages(self):
    currentRow = self.classWidget.currentRow()
    if currentRow == -1: return
    currentClass = str(self.classWidget.item(currentRow).text())
    self.imageWidget.clear()
    for x in self.train_model.images(currentClass):
      self.imageWidget.addItem(x)


  def closeEvent(self, event):
      handle_ctrlc()


if __name__ == "__main__":
  app = QtGui.QApplication(sys.argv)
  model = TrainModel("set1")
  widget = TrainInterface(model)
  widget.show()


  signal.signal(signal.SIGINT, handle_ctrlc)


  sys.exit(app.exec_())



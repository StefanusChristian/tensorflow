import signal
import sys
from PyQt4 import QtGui
from PyQt4 import QtCore
from  capturer import Capturer
import time
import numpy as np
import classify_model
#import test



#model = classify_model.Model("mobilenet_quant_v1_224.tflite","labels.txt",4)
#model = classify_model.Model("people.tflite","people_labels.txt",4)
model = classify_model.Model("andy2.tflite","andy2_labels.txt",4)
#model = classify_model.Model("andy.tflite","andy_labels.txt",4)
# capturer = Capturer()
# capturer.Start()


def FormatProbs(probs):
    text = ""
    for index, label, prob in probs:
        if prob > .5: color = "green"
        elif prob > .3: color =  "yellow"
        elif prob > .15: color =  "orange"
        else: color="red"
        prob_length = prob * 255.
        text += "<table width='%d' style='background-color:%s'><tr><td>&nbsp;</td></tr></table> %4.3f %s"%(prob_length, color, prob, label)
    text += ""
    s = time.time()
    e = time.time()
    sort_time = e - s
    return text


class InferenceView(QtGui.QWidget):
    def __init__(self, capturer):
        QtGui.QWidget.__init__(self)
        self.setWindowTitle("TensorFlow Lite Inference")
        self.text_label = QtGui.QLabel()
        layout = QtGui.QHBoxLayout()
        self.image_label = QtGui.QLabel()
        self.setLayout(layout)
        layout.addWidget(self.text_label)
        layout.addWidget(self.image_label)
        # Make a timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.connect(self.timer,QtCore.SIGNAL("timeout()"), self.Update)
        self.timer.start()
        self.toproc = None

        capturer.Subscribe(self.ReceiveImage, (224,224))

    def closeEvent(self, event):
        handle_ctrlc()

    def Update(self):
        if not self.toproc: return
        my_bitmap, my_labels, tflite_time = self.toproc
        self.toproc = None
        text = FormatProbs(my_labels)
        text += "<table><tr><td>tflite</td><td width='25%%'>%.0f ms</td></tr></table> "%(tflite_time)

        self.text_label.setText(text)
        qimage = QtGui.QImage(my_bitmap, 224, 224, QtGui.QImage.Format_RGB888)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.image_label.setPixmap(qpixmap)


    def ReceiveImage(self, bitmap):
        input = np.reshape(bitmap, (1,224,224,3))
        start = time.time()
        probs = model.invoke(input, 4)
        elapsed = time.time() - start
        self.toproc = [bitmap, probs, elapsed]

def handle_ctrlc(*args):
    capturer.Shutdown()
    sys.exit(0)


# timer.connect(timer,QtCore.SIGNAL("timeout()"), printstuff)
# widget.setAttribute(QtGui.WA_DeleteOnClose)



if __name__ == "__main__":

    signal.signal(signal.SIGINT, handle_ctrlc)
    # test.inference_subscribers.append(recvimage)


    app = QtGui.QApplication(sys.argv)
    view = InferenceView()
    view.show()
    app.setQuitOnLastWindowClosed(True)
    app.exec_()

import classify_gui
import train_gui
from  capturer import Capturer
import signal
import sys
from PyQt4 import QtGui
from PyQt4 import QtCore


def handle_ctrlc(*args):
    capturer.Shutdown()
    sys.exit(0)
signal.signal(signal.SIGINT, handle_ctrlc)


capturer = Capturer()
capturer.Start()

app = QtGui.QApplication(sys.argv)
mainwin = QtGui.QTabWidget()


view = classify_gui.InferenceView(capturer)
#view.show()
mainwin.setWindowTitle("TensorFlow Lite Inference")
mainwin.setMinimumWidth(800)
model = train_gui.TrainModel("set1")
train_view = train_gui.TrainInterface(model, capturer)
mainwin.setMinimumHeight(600)
mainwin.addTab(train_view, "Train")
mainwin.addTab(view, "Inference")
mainwin.show()
app.setQuitOnLastWindowClosed(True)
app.exec_()

import numpy as np
import threading
import os

class OpenCvCameraCapture:
    def __init__(self, capturer):
        self.capturer = capturer
        self.cap = None
        try:
            import cv2
            print("Found OpenCV (cv2)")
            self.cv2 = cv2
            cams = "/sys/class/video4linux"
            paths = os.listdir(cams)
            if paths:
                device = int(paths[0].replace("video", ""))
                print("Trying V4L device %s..." % paths[0])
                self.cap = cv2.VideoCapture(device)
        except ImportError:
            pass

    def GetLoop(self):
        while self.capturer.STATE!=self.capturer.STATE_SHOULD_STOP:
            ret, frame = self.cap.read()
            if not ret:
                return
            frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            for subscriber, desired_size in self.capturer.subscribers:
                sub_frame = self.cv2.resize(frame, desired_size)
                subscriber(sub_frame)

    def Valid(self):
        return self.cap is not None


class RaspberryPiCamera:
    def __init__(self, capturer):
        self.capturer = capturer
        self.cam = None

        try:
            import picamera
            print("Found picamera")
            self.cam = picamera.PiCamera()
            self.cam.framerate = 24
            self.cam.resolution=(224, 224)
            print("Trying Raspberry Pi Camera...")

        except ImportError:
            pass

    def GetLoop(self):
        image = np.empty((224*224*3), dtype=np.uint8)
        for i in self.cam.capture_continuous(image, use_video_port=True, format='rgb'):

            if self.capturer.STATE==self.capturer.STATE_SHOULD_STOP:
                return
            for subscriber, desired_size in self.capturer.subscribers:
                subscriber(np.array(image))

    def Valid(self):
        return self.cam is not None



class Capturer:
    def __init__(self):
        self.STATE_STOPPED = 0
        self.STATE_RUNNING = 1
        self.STATE_SHOULD_STOP = 2

        self.STATE = self.STATE_STOPPED
        self.subscribers = []


    def Subscribe(self, fn, desired_size):
        self.subscribers.append((fn, desired_size))

    def Shutdown(self):
        global should_exit
        self.STATE = self.STATE_SHOULD_STOP

    def ThreadMain(self, particular_camera_object):
        def f():
            particular_camera_object.GetLoop()
            self.STATE = self.STATE_STOPPED
        return f


    def Start(self):
        if self.STATE != self.STATE_STOPPED: return
        capture_backends = []
        capture_backends.append(OpenCvCameraCapture(self))
        capture_backends.append(RaspberryPiCamera(self))

        self.STATE = self.STATE_RUNNING
        for i in capture_backends:
            if i.Valid():
                t = threading.Thread(target=self.ThreadMain(i))
                t.start()
                return

        raise RuntimeError("No camera found!")


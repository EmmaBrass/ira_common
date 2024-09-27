# Setup and initialisation for the camera
import logging
import cv2
from subprocess import PIPE, run

class Camera():

    def __init__(self, port_num) -> None:
        self.port_num = port_num
        self.start_up()
    
    def start_up(self):
        # Load camera video feed.   
        cam_id = self.port_num
        self.cam = cv2.VideoCapture(cam_id)
        if not self.cam.isOpened():
            print("Error: Could not open the USB camera.")
            exit()
        #cam.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
        print("Have turned on camera now")

    def read(self):
        for i in range(10):
            ret, frame = self.cam.read() 
        return frame

    def release(self):
        return self.cam.release()

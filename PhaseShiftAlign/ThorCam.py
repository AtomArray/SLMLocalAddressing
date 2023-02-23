import ctypes
import ctypes.util
import time

import sys
import os

analysis_tools_path = "Y:\\analysis_tools\\"
contents = os.listdir(analysis_tools_path)
for subdir in contents:
    if not analysis_tools_path in sys.path:
        sys.path.append(analysis_tools_path + subdir)

import ThorCam_v2
from ThorCam_v2 import Camera


class ThorCam():
    def __init__(self, color="blue"): 
        camera_data = ThorCam_v2.create_camera_list(5)
        ThorCam_v2.uc480.is_GetCameraList(ctypes.byref(camera_data))

        # Define serial numbers for the two cameras.
        blue_cam_serno = "4102709291" # Monitoring blue Rydberg beam
        red_cam_serno = "4103004973" # Monitoring red Rydberg beam
        blue_new_cam_serno = "4002820388" # New camera to monitor the blue Rydberg beam
        phase_shift_serno = "4102801671" # Phase shift camera
        # phase_shift_align_serno = "4102709291" # Phase shift alignment
        phase_shift_align_serno = "4102801671" # Phase shift alignment

        color = color.lower()

        if color=="blue":
            target_cam_serno = blue_cam_serno
        elif color=="red":
            target_cam_serno = red_cam_serno
        elif color=="blue_new":
            target_cam_serno = blue_new_cam_serno
        elif color=="phase_shift":
            target_cam_serno = phase_shift_serno
        elif color=="phase_shift_align":
            target_cam_serno = phase_shift_align_serno


        camera = None
        for cam in camera_data.Cameras:
            if cam.SerNo == target_cam_serno.encode():
                camera = cam
                print("Serial #: ", cam.SerNo, ". ", "%s cam found!" %color)
                sys.stdout.flush()
            else:
                print("NOT THIS", cam.SerNo, target_cam_serno)
                sys.stdout.flush()


        if camera == None:
            print("Didn't find camera!")
            return None


        cam = Camera(camera.CameraID, "Y:\\analysis_tools\\ThorCam\\camera_profile.tcp")
        # cam = Camera(camera.CameraID)
        # cam = Camera(camera.CameraID, "Y:\\analysis_tools\\ThorCam\\thorcam_new_blue_profile.tcp")



        if color=="blue":
            self.aoi = [530, 544, 80, 80]
            target_exposure = 0.012
        elif color=="red":
            self.aoi = [660, 700, 80, 80]

            target_exposure = 1.0
        elif color=="blue_new":
            self.aoi = [582, 517, 80, 80]
            # self.aoi = [0, 0, 1280, 1280]
            target_exposure = 1.0
        elif color=="phase_shift":
            self.aoi = [630, 130, 80, 80]
            target_exposure = 0.15
        elif color=="phase_shift_align":
            # self.aoi = [430, 365, 820, 70]
            self.aoi = [540, 360, 680, 70] # Good for 20 traps that we use
            # self.aoi = [120, 300, 1160, 200]


            target_exposure = 0.6


        cam.setAOI(self.aoi[0], self.aoi[1], self.aoi[2], self.aoi[3])


        print("%s: target_exposure" %color, target_exposure)

        cam.setExposure(target_exposure)
        print("Current exposure:", cam.getExposure())
        sys.stdout.flush()


        self.cam = cam


    def shutDown(self):
        self.cam.shutDown()

    def getAOI(self):
        return self.aoi # This is hardcoded - should fix!

    def getImage(self):
        return self.cam.captureImage()


def main():
    pass
    # blueCam = ThorCam()
    # blueCam.shutDown()



if __name__ == '__main__':
        main()

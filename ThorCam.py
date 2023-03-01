import ctypes
import ctypes.util
import time

import sys
import os


import matplotlib.pyplot as plt

import ThorCamHardware
from ThorCamHardware import Camera
import numpy as np



class ThorCam():
	is_test = False

	def __init__(self, color="imaging"): 
		camera_data = ThorCamHardware.create_camera_list(2)
		ThorCamHardware.uc480.is_GetCameraList(ctypes.byref(camera_data))

		# Define serial numbers for the two cameras.
		# blue_cam_serno = "4102709291" # Monitoring blue Rydberg beam
		# red_cam_serno = "4103004973" # Monitoring red Rydberg beam
		# blue_new_cam_serno = "4002820388" # New camera to monitor the blue Rydberg beam
		# phase_shift_serno = "4102709291" # Phase shift camera
		# trapping_cam_serno = "4102709291"
		# imaging_cam_serno = "4102801671"
		trapping_cam_serno = "4102620791"
		imaging_cam_serno = "4103606955"

		color = color.lower()

		if color=="imaging":
			target_cam_serno = imaging_cam_serno
		elif color=="trapping":
			target_cam_serno = trapping_cam_serno
		elif color=="test":
			self.is_test = True
			return


		camera = None
		for cam in camera_data.Cameras:
			print("Serial #: ", cam.SerNo)
			if cam.SerNo.decode() == target_cam_serno:
				camera = cam
				print("Serial #: ", cam.SerNo, ". ", "%s cam found!" %color)


		if camera == None:
			print("Didn't find camera!")
			return


		cam = Camera(camera.CameraID, "camera_profile_test.tcp")

		# self.cam = cam
		# return
		# cam = Camera(camera.CameraID, "ThorCam_savedProfile.tcp")
		
		# cam = Camera(camera.CameraID) # DK Dec 1 2018
		# cam = Camera(camera.CameraID, "Y:\\analysis_tools\\ThorCam\\thorcam_new_blue_profile.tcp")



		if color=="trapping": #the vertical coordinate is inverted relative to the camera output
			self.aoi = [0, 0, 1280, 1024]

			target_exposure = 2 # ms unit
		elif color=="imaging":
			#self.aoi = [400, 200, 700, 800]
			self.aoi = [000, 0, 1280, 1024]
			target_exposure = 0.1 # ms unit
		else:
			self.aoi = [0, 0, 1280, 1024]
			target_exposure = 1 # ms unit




		cam.setAOI(self.aoi[0], self.aoi[1], self.aoi[2], self.aoi[3])

		print("%s: target_exposure" %color, target_exposure)
		cam.setExposure(target_exposure)

		cam.printSettings()

		self.cam = cam
		cam.captureImageTest()
	# Nov 8 2018, Added by DK
	def setExposure(self,exposure):
		if self.is_test:
			return True


		self.cam.setExposure(exposure)
		return self.cam.getExposure()
		# print("Current exposure:", self.cam.getExposure())

	def setAOI(self, a, b, c, d):
		if self.is_test:
			return True
		self.cam.setAOI(a, b, c, d)


	def shutDown(self):
		if self.is_test:
			return True

		self.cam.shutDown()

	def getAOI(self):
		if self.is_test:
			return True

		return self.aoi # This is hardcoded - should fix!

	def getImage(self):
		if self.is_test:
			return np.zeros((128, 128))
		return self.cam.captureImage()


def main():
	# pass
	blueCam = ThorCam(color="imaging")
	im = blueCam.getImage()

	plt.imshow(im)
	plt.colorbar()
	plt.show()


	im = blueCam.getImage()
	plt.imshow(im)
	plt.colorbar()
	plt.show()


	blueCam.shutDown()



if __name__ == '__main__':
	main()
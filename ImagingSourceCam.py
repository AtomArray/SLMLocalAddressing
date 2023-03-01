# -*- coding: utf-8 -*-
# Adapted from GitHub: TheImagingSource/IC-Imaging-Control-Samples
"""
Created on Mon Nov 21 09:46:46 2016

Sample for tisgrabber to OpenCV Sample 2

Open a camera by name
Set a video format hard coded (not recommended, but some peoples insist on this)
Set properties exposure, gain, whitebalance
"""
import ctypes
import tisgrabber as IC
import numpy as np
import time
import matplotlib.pyplot as plt


class ImagingSourceCam:
	is_test = False

	def __init__(self):
		# Create the camera object.
		self.Camera = IC.TIS_CAM()

		# List availabe devices as uniqe names. This is a combination of camera name and serial number
		Devices = self.Camera.GetDevices()

		for i in range(len( Devices )):
			print( str(i) + " : " + str(Devices[i]))

		# Open a device with hard coded unique name:
		# self.Camera.open("DMM 27UP031-ML 31910373")
		# self.Camera.open("DMx 72BUC02 18810076")
		#self.Camera.open("DMK 27BUP031 47910655")
		self.Camera.open("TIS UVC Device 12120711")


		# Print all available formats for camera
		# fmts =self.Camera.GetVideoFormats()
		# for fmt in fmts:
		# 	print(fmt)

		# Should use: Y800 instead of RGB24
		# self.Camera.SetVideoFormat("Y800 (1296x972) [Binning 2x]") # "RGB24 (2592x1944)")
		self.Camera.SetVideoFormat("Y16 (2592x1944)") # For the new imaging and objective 3-11-2022
		# self.Camera.SetVideoFormat("Y16 (1296x972) [Binning 2x]")
		self.Camera.SetFrameRate(15) ### SET FRAME RATE FROM 4 TO 1 ###


		if self.Camera.IsDevValid() != 1:
			print("ERROR: Unable to connect to camera.")
			return

		target_exposure_ms = 0.05 #ms unit
		gain = 1


		self.initializeCameraSettings(target_exposure_ms,gain)
		self.Camera.StartLive(0)

	def __del__(self):
		self.Camera.StopLive()


	def initializeCameraSettings(self,target_exposure_ms,gain):
		self.setExposure(target_exposure_ms)
		self.setGain(gain)
		

	def setExposure(self,exposure):
		if self.is_test:
			return True
		exposure_seconds = 0.001*exposure

		self.Camera.SetPropertySwitch("Exposure","Auto",0)
		 # "0" is off, "1" is on.
		self.Camera.SetPropertyAbsoluteValue("Exposure","Value",exposure_seconds)

		ExposureTime=[0]
		self.Camera.GetPropertyAbsoluteValue("Exposure","Value",ExposureTime)
		print("Exposure time abs: ", 1000*ExposureTime[0])
		return(1000*ExposureTime[0])

	def setGain(self,gain):
		if self.is_test:
			return True

		self.Camera.SetPropertySwitch("Gain","Auto",0)
		 # "0" is off, "1" is on.
		self.Camera.SetPropertyValue("Gain","Value",gain)

		GainValue = self.Camera.GetPropertyValue("Gain","Value")
		print("Gain value: ", GainValue)
		return(GainValue)

	def setAOI(self, a, b, c, d):
		if self.is_test:
			return True
		#stub

	def getAOI(self):
		if self.is_test:
			return True
		#stub

	def shutDown(self):
		if self.is_test:
			return True

		self.Camera.StopLive()

	def getImage(self):
		self.Camera.SnapImage()

		im = self.Camera.GetImage()
		im = im[:, :, 0] # Take one color channel

		# 2x2 software binning of image
		width, height = im.shape
		reshaped = im.reshape(width//2, 2, height//2, 2)
		binned_im = reshaped.mean(3).mean(1)

		im = binned_im

		return im


		# # im = np.array(self.Camera.GetImage())
		# # im_mean = np.mean(im, axis=2)
		# im_mean = np.mean(self.Camera.GetImage(), axis=2).astype(np.int)
		# im_mean = np.bitwise_and(im_mean, 0xf0)
		# if np.max(im_mean) == 0:
		# 	im_mean[0, 0] = 1
		# # print(im_mean)
		# return im_mean


def main():
	cam = ImagingSourceCam()
	# print(cam)
	cam.getImage() #this needs to be done for some reason such that the exposure is successfuly set on first try (becomes correct on second image), but set gain works OK without it.
	im = np.array(cam.getImage())

	plt.imshow(im, cmap="Greys_r", vmin=0, vmax=20)
	plt.show()

	cam.shutDown()

# def main2():
# 	a = np.random.rand(100, 100)
# 	plt.imshow(a)
# 	plt.show()


if __name__ == "__main__":
	main()
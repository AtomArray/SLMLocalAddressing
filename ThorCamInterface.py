import ctypes
import ctypes.util
import time
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
import pyqtgraph as pg
from scipy.optimize import curve_fit

import sys
import os
import numpy as np

import matplotlib.pyplot as plt

from ThorCam import ThorCam
from ImagingSourceCam import ImagingSourceCam

UPDATE_THREAD_TIME = 0.5
LAST_CALIBRATION_FILENAME = "LastThorCamCalibration.npz"

# This class encodes a table view with several
# user programmable settings for how the SLM operates.
# More settings will likely be added over time.
# A dictionary encodes the settings displayed and edited in the table.
class ThorCamSettings(QtWidgets.QTableWidget):
	ExposureLabel = "Exposure (ms)"
	ThresholdLabel = "Threshold"
	ShowTargetsLabel = "Show Targets"

	def __init__(self, thorcam):
		super().__init__()

		self.thorcam = thorcam

		self.settings = {
			self.ExposureLabel: 20,
			self.ThresholdLabel: 80,
			self.ShowTargetsLabel: 1
		}


		self.setColumnCount(2)
		self.setRowCount(len(self.settings.keys()))

		self.setFont(QtGui.QFont('Arial', 9))

		self.keys = list(self.settings.keys())
		for i in range(len(self.keys)):
			self.setItem(i, 0, QtWidgets.QTableWidgetItem(self.keys[i]))
			self.setItem(i, 1, QtWidgets.QTableWidgetItem("%s" %self.settings[self.keys[i]]))

			self.item(i, 0).setFlags(QtCore.Qt.ItemIsEnabled)

		self.setMinimumWidth(170)
		header = self.horizontalHeader()
		header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
		header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)



		self.itemChanged.connect(self.settingChanged)

	# Update dictionary key-value pair according to edited table item.
	def settingChanged(self, item):
		print("Setting changed")
		sys.stdout.flush()
		index = item.row()

		settingName = self.item(index, 0).text()
		try:
			newValue = float(self.item(index, 1).text())

			self.settings[settingName] = newValue

			if settingName == self.ExposureLabel:
				self.setExposure(newValue)

		except:
			print("Error! Can't parse input:", self.settings[settingName])

	def setExposure(self, newValue):
		newExposure = self.thorcam.setExposure(newValue)
		self.settings[self.ExposureLabel] = newExposure
		self.item(self.keys.index(self.ExposureLabel), 1).setText(str(newExposure))


	def getExposure(self):
		return float(self.settings[self.ExposureLabel])

	def getThreshold(self):
		return float(self.settings[self.ThresholdLabel])

	def getShowTargets(self):
		return bool(self.settings[self.ShowTargetsLabel])




class ThorCamInterface(QtWidgets.QWidget):
	def __init__(self, cameraName="imaging", enable=True):
		super().__init__()
		self.imageDisplay = pg.ImageView()
		self.imageDisplay.ui.roiBtn.hide()
		self.imageDisplay.ui.menuBtn.hide()
		self.imageDisplay.ui.histogram.hide()


		self.updateThread = CameraUpdateThread()
		self.updateThread.updateSignal.connect(self.threadUpdateImage)


		if enable:
			self.thorcam = ImagingSourceCam() # Use transmission ImagingSource camera
			#self.thorcam = ThorCam("trapping") # Use trapping side Thorcam #4/21/2022 DB commented this and uncommented above line
			# Using the ImagingSourceCam for testing Sophie 2/28/2023 		
		else:
			self.thorcam = ThorCam("test")



		self.settings = ThorCamSettings(self.thorcam)
		self.histogram = pg.PlotWidget()
		self.histogramData = self.histogram.plot([0, 1], [0], stepMode=True)
		self.thresholdMarker = pg.InfiniteLine(angle=90)
		self.histogram.addItem(self.thresholdMarker)
		self.histogram.setFixedHeight(100)

		self.numIdentifiedTrapsLabel = QtWidgets.QLabel("# Identified traps:")

		self.settings.setExposure(2)

		self.positionsToMark = []


		# Define layout of graphical interface
		self.layout = QtWidgets.QGridLayout(self)

		self.layout.addWidget(self.imageDisplay, 0, 0)
		self.layout.addWidget(self.settings, 0, 1)
		self.layout.addWidget(self.histogram, 1, 0)
		self.layout.addWidget(self.numIdentifiedTrapsLabel, 1, 1)


		self.origin = [0, 0]
		self.xMarker = [0, 0]
		self.yMarker = [0, 0]
		self.corner = [0, 0]
		self.eX = np.array([0, 1])
		self.eY = np.array([1, 0])
		self.loadPreviousCalibration()

		self.slmCoordinates = []


		# self.setFixedWidth(750)
		self.setFixedHeight(500)

		self.updateThread.start()


	def pauseUpdateThread(self):
		self.updateThread.threadShouldUpdate = False

	def resumeUpdateThread(self):
		self.updateThread.threadShouldUpdate = True

	def prepareForCalibration(self):
		self.settings.setExposure(0.1)
		self.pauseUpdateThread()


	def loadPreviousCalibration(self):
		try:
			print("Loading last ThorCam calibration...")
			lastCalibration = np.load(LAST_CALIBRATION_FILENAME)

			self.eX = lastCalibration["eX"]
			self.eY = lastCalibration["eY"]
			self.origin = lastCalibration["origin"]
			self.xMarker = lastCalibration["xMarker"]
			self.yMarker = lastCalibration["yMarker"]
			self.corner = lastCalibration["corner"]
		except Exception as e:
			print("Unable to load ThorCam calibration:", e)


	def doneWithCalibration(self, cornerPositions, norm, offset_from_origin):
		self.resumeUpdateThread()

		self.origin, self.xMarker, self.yMarker, self.corner = np.array(cornerPositions)
		

		self.eX = (self.xMarker - self.origin) / norm # Basis vectors for SLM coordinates
		self.eY = (self.yMarker - self.origin) / norm

		self.origin -= self.eX * offset_from_origin + self.eY * offset_from_origin

		np.savez(LAST_CALIBRATION_FILENAME,
			eX=self.eX,
			eY=self.eY,
			origin=self.origin,
			xMarker=self.xMarker,
			yMarker=self.yMarker,
			corner=self.corner)

	def convertSLMCoordsToCameraCoords(self, y, x):
		pos = (self.origin + self.eX * x + self.eY * y).astype(np.int32)

		return pos


	def markSLMCoordinates(self, coords):
		self.slmCoordinates = coords


	def setFullAOI(self):
		self.thorcam.setAOI(0, 0, 1280, 1024)


	def threadUpdateImage(self):
		self.updateImage(ignoreZeroOrder=True)

	def updateNumTrapsLabel(self, num, std=0.0):
		self.numIdentifiedTrapsLabel.setText("# Identified Traps: %d\nStd. dev: %.1f%%" %(int(num), 100.0*std))


	def updateImage(self, num_traps=1000, ignoreZeroOrder = False, singleTrapSLMCoord=None, useGaussian=False):
		if self.thorcam.is_test:
			return []
		# print("UPDATING IMAGE")
		self.thresholdMarker.setValue(self.settings.getThreshold())

		im = self.thorcam.getImage()
		#print(im)
		# np.save("LastImage.npy", im)
		if type(im) == type(None):
			return []


		# print("IDENTIFYING TRAPS")
		if type(singleTrapSLMCoord) == type(None):
			# Find up to num_traps traps
			traps = self.identifyTraps(im, num_traps=num_traps, threshold=self.settings.getThreshold(),
									   ignoreZeroOrder=ignoreZeroOrder, useGaussian=useGaussian)
		else:
			# Try to find a single trap near specific SLM coordinates.
			traps = self.identifySingleTrap(im, singleTrapSLMCoord)

		# print("DONE")
		# traps = [] #DB, stub



		rgbImage = np.repeat(im[:, :, np.newaxis], 3, axis=2) # Duplicate image into three-channel RGB 


		def setPixelValue(image, x, y, color):
			# If out of bounds, do nothing
			if int(x) < 0 or int(x) >= rgbImage.shape[0]:
				return
			if int(y) < 0 or int(y) >= rgbImage.shape[1]:
				return

			rgbImage[int(x), int(y), :] = color

		def markPosition(image, pos, radius=2, color=[255, 0, 0]):
			x0, y0 = np.array(pos, dtype=np.int32)

			if x0 <= 0 or y0 <= 0:
				return


			for x in range(x0-radius, x0+radius+1):
				for y in range(y0-radius, y0+radius+1):
					setPixelValue(rgbImage, x, y, color)

		# Mark position of traps with a single red dot
		for t in traps:
			setPixelValue(rgbImage, t[0], t[1], [255, 0, 0])
			# rgbImage[t[1], t[0], :] = [255, 0, 0]


		markPosition(rgbImage, self.origin, radius=2, color=[255, 0, 0])
		markPosition(rgbImage, self.xMarker, radius=2, color=[0, 0, 255])
		markPosition(rgbImage, self.yMarker, radius=2, color=[0, 0, 255])
		markPosition(rgbImage, self.corner, radius=2, color=[0, 255, 0])


		if type(singleTrapSLMCoord) == type(None):
			if self.settings.getShowTargets():
				for c in self.slmCoordinates:
					pos = self.convertSLMCoordsToCameraCoords(c[0], c[1])

					markPosition(rgbImage, pos, radius=1, color=[255, 0, 255])
		else:
			markPosition(rgbImage, traps[0][:2], radius=1, color=[255, 0, 255])

		# Display image
		# It is standard to transpose the image, since this image view displays things typically backwards from normal
		# image representation (columns vs rows) -- see documentation.
		# Also, add white border around image, and fix the min/max levels.
		# lut = Lookup table. It seems necessary to set this to None since we are supplying a full RGB image.
		# 		(I don't know exactly what the lookup table does, but it must map from grayscale --> RGB, and is therefore not necessary here).
		self.imageDisplay.imageItem.setImage(rgbImage.transpose([1,0,2]), border=pg.mkPen((255,255,255)), levels=(0, 255), lut=None)


		if len(traps) == 0:
			self.histogramData.setData([0, 1], [0])
			return []

		hist, bin_edges = np.histogram(traps[:, 2], bins=16, range=[0, 256])
		self.histogramData.setData(bin_edges, hist)


		fractional_variation = np.std(traps[:, 2]) / np.mean(traps[:, 2])
		self.updateNumTrapsLabel(len(traps), fractional_variation)

		# print("Number of identified traps: %d (Fractional std. dev: %.1f%%)" %(len(traps), 100*fractional_variation))

		return traps


	def identifyTraps(self, image, num_traps=1000, threshold=80, blockade_size=10,
						ignoreZeroOrder=False, zeroOrderBlockade = 20, useGaussian=False):

		# image = np.array(im, dtype=np.int32)
		# num_traps = 1

		t1 = time.time()
		# Identify all points in image that are local maxima
		# (They are larger than their right neighbr, left neighbor, upper neighbor, lower neighbor)
		c0 = image[1:-1, 1:-1] > threshold
		c1 = image[1:-1, 1:-1] >= image[2:, 1:-1]
		c2 = image[1:-1, 1:-1] >= image[:-2, 1:-1]
		c3 = image[1:-1, 1:-1] >= image[1:-1, 2:]
		c4 = image[1:-1, 1:-1] >= image[1:-1, :-2]

		localMaximaMask = c0 * c1 * c2 * c3 * c4

		candidateTrapIndices = np.argwhere(localMaximaMask > 0)

		# Currently these indices are oriented within a reduced image cutting off
		# one pixel on all sides.
		# Let's turn them into original image coordinates.
		candidateTrapIndices += 1

		candidateTrapValues = image[candidateTrapIndices[:, 0], candidateTrapIndices[:, 1]]
		
		# Sorted indices of candidate traps in descending peak value
		sortedIndices = np.argsort(candidateTrapValues)[::-1]

		# Mask that shows which pixel coordinates we could select as candidates
		# If we pick one candidate, we mark a small region around it as unavailable
		# in this mask.
		availablePositions = np.ones(image.shape, dtype=np.bool)

		if ignoreZeroOrder:
			x0 = int(max(0, self.origin[0] - zeroOrderBlockade))
			x1 = int(min(image.shape[0]-1, self.origin[0] + zeroOrderBlockade))

			y0 = int(max(0, self.origin[1] - zeroOrderBlockade))
			y1 = int(min(image.shape[1] - 1, self.origin[1] + zeroOrderBlockade))

			availablePositions[x0:x1, y0:y1] = 0

		candidateTraps = []
		for i in sortedIndices:
			ind = candidateTrapIndices[i]
			# print('i = ',i)
			# print('ind = ',ind)

			if availablePositions[ind[0], ind[1]]:
				candidateTraps.append([ind[0], ind[1], candidateTrapValues[i]])
				# print(candidateTrapValues[i])

				x0 = int(max(0, ind[0] - blockade_size))
				x1 = int(min(image.shape[0]-1, ind[0] + blockade_size))

				y0 = int(max(0, ind[1] - blockade_size))
				y1 = int(min(image.shape[1]-1, ind[1] + blockade_size))

				availablePositions[x0:x1, y0:y1] = 0

		candidateTraps = np.array(candidateTraps)

		candidateTraps = candidateTraps[:num_traps]


		
		if len(candidateTraps) == 0:
			return candidateTraps



		# At this point, candidateTraps encodes the position (x, y) and peak pixel value
		# of the center of each trap. We can then try to fit the small region around each
		# peak to a 2D Gaussian, or a paraboloid, or something like this.
		# However, this generally is a bit slow -- order 300-500 ms to fit 400 traps.
		# Since the beams are not perfectly Gaussian, it's also not clear this perfectly captures
		# the peak intensity.

		# A much simpler, much faster approach is to simply take the x,y position and amplitude for the
		# peak pixel value. Let's adopt this approach for now.


		if useGaussian: # Fit to 2D Gaussians
			opt_params = fit_traps_to_2D_gaussians(image, candidateTraps[:, :2]) 					# Fit to 2D Gaussian
		elif False: # Fit to 2D Paraboloid
			opt_params2 = fit_traps_to_2D_paraboloids(image, candidateTraps[:, :2])				# Fit to 2D Paraboloid
		elif False: # Bin around traps
			opt_params = bin_traps(image, candidateTraps[:, :2])
		else: # Take peak pixel value.
			opt_params = np.zeros((len(candidateTraps), 4)) # Four indices: [Amp, x0, y0, sigma]	# Take values from peak pixel
			opt_params[:, 0] = candidateTraps[:, 2] # Amplitudes
			opt_params[:, 1] = candidateTraps[:, 0] # x0
			opt_params[:, 2] = candidateTraps[:, 1] # y0
		
		# plt.subplot(211)
		# plt.plot(candidateTraps[:, 2], label="Peak value")
		# plt.plot(opt_params[:, 0], label="2D Gaussian")
		# plt.plot(opt_params2[:, 0], label="2D Paraboloid")
		# plt.xlabel("Trap index")
		# plt.ylabel("Amplitude")
		# plt.legend()

		# # plt.show()
		# plt.subplot(212)

		# plt.plot(opt_params[:, 0] / opt_params2[:, 0], label="Ratio: Gaussian / Paraboloid")
		# plt.legend()
		# plt.show()
		# for i in range(len(opt_params)):
		# 	print(candidateTraps[i, 2], opt_params[i])
		# print(opt_params)

		if len(opt_params) == 0:
			return []

		relevantParamsMask = np.array([1,2,0])

		return opt_params[:, relevantParamsMask]

	def identifySingleTrap(self, im, coordinates):
		cameraCoords = self.convertSLMCoordsToCameraCoords(coordinates[0], coordinates[1])

		opt_params = fit_traps_to_2D_gaussians(im, [cameraCoords])


		relevantParamsMask = np.array([1,2,0])
		return opt_params[:, relevantParamsMask]

	def __del__(self):
		# return
		self.thorcam.shutDown()

class CameraUpdateThread(QtCore.QThread):
	updateSignal = QtCore.pyqtSignal()

	def __init__(self):
		super().__init__()

		self.threadShouldUpdate = True

	def run(self):
		print("Starting")
		while True:
			if self.threadShouldUpdate:
				self.updateSignal.emit()

			time.sleep(UPDATE_THREAD_TIME)




def paraboloid_2D_fixed_curvature(pos, amplitude, x0, y0):
	curvature = 6.0

	y = pos[0]
	x = pos[1]

	dist_squared = (x - x0)**2.0 + (y - y0)**2.0
	ret_val = amplitude - curvature * dist_squared

	return ret_val.ravel()

def paraboloid_2D(pos, amplitude, x0, y0, curvature):
	y = pos[0]
	x = pos[1]

	dist_squared = (x - x0)**2.0 + (y - y0)**2.0
	ret_val = amplitude - curvature * dist_squared

	return ret_val.ravel()


def gaussian_2D_fixed_offset(pos, amplitude, x_0, y_0, sigma):
	offset = 1
	# x = gaussian_2D(pos, amplitude, x_0, y_0, sigma, offset)
	return gaussian_2D(pos, amplitude, x_0, y_0, sigma, offset)

def gaussian_2D(pos, amplitude, x0, y0, sigma, offset):
	y = pos[0]
	x = pos[1]

	if amplitude < 10.0 or sigma < 3 or sigma > 12:
		return -100000



	dist_squared = (x - x0)**2 + (y - y0)**2
	# dist_squared = np.power(x - x0, 2) + np.power(y - y0, 2)


	# dist_squared2 = (x - x0)**2.0 + (y - y0)**2.0

	ret_val = offset + amplitude * np.exp(-2 * dist_squared / sigma**2.0)
	# ret_val2 = offset + amplitude * np.exp(-2 * dist_squared / sigma**2.0)


	# asdfsd = ret_val.ravel()
	return ret_val.ravel()

def bin_traps(image, trap_positions):
	trap_intensities = []

	opt_params = np.zeros((len(trap_positions), 3))

	for i in range(len(trap_positions)):
		t = trap_positions[i]
		x0, y0 = t
		r = 7
		window = image[x0-r:x0+r+1, y0-r:y0+r+1]

		opt_params[i, 0] = np.mean(window)
		opt_params[i, 1] = t[0]
		opt_params[i, 2] = t[1]
		# print(opt_params[i])

	return opt_params

		# plt.imshow(window)

		# plt.show()

def fit_traps_to_2D_gaussians(image, trap_positions):
	num_params = 4

	opt_params = np.zeros((len(trap_positions), num_params))
	param_errors = np.zeros((len(trap_positions), num_params))

	approximate_amplitude = 100
	approximate_sigma = 3.5 * 2
	approximate_offset = 2

	for index in range(len(trap_positions)):
		trap_position = trap_positions[index]
		approximate_x0, approximate_y0 = trap_position[0], trap_position[1]


		p0=(approximate_amplitude,
		approximate_x0,
		approximate_y0,
		approximate_sigma)

		region_radius = 15;
		bounds = [[int(approximate_x0 - region_radius), int(approximate_x0 + region_radius+1)],
		        [int(approximate_y0 - region_radius), int(approximate_y0 + region_radius+1)]]

		x = np.arange(bounds[0][0], bounds[0][1])
		y = np.arange(bounds[1][0], bounds[1][1])


		if bounds[0][0] < 0 or bounds[0][1] > image.shape[0] or bounds[1][0] < 0 or bounds[1][1] > image.shape[1]:
			print("ERROR: Finding traps that are borderline out of bounds!")
			return []

		# plt.imshow(np.array(image)[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]])
		# plt.show()

		# break

		y,x = np.meshgrid(y,x)


		image_region = image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]]

		flattened_image = np.ndarray.flatten(image_region)

		# fit_bounds = [(10.0, approximate_x0 - region_radius, approximate_y0 - region_radius, 2.0), 
		# 			  (255.0, approximate_x0 + region_radius, approximate_y0 + region_radius, 4.0)]
		
		# Method 'lm' seems to be significantly faster than 'trf' in this case.
		popt, pcov = curve_fit(gaussian_2D_fixed_offset, (y,x), flattened_image, p0=p0, method='lm',
						ftol=1e-8)


		# plt.subplot(211)
		# plt.imshow(image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]],
		# 	vmin=0, vmax=255)
		


		# plt.subplot(212)
		# plt.imshow(gaussian_2D_fixed_offset((y,x), *popt).reshape(*image_region.shape),
		# 	 vmin=0, vmax=255)
		# plt.show()


		# continue
		# print(popt)
		# print(approximate_x0, approximate_y0)
		opt_params[index, :] = popt[:]
		param_errors[index, :] = np.sqrt(np.diag(pcov[:]))
		




	# print(opt_params[-1])
	return opt_params

def fit_traps_to_2D_paraboloids(image, trap_positions):
	num_params = 3

	opt_params = np.zeros((len(trap_positions), num_params))
	param_errors = np.zeros((len(trap_positions), num_params))

	approximate_amplitude = 200
	approximate_curvature = 3.5 # Not sure if this is a good guess

	for index in range(len(trap_positions)):
		trap_position = trap_positions[index]
		approximate_x0, approximate_y0 = trap_position[0], trap_position[1]


		p0=(approximate_amplitude,
		approximate_x0,
		approximate_y0)
		# approximate_curvature)

		region_radius = 1;
		bounds = [[int(approximate_x0 - region_radius), int(approximate_x0 + region_radius+1)],
		        [int(approximate_y0 - region_radius), int(approximate_y0 + region_radius+1)]]

		x = np.arange(bounds[0][0], bounds[0][1])
		y = np.arange(bounds[1][0], bounds[1][1])


		if bounds[0][0] < 0 or bounds[0][1] > image.shape[0] or bounds[1][0] < 0 or bounds[1][1] > image.shape[1]:
			print("ERROR: Finding traps that are borderline out of bounds!")
			return []

		# plt.imshow(np.array(image)[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]])
		# plt.show()
		# break



		y,x = np.meshgrid(y,x)


		image_region = image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]]

		flattened_image = np.ndarray.flatten(image_region)



		# fit_bounds = [(10.0, approximate_x0 - region_radius, approximate_y0 - region_radius, 2.0), 
		# 			  (255.0, approximate_x0 + region_radius, approximate_y0 + region_radius, 4.0)]
		
		# Method 'lm' seems to be significantly faster than 'trf' in this case.
		popt, pcov = curve_fit(paraboloid_2D_fixed_curvature, (y,x), flattened_image, p0=p0, method='lm',
						ftol=1e-8)

		# plt.subplot(211)
		# plt.imshow(image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]],
		# 	vmin=0, vmax=255)
		


		# plt.subplot(212)
		# plt.imshow(paraboloid_2D((y,x), *popt).reshape(3,3),
		# 	 vmin=0, vmax=255)
		# plt.show()


		# continue
		# print(approximate_x0, approximate_y0)
		opt_params[index, :] = popt[:]
		param_errors[index, :] = np.sqrt(np.diag(pcov[:]))
		# break





	# print(opt_params[-1])
	return opt_params


def main():
	app = QtWidgets.QApplication(sys.argv)

	thorCamInterface = ThorCamInterface()

	
	thorCamInterface.show()

	app.exec_()

def main2():
	im = np.load("LastImage.npy")

	t1 = time.process_time()
	traps = ThorCamInterface.identifyTraps(None, im)
	t2 = time.process_time()

	print("Took %.3f sec" %(t2-t1))
	print(len(traps))
	# plt.imshow(im)

	# plt.show()

if __name__ == '__main__':
	main() #DB, changed to main from main2 on 11/5/19
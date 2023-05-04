import sys
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
# tesst 
# from PySide2 import (QtWidgets, QtCore, QtGui)
import pyqtgraph as pg
import numpy as np
import cv2
import time
from threading import Thread
import numexpr
import imageio
import matplotlib.pyplot as plt
from ThorCamInterface import ThorCamInterface
from AODClient import AODClient
import zernike
from skimage.transform import rotate, rescale, resize 
from skimage.io import imread, imshow

CURRENT_TRAP_ARRANGEMENT = "CurrentTrapArrangement.txt"

X_CENTER = 130
Y_CENTER = 83 #60
X_OFFSET_MIN = 5
Y_OFFSET_MIN = 5

MANDATORY_APERTURE = 512

def roundTo(x, base):
        # Round to nearest "base". Make sure to return an integer value
	return int(base) * round(x / base)

# def gaussian_2d(x, y, x0, y0, sigma):
# 	return np.exp(-((x-x0)**2.0 + (y-y0)**2.0)/(2.0*sigma**2.0))

#Note that the Gaussian intensity pattern and the gaussian distribution are defined differently
def gaussian_2d(x, y, x0, y0, sigma):
	return np.exp(-2.0*((x-x0)**2.0 + (y-y0)**2.0)/(sigma**2.0))

	
# This class simply wraps around the pyqtgraph ImageView class
# with the only difference that we hide the default histogram controls.
class SimpleImageView(QtWidgets.QWidget):
	def __init__(self, title):
		super().__init__()

		self.layout = QtWidgets.QVBoxLayout(self) #QtGui.QVBoxLayout(self)

		self.imageView = pg.ImageView()

		self.imageView.ui.roiBtn.hide()
		self.imageView.ui.menuBtn.hide()
		self.imageView.ui.histogram.hide()

		self.imageView.setImage(np.zeros((100, 100)))

		self.imageView.imageItem.setBorder('#bbbbbb') # Gray border around image


		titleLabel = QtWidgets.QLabel(title)
		titleLabel.setAlignment(QtCore.Qt.AlignCenter)

		self.layout.addWidget(titleLabel)
		self.layout.addWidget(self.imageView)

	def setImage(self, *args, **args2):
		self.imageView.setImage(*args, **args2)



class TitleLabel(QtWidgets.QLabel):
	def __init__(self, *args):
		super().__init__(*args)

		self.setAlignment(QtCore.Qt.AlignCenter)
		#self.setFont(QtGui.QFont('Arial', 20))

class ProgressBar(QtWidgets.QWidget):
	def __init__(self):
		super().__init__()

		self.layout = QtWidgets.QVBoxLayout(self) #QtGui.QVBoxLayout(self)

		self.label = QtWidgets.QLabel()
		self.progressBar = QtWidgets.QProgressBar()

		self.layout.addWidget(self.label)
		self.layout.addWidget(self.progressBar)

	def setRange(self, minVal, maxVal):
		self.progressBar.setRange(minVal, maxVal)

	def setValue(self, iteration, variation):
		self.label.setText("Iter: %d\tVariation: %.2f%%" %(iteration, variation))
		self.progressBar.setValue(iteration)


# This class encodes a table view with several
# user programmable settings for how the SLM operates.
# More settings will likely be added over time.
# A dictionary encodes the settings displayed and edited in the table.
class SLMSettings(QtWidgets.QTableWidget):
	NumericalIterationsLabel = "# Numerical Iterations"
	GaussianWidthLabel = "Input Gaussian Width"
	ZeroOrderOffsetXLabel = "Offset from 0th (X)"
	ZeroOrderOffsetYLabel = "Offset from 0th (Y)"

	def __init__(self):
		super().__init__()

		self.settings = {
			self.NumericalIterationsLabel: 20,
			self.GaussianWidthLabel: 0.5,
			self.ZeroOrderOffsetXLabel: 40,
			self.ZeroOrderOffsetYLabel: 40,
		}


		self.setColumnCount(2)
		self.setRowCount(len(self.settings.keys()))

		self.setFont(QtGui.QFont('Arial', 9))

		keys = list(self.settings.keys())
		for i in range(len(keys)):
			self.setItem(i, 0, QtWidgets.QTableWidgetItem(keys[i]))
			self.setItem(i, 1, QtWidgets.QTableWidgetItem("%s" %self.settings[keys[i]]))

			self.item(i, 0).setFlags(QtCore.Qt.ItemIsEnabled)

		header = self.horizontalHeader()
		header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
		header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

		self.itemChanged.connect(self.settingChanged)

	# Update dictionary key-value pair according to edited table item.
	def settingChanged(self, item):
		index = item.row()

		settingName = self.item(index, 0).text()
		newValue = float(self.item(index, 1).text())

		self.settings[settingName] = newValue

	def getNumNumericalIterations(self):
		return int(self.settings[self.NumericalIterationsLabel])

	def getGaussianWidth(self):
		return self.settings[self.GaussianWidthLabel]

	def setZeroOrderOffset(self, xOffset, yOffset):
		keys = list(self.settings.keys())
		for i,k in enumerate(keys):
			if k == self.ZeroOrderOffsetXLabel: index_x = i
			if k == self.ZeroOrderOffsetYLabel: index_y = i

		newXOffset = max(X_CENTER - xOffset, X_OFFSET_MIN)
		newYOffset = max(Y_CENTER - yOffset, Y_OFFSET_MIN)

		self.itemChanged.disconnect()
		self.setItem(index_x, 1, QtWidgets.QTableWidgetItem("%s" %newXOffset))
		self.setItem(index_y, 1, QtWidgets.QTableWidgetItem("%s" %newYOffset))
		self.settings[self.ZeroOrderOffsetXLabel] = newXOffset
		self.settings[self.ZeroOrderOffsetYLabel] = newYOffset
		self.itemChanged.connect(self.settingChanged)

	def getZeroOrderOffset(self):
		return int(self.settings[self.ZeroOrderOffsetXLabel]), int(self.settings[self.ZeroOrderOffsetYLabel])

	def getCurrentSettings(self):
		return self.settings

	def setCurrentSettings(self, settings):
		for k in settings.keys():
			# print("LOADING SETTING:", k, " (val:", settings[k], ")")
			if k in self.settings.keys():
				self.settings[k] = settings[k]


		# Update the displayed table elements
		all_keys = list(self.settings.keys())
		for i in range(len(all_keys)):
			if all_keys[i] in settings.keys():
				print(i, all_keys[i])
				self.item(i, 1).setText("%s" %str(settings[all_keys[i]]))


# This class encodes a table view with coefficients. They are in units of milli-waves.
class SLMZernikeCoefficients(QtWidgets.QTableWidget):
	def __init__(self, updateCallback):
		super().__init__()

		self.polynomial_labels = []
		self.polynomial_indices = []

		self.coefficients = np.array([zernike.ordered_polynomials[i][2] for i in range(len(zernike.ordered_polynomials))])
		self.coefficients = np.append(self.coefficients, 0)#This is for adding the rotation angle Sepehr 9/8/2022
		self.coefficients = np.append(self.coefficients, 0)#This is for adding the Aperture mask
		self.coefficients = np.append(self.coefficients, 1)#This is for zooming into the hologram



		for i in range(len(zernike.ordered_polynomials)):
			self.polynomial_labels.append("%s, %s" %(zernike.ordered_polynomials[i][0], zernike.ordered_polynomials[i][1]))
			self.polynomial_indices.append(zernike.ordered_polynomials[i][0])


		self.setColumnCount(2)
		self.setRowCount(len(zernike.ordered_polynomials)+3)### Must adjust is adding more rows

		self.setFont(QtGui.QFont('Arial', 9))

		keys = self.polynomial_labels
		for i in range(len(keys)):
			self.setItem(i, 0, QtWidgets.QTableWidgetItem(keys[i]))
			self.setItem(i, 1, QtWidgets.QTableWidgetItem("%s" %self.coefficients[i]))

			self.item(i, 0).setFlags(QtCore.Qt.ItemIsEnabled)

		#ADDING A ROTATION ANGLE ITEM HERE#Sepehr 9/8/2022

		self.setItem(i+1, 0, QtWidgets.QTableWidgetItem('Rotation angle'))
		self.setItem(i+1, 1, QtWidgets.QTableWidgetItem("0.0"))
		self.item(i+1, 0).setFlags(QtCore.Qt.ItemIsEnabled)
		##############

		#ADDING A MASK DIAMETER#Sepehr 9/8/2022

		self.setItem(i+2, 0, QtWidgets.QTableWidgetItem('mask radius'))
		self.setItem(i+2, 1, QtWidgets.QTableWidgetItem('512'))
		self.item(i+2, 0).setFlags(QtCore.Qt.ItemIsEnabled)
		##############

		#ADDING A ZOOM OPTION HERE #Sophie 03/16/2023 
		self.setItem(i+3, 0, QtWidgets.QTableWidgetItem('Zoom'))
		self.setItem(i+3, 1, QtWidgets.QTableWidgetItem("1"))
		self.item(i+3, 0).setFlags(QtCore.Qt.ItemIsEnabled)

		############## 
		header = self.horizontalHeader()
		header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
		header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

		self.updateCallback = updateCallback

		self.itemChanged.connect(self.coefficientChanged)

	# Update dictionary key-value pair according to edited table item.
	def coefficientChanged(self, item):
		index = item.row()

		try:
			self.coefficients[index] = float(self.item(index, 1).text())

			self.updateCallback()
		except:
			print("ERROR SETTING FLOATING VALUE.")


	def getCurrentCoefficients(self):
		return self.coefficients

	def getPolynomialLabels(self):
		return self.polynomial_labels ##ADDED 10/13/2022

	def setCurrentCoefficients(self, coefficients):
		self.coefficients = coefficients


# Main interface for handling the SLM.
class SLMController(QtWidgets.QWidget):
	def __init__(self, app, shouldEnableSLMDisplay=True, shouldEnableThorcam=True):
		super().__init__()


		self.app = app

		print("Creating SLM Display...")
		self.display = SLMDisplay(enable = shouldEnableSLMDisplay)
		self.shouldEnableSLMDisplay = shouldEnableSLMDisplay

		print("Done!")


		self.dims = np.array([1024, 1024], dtype=np.int) # Width, height
		self.oversampling_factor = 1 #idk what this would do, but for now set to 1, previously 10, 03-11-2022 #changed to 1 form 10 Sophie 04/21/23
		self.target_arrangement_factorizes = False
		# self.dims = [1024, 1024]
		# self.dims = [256, 256]


		self.targetTrapPositions = []
		self.cameraCornerPositions = []



		# Establish 2d arrays for each field:
		# 1) phaseProfile: phase of each SLM pixel
		# 2) targetIntensityProfile: desired intensity pattern in image plane
		# 3) incidentIntensityProfile: assumed illumination pattern on SLM display
		# 4) incidentAmplitudeProfile: assumed to be sqrt(incidentIntensityProfile)
		# 5) outputAmplitudeProfile: the 2d FFT of the input field
		# 6) outputIntensityProfile: The intensity corresponding to the output field amplitude
		self.phaseProfile = np.zeros(self.dims)
		self.targetIntensityProfile = np.zeros(self.dims)
		self.incidentIntensityProfile = np.zeros(self.dims)
		self.incidentAmplitudeProfile = np.zeros(self.dims)

		self.outputIntensityProfile = np.zeros(self.dims)
		

		
		titleLabel = QtWidgets.QLabel("SLM Controller")
		titleLabel.setAlignment(QtCore.Qt.AlignCenter)
		titleLabel.setFont(QtGui.QFont('Arial', 20))

		# Establish image views that display the current settings for relevant 2d arrays
		self.incidentIntensityView = SimpleImageView("Incident Intensity")
		self.phaseImageView = SimpleImageView("Phase Profile")
		self.simulatedImagePlaneView = SimpleImageView("Simulated Image Plane")
		self.cameraImageView = SimpleImageView("")



		
		self.numericalFeedbackButton = QtWidgets.QPushButton("Feedback on numerics")
		self.numericalFeedbackButton.clicked.connect(self.feedbackToSimulatedField)

		self.numericalFeedbackProgressBar = ProgressBar()

		print("Creating ThorCam interface...")
		self.thorCamInterface = ThorCamInterface(enable = shouldEnableThorcam)
		print("Done!")

		self.calibrateThorCamButton = QtWidgets.QPushButton("Calibrate ThorCam")
		self.calibrateThorCamButton.clicked.connect(self.calibrateThorCam)


		self.feedbackOnCameraButton = QtWidgets.QPushButton("Feedback on ThorCam")
		self.feedbackOnCameraButton.clicked.connect(self.feedbackOnCamera)

		self.feedbackOnFileButton = QtWidgets.QPushButton("Feedback on file")
		self.feedbackOnFileButton.clicked.connect(self.feedbackOnFile)

		self.feedbackOnCameraButton.setEnabled(True) ####SEPEHR#####
		self.feedbackOnFileButton.setEnabled(True)

		self.calibrateSLMCornersButton = QtWidgets.QPushButton("Calibrate SLM Corners")
		self.calibrateSLMCornersButton.clicked.connect(self.calibrateSLMCorners)

		self.calibrateAODButton = QtWidgets.QPushButton("Calibrate AOD")
		self.calibrateAODButton.clicked.connect(self.calibrateAOD)

		self.slmSettings = SLMSettings()
		self.slmZernikeCoefficients = SLMZernikeCoefficients(self.updateSLMDisplay)


		# Establish default target pattern for testing purposes
		for i in range(10):
			for j in range(10):
				self.targetTrapPositions.append([i*5,j*5])


		# What type of incident intensity profile do we want to start with by default?
		self.setGaussianIncidentIntensity(self.slmSettings.getGaussianWidth())
		self.setPhaseProfile(self.phaseProfile)




		# Define layout of graphical interface
		self.layout = QtWidgets.QGridLayout(self)

		self.layout.addWidget(titleLabel, 0, 0, 1, 3)

		#self.layout.addWidget(self.incidentIntensityView, 1, 0)
		self.layout.addWidget(self.phaseImageView, 1, 0)
		self.layout.addWidget(self.simulatedImagePlaneView, 1, 1)

		self.layout.addWidget(self.numericalFeedbackButton, 2, 0)
		self.layout.addWidget(self.numericalFeedbackProgressBar, 2, 1)

		self.layout.addWidget(self.slmSettings, 2, 2, 1, 1)
		self.layout.addWidget(self.slmZernikeCoefficients, 1, 2, 1, 1)



		# self.layout.addWidget(self.cameraImageView, 3, 0, 1, 3)
		self.layout.addWidget(self.calibrateThorCamButton, 3, 0)
		self.layout.addWidget(self.feedbackOnCameraButton, 3, 1)
		self.layout.addWidget(self.feedbackOnFileButton, 3, 2)



		self.layout.addWidget(self.calibrateSLMCornersButton, 4, 0)
		self.layout.addWidget(self.calibrateAODButton, 4, 1)

		self.layout.addWidget(self.thorCamInterface, 5, 0, 1, 3)

		self.localBlazeAmp = 0.2
		self.localBlazeZero = 0.05

		for i in range(3):
			#self.layout.setColumnStretch(i, 1)
			self.layout.setColumnMinimumWidth(i, 250)

		# self.setFixedWidth(850) #On beast
		# self.setFixedHeight(800)

		# self.setFixedWidth(2000) #Changed this to use on laptop
		# self.setFixedHeight(2000)

		self.setFixedWidth(1500) #Changed this to use on monitor
		self.setFixedHeight(1000)

		print("Establishing initial blaze grating.")
		self.setBlazeGrating(0.027, 0.027, apertureSize=512)
		#self.setBlazeGrating(0.05, 0.05, apertureSize=512)
		#self.setBlazeGrating(0.0855, 0.055, apertureSize=512)
		#self.setBlazeGrating(0.125, 0.06, apertureSize=512)

		print("Done!")




	def closeEvent(self, event):
		self.display.close()

	def getCurrentParameters(self):
		settings = self.slmSettings.getCurrentSettings()

		parameters = {}
		parameters['phaseProfile'] = self.phaseProfile
		parameters['targetTrapPositions'] = self.targetTrapPositions
		parameters['outputPhases'] = self.outputPhases

		try:
			parameters['correctionFactors'] = self.correctionFactors
		except:
			parameters['correctionFactors'] = None


		return settings, parameters

	def setCurrentParameters(self, parameters):
		settings, params = parameters

		self.slmSettings.setCurrentSettings(settings)

		self.targetTrapPositions = params['targetTrapPositions']

		zeroOrderOffsetX, zeroOrderOffsetY = self.slmSettings.getZeroOrderOffset()

		self.updateTargetIntensityProfile(zeroOrderOffsetX, zeroOrderOffsetY)


		self.correctionFactors = params['correctionFactors']
		self.outputPhases = params['outputPhases']
		print("Target intensity profile has num targets:", np.sum(self.targetIntensityProfile > 0))

		self.setPhaseProfile(params['phaseProfile'], calculateOutput = False)
		print("Updating SLM display")
		self.updateSLMDisplay()

	def updateZernikeFromString(self, string): ##ADDED 10/13/2022 DB
		try:
			print("Changing Zernike coefficient")
			zernike_polynomial_list = self.slmZernikeCoefficients
			currentCoefficients = self.slmZernikeCoefficients.getCurrentCoefficients() ##ADDED 10/13/2022
			polynomialLabels = self.slmZernikeCoefficients.getPolynomialLabels() ##ADDED 10/13/2022
			lines = string.split("\n")

			for line in lines:
				if len(line.split()) >= 2:
					polynomial = line.split()[-2]
					coefficient = line.split()[-1]
					for ind in range(len(polynomialLabels)):
						runningLabel = polynomialLabels[ind].split(", ")[2]
						if runningLabel.upper().strip() == polynomial.upper().strip():
							print("found match")
							currentCoefficients[ind] = coefficient
							self.slmZernikeCoefficients.setItem(ind, 1, QtWidgets.QTableWidgetItem("%s" %currentCoefficients[ind]))
                
			print(currentCoefficients)
			self.slmZernikeCoefficients.setCurrentCoefficients(currentCoefficients)
			self.updateSLMDisplay()

		except Exception as e:
			print("Error:", e)
	
	def setCalibrationBlazeGrating(self, string): 
		try: 
			print("Setting calibration blaze grating")
			i = int(string.split()[-1])

			blazeAmp = self.localBlazeAmp #used to be 1 #used to be 0.2 
			blazeZero = self.localBlazeZero
			cornerGratings = [
				[blazeZero, blazeZero],
				[blazeZero, blazeAmp],
				[blazeAmp, blazeZero],
				[blazeAmp, blazeAmp]
			]

			norm = (blazeAmp - blazeZero) * self.dims[0]
			offset_from_origin = blazeZero * self.dims[0]
			print(norm)

			print()
			print()
			print("Setting corner: " + str(i))

			self.setBlazeGrating(cornerGratings[i][0], cornerGratings[i][1], apertureSize=10000)

			self.app.processEvents()
			time.sleep(0.1)

		except Exception as e:
			print("Error:", e)
		
	def savePhaseProfile(self):
		np.savez("PhaseProfile.npz",
			phaseProfile=self.phaseProfile,
			targetIntensityProfile=self.targetIntensityProfile,
			correctionFactors = self.correctionFactors,
			outputPhases = self.outputPhases
			)

	def loadPhaseProfile(self):
		try:			
			data = np.load("PhaseProfile.npz")
			loadedPhaseProfile = data["phaseProfile"]
			self.targetIntensityProfile = data["targetIntensityProfile"]
			self.correctionFactors = data["correctionFactors"]
			self.outputPhases = data["outputPhases"]



			self.setPhaseProfile(loadedPhaseProfile)

			self.updateSLMDisplay()
		
		except:
			print("ERROR: Unable to load phase profile.")
			pass


	def feedbackOnFile(self):
		self.thorCamInterface.pauseUpdateThread()

		intensities = np.load("FeedbackFile.npy")

		if len(intensities) != np.count_nonzero(self.targetIntensityProfile):
			print("ERROR: Feedback file has wrong number of elements.")
			self.thorCamInterface.resumeUpdateThread()
			return


		targetMask = self.targetIntensityProfile > 0
		numTargets = np.count_nonzero(targetMask)

		outputAmplitudes = np.sqrt(intensities) # Using measured intensities instead of numerics

		# Take the mean of all pixels where we are supposed to have a trap
		meanAmplitudes = np.mean(outputAmplitudes)

		self.correctionFactors *= meanAmplitudes / outputAmplitudes


		normalizedTargetAmplitudes = self.correctionFactors * self.targetIntensityProfile[targetMask]
		sqrt_num_squared = np.sqrt(np.sum(normalizedTargetAmplitudes)**2.0)
		normalizedTargetAmplitudes /= sqrt_num_squared

		modifiedOutputField = np.zeros(self.dims, dtype=np.complex64)
		modifiedOutputField[targetMask] = normalizedTargetAmplitudes * self.complexExponential(self.outputPhases)


		fftw_output_field = np.fft.ifftshift(modifiedOutputField)
		# correctedInputField = np.fft.fftshift(self.fftw_backward())
		correctedInputField = np.fft.fftshift(np.fft.ifft2(fftw_output_field))


		self.setPhaseProfile(np.angle(correctedInputField))

		self.updateSLMDisplay()

		# phaseProfile_2pi = np.fmod(
		# 		(np.angle(correctedInputField) + (2.0*np.pi)) / (2.0*np.pi),
		# 		1)
		# phaseProfile_bytes = np.array(phaseProfile_2pi * 256, dtype=np.uint8)
		# self.display.setImage(phaseProfile_bytes, self.slmSettings.getPhaseCurvature())



		self.thorCamInterface.resumeUpdateThread()



	def feedbackOnCamera(self):
		self.thorCamInterface.pauseUpdateThread()

		targetMask = self.targetIntensityProfile > 0
		numTargets = np.count_nonzero(targetMask)
		# print("NUM TARGETS:", numTargets)

		# correctionFactors = np.zeros(numTargets)
		# correctionFactors[:] = 1.0

		normalizedTargetAmplitudes = np.zeros(numTargets)
		modifiedOutputField = np.zeros(self.dims, dtype=np.complex64)


		zeroOrderOffset = self.slmSettings.getZeroOrderOffset()
		xPositions = self.targetTrapPositions[:, 0] + zeroOrderOffset[0]
		yPositions = self.targetTrapPositions[:, 1] + zeroOrderOffset[1]


		indexingOrder = self.targetIntensityProfileIndexing[targetMask]
		print("Indexing order:", indexingOrder)

		# TODO: Need to sort these properly according to targetMask
		targetCamPositions = np.array(
			[self.thorCamInterface.convertSLMCoordsToCameraCoords(xPositions[i], yPositions[i]) for i in range(len(xPositions))]
		)


		# method = 0 # Gerchberg-Saxton algorithm
		# method = 1 # Weighted Gerchberg-Saxton
		method = 2 # Weighted Gerchberg-Saxton with phase fixing (Donggyu's idea)
		phaseFixingCutoffIteration = 15

		self.correctionFactors = np.ones(numTargets)## Initializing correction factor, Sepehr 03-11-2022
		for it in np.arange(5):

			traps = self.thorCamInterface.updateImage(ignoreZeroOrder=True)

			if len(traps) != numTargets:
				print("Found wrong number of traps!")
				self.thorCamInterface.resumeUpdateThread()
				return


			# print("Found traps:", traps)

			fittedTrapIntensities = np.zeros(numTargets)
			for t in traps:
				pos = t[:2]

				dist_to_potential_targets = np.sqrt(np.sum((pos - targetCamPositions)**2.0, axis=1))
				closest_match = np.argmin(dist_to_potential_targets)
				fittedTrapIntensities[closest_match] = t[2]

			# fittedTrapAmplitudes = fittedTrapAmplitudes[::-1]
			# print(fittedTrapIntensities)

			print("Trap intensities:", fittedTrapIntensities)
			if 0 in fittedTrapIntensities:
				print("ERROR: Found a trap with 0 intensity!")

				self.thorCamInterface.resumeUpdateThread()
				return


			if method == 2:
				pass # Don't update output phase profile
			else:
				self.outputPhases = np.angle(self.outputAmplitudeProfile[targetMask])

			# outputAmplitudes = np.abs(self.outputAmplitudeProfile[targetMask])
			outputAmplitudes = np.sqrt(fittedTrapIntensities) # Using measured intensities instead of numerics

			# Take the mean of all pixels where we are supposed to have a trap
			meanAmplitudes = np.mean(outputAmplitudes)

			# print("testing")
			# print(self.correctionFactors)
			# print(type(self.correctionFactors))
			# print("done testing")
			# if self.correctionFactors == None:
			# 	self.correctionFactors = np.ones_like(outputAmplitudes)

			self.correctionFactors *= meanAmplitudes / outputAmplitudes *0.2
			# print("CORRECTION FACTORS:", self.correctionFactors)


			print("Correction factors shape:", self.correctionFactors.shape)
			normalizedTargetAmplitudes = self.correctionFactors * self.targetIntensityProfile[targetMask]
			sqrt_num_squared = np.sqrt(np.sum(normalizedTargetAmplitudes)**2.0)
			normalizedTargetAmplitudes /= sqrt_num_squared

			# print("OUTPUT PHASES:", self.outputPhases)
			# print("TARGET MASK:", targetMask)
			print("Modified output field shape:", modifiedOutputField.shape)
			print("Target mask shape:", targetMask.shape)
			print("Normalized target amplitudes shape:", normalizedTargetAmplitudes.shape)
			print("Output phases shape:", self.outputPhases.shape)
			modifiedOutputField[targetMask] = normalizedTargetAmplitudes * self.complexExponential(self.outputPhases[targetMask])


			self.fftw_output_field = np.fft.ifftshift(modifiedOutputField)
			# correctedInputField = np.fft.fftshift(self.fftw_backward())
			correctedInputField = np.fft.fftshift(np.fft.ifft2(self.fftw_output_field))


			self.setPhaseProfile(np.angle(correctedInputField))


			self.updateSLMDisplay()
			time.sleep(0.5)
			# phaseProfile_2pi = np.fmod(
			# 		(np.angle(correctedInputField) + (2.0*np.pi)) / (2.0*np.pi),
			# 		1)
			# phaseProfile_bytes = np.array(phaseProfile_2pi * 256, dtype=np.uint8)
			# self.display.setImage(phaseProfile_bytes, self.slmSettings.getPhaseCurvature())

			
			# simulatedTrapIntensities = self.outputIntensityProfile[targetMask]
			# fractionalVariation = np.std(simulatedTrapIntensities) / np.mean(simulatedTrapIntensities)
			

			# self.numericalFeedbackProgressBar.setValue(it, fractionalVariation*100)


			self.app.processEvents() # Allows image views to redraw


		# t2 = time.time()
		# print("Took %.3f sec to perform feedback." %(t2-t1))



		self.thorCamInterface.resumeUpdateThread()

	def updateSLMDisplay(self):
		# phaseProfile_2pi = np.fmod((self.phaseProfile + (2.0*np.pi)) / (2.0*np.pi), 1)
		# phaseProfile_bytes = np.array(phaseProfile_2pi * 256, dtype=np.uint8)

		self.display.setImage(self.phaseProfile, self.slmZernikeCoefficients.getCurrentCoefficients())

	def calibrateThorCam(self):
		self.thorCamInterface.prepareForCalibration()

		# self.thorCamInterface.setFullAOI()
		blazeAmp = 0.2 #used to be 1 #used to be 0.2 
		blazeZero = 0.05
		cornerGratings = [
			[blazeZero, blazeZero],
			[blazeZero, blazeAmp],
			[blazeAmp, blazeZero],
			[blazeAmp, blazeAmp]
		]

		norm = (blazeAmp - blazeZero) * self.dims[0]
		offset_from_origin = blazeZero * self.dims[0]
		print(norm)


		measuredCornerPositions = []
		for i in range(len(cornerGratings)):
			print()
			print()
			print("Finding corner", i)


			# self.setBlazeGrating(cornerGratings[i][0], cornerGratings[i][1], apertureSize=150)
			self.setBlazeGrating(cornerGratings[i][0], cornerGratings[i][1], apertureSize=10000)

			self.app.processEvents()

			time.sleep(0.4)

			num_tries = 4
			for t in range(num_tries):
				print("Try", t)
				traps = self.thorCamInterface.updateImage(num_traps=1, useGaussian=True)
				if len(traps) == 0:
					print("CALIBRATION ERROR: Unable to find beam!")
					loc = [0, 0]
				else:
					loc = traps[0][:2]
					break
				time.sleep(0.1)
			measuredCornerPositions.append(loc)

			print("Loc:", loc)

			self.app.processEvents()
			time.sleep(0.1)
		measuredCornerPositions = np.array(measuredCornerPositions)

		self.thorCamInterface.doneWithCalibration(measuredCornerPositions, norm, offset_from_origin)
	
	def saveLocalCorners(self, string): 
		local_corners_array = np.zeros((4,2))
		trap_corners_array = np.zeros((4,2))

		local_corners_string = string.split("_")[1]
		trap_corners_string = string.split("_")[2]

		local_coordinates = local_corners_string.split(",")
		trap_coordinates = trap_corners_string.split(",")
		
		for i in range(len(local_coordinates)-1): 
			coordinates_split = local_coordinates[i].split(" ")
			local_corners_array[i][0] = float(coordinates_split[0])
			local_corners_array[i][1] = float(coordinates_split[1]) 

		for i in range(len(trap_coordinates)-1): 
			coordinates_split = trap_coordinates[i].split(" ")
			trap_corners_array[i][0] = float(coordinates_split[0])
			trap_corners_array[i][1] = float(coordinates_split[1]) 

		norm = (self.localBlazeAmp - self.localBlazeZero) * self.dims[0]
		offset_from_origin = self.localBlazeZero * self.dims[0]
		

		print("local coordinates")
		print(local_corners_array)
		print("trap coordinates ")
		print(trap_corners_array)

		self.thorCamInterface.doneWithCalibration(local_corners_array, trap_corners_array, norm,offset_from_origin)


	def saveTrapCorners(string): 
		print(string)

	def calibrateSLMCorners(self):
		self.thorCamInterface.pauseUpdateThread()

		self.thorCamInterface.settings.setExposure(0.1)


		xPositions = np.array(self.targetTrapPositions)[:, 0]
		yPositions = np.array(self.targetTrapPositions)[:, 1]

		zeroOrderX, zeroOrderY = self.slmSettings.getZeroOrderOffset()


		minX = np.min(xPositions) + zeroOrderX
		maxX = np.max(xPositions) + zeroOrderX
		minY = np.min(yPositions) + zeroOrderY
		maxY = np.max(yPositions) + zeroOrderY

		# Right now, we will assume that we are on a rectangular lattice
		# where traps exist at each of the four combinations:
		corners = np.array([
			[maxX, maxY],
			[minX, maxY],
			[maxX, minY],
			[minX, minY]
		])


		self.cameraCornerPositions = []
		for i in range(len(corners)):
			time.sleep(0.2)

			traps = self.thorCamInterface.updateImage(num_traps=1, singleTrapSLMCoord=corners[i])[0]
			print("Found corner %d:" %i, traps)
			self.cameraCornerPositions.append(traps[:2])


			self.app.processEvents()

		self.thorCamInterface.resumeUpdateThread()

	#trapSLMFourCorners = 
	# We want to adjust the local addressing SLM such that it matches the positions of the four corners 
	# def rotation_mat(self, theta): 
	# 	return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	
	# def rotate(self, array, theta): 
	# 	array_rot = np.zeros(np.shape(array))
	# 	for i in range(len(array)): 
	# 		array_rot[i] = np.matmul(self.rotation_mat(theta), array[i])
	# 	return array_rot
		
	# def to_origin_frame(self, array): 
	# 		origin =  np.reshape(np.array([array[0][0], array[0][1]]*4),(4,2))
	# 		return array - origin

	# def generate_trapSLM_corners(self): 
	# 	test_array = [np.array([544.99621485, 321.99843666]), np.array([537.00379237, 128.00004969]), np.array([735.00273043, 316.00252588]), np.array([727.99998694, 121.99667972])]
	# 	test_array = np.ones(np.shape(test_array))*20+test_array
	# 	return test_array
	
	# def get_d_from_coords(self, corners, point):
		xDis = corners[0][0]-corners[point][0]
		yDis = corners[0][1]-corners[point][1]
		d = np.sqrt(xDis**2 + yDis**2)
		return d
	
	
	# def calibrateLA(self): 
	# 	self.calibrateThorCam()
	# 	self.calibrateSLMCorners()
	# 	la_corners = self.cameraCornerPositions
	# 	la_corners = self.to_origin_frame(la_corners)
	# 	trap_corners = self.generate_trapSLM_corners()
	# 	trap_corners = self.to_origin_frame(trap_corners)
	# 	trap_corners = trap_corners * np.reshape(np.array([1/3, 1/2]*4), (4,2)) 
	# 	trap_corners = self.rotate(trap_corners, np.pi/6)
	# 	#trap_corners -= self.origin(trap_corners)

	# 	delta_x = trap_corners[0][0] - la_corners[0][0]
	# 	delta_y = trap_corners[0][1] - la_corners[0][1]
	
	# 	offset = np.reshape(np.array([delta_x, delta_y]*4),(4,2))
		
	# 	la_corners_offset = la_corners + offset

	# 	dtrapx = self.get_d_from_coords(trap_corners, 2)
	# 	dtrapy = self.get_d_from_coords(trap_corners, 1)
	# 	dlax = self.get_d_from_coords(la_corners_offset, 2)
	# 	dlay = self.get_d_from_coords(la_corners_offset, 1) 
	# 	mag = np.reshape(np.array([dtrapx / dlax, dtrapy / dlay]*4), (4,2)) 
		
	# 	la_corners_offset_mag = (la_corners_offset)*mag
	
	# 	a = trap_corners[1]/np.linalg.norm(trap_corners[1])
	# 	b = la_corners_offset_mag[1]/np.linalg.norm(la_corners_offset_mag[1])
	# 	theta = np.arccos(np.dot(a,b))

	# 	la_corners_offset_mag_rot = self.rotate(la_corners_offset_mag, theta)

	# 	for i in range(len(la_corners)): 
	# 		x = la_corners[i][0]
	# 		y = la_corners[i][1]
	# 		plt.scatter(x, y, color= "g", alpha= 0.5)
	# 		plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=12)
	# 	for i in range(len(trap_corners)): 
	# 		x = trap_corners[i][0]
	# 		y = trap_corners[i][1]
	# 		plt.scatter(x, y, color= "r", alpha= 0.5 )
	# 		plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=12)
	# 	for i in range(len(la_corners_offset)): 
	# 		x = la_corners_offset[i][0]
	# 		y = la_corners_offset[i][1]
	# 		plt.scatter(x, y, color= "b", alpha=0.5)
	# 		plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=12)
	# 	for i in range(len(la_corners_offset_mag)): 
	# 		x = la_corners_offset_mag[i][0]
	# 		y = la_corners_offset_mag[i][1]
	# 		plt.scatter(x, y, color= "y",  alpha= 0.5)
	# 		plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=12)
	# 	for i in range(len(la_corners_offset_mag_rot)): 
	# 		x = la_corners_offset_mag_rot[i][0]
	# 		y = la_corners_offset_mag_rot[i][1]
	# 		plt.scatter(x, y, color= "k",  alpha= 0.5)
	# 		plt.text(x * (1 + 0.01), y * (1 + 0.01) , i, fontsize=12)
	# 	plt.axis('scaled')
	# 	plt.show() 

	# 	return [offset, mag, theta]
	
	# We want to find AOD frequencies that match the positions of the four corners
	def calibrateAOD(self):

		if len(self.cameraCornerPositions) != 4:
			print("ERROR: Must calibrate SLM corners first.")
			return

		aodClient = AODClient() # Connects to AWGController_DualChannel
		if not aodClient.connected:
			print("ERROR: Unable to connect with AOD controller.")
			return


		# self.cameraCornerPositions = [[379, 939]] # First coord is Y, second coord is X

		self.thorCamInterface.pauseUpdateThread()
		self.thorCamInterface.settings.setExposure(0.3)

		amp = 0.9 # Amplitude of AOD trap in arbitrary units.
		# amp = 3.0
		
		if False: #True for imaging source, false for thorlabs
			tolerance = 0.1 # in units of pixels on the camera

			# Conversion from AOD freq to pixels:
			pixels_per_MHz = (635. - 498.) / 5.0 #Imaging source 08-11-2022
			#OLD NUMBER(933.5 - 695.7) / 10.0
			
			# pixels_per_MHz *= 1.1 #OBSOLETE
			#pixels_per_MHz *= 1.3 # for ImagingSource OBSOLETE
		else:
			#amp = 3.0
			tolerance = 0.1 # in units of pixels on the camera

			# Conversion from AOD freq to pixels:
			pixels_per_MHz = (933.5 - 695.7) / 10.0
			pixels_per_MHz *= 1.1

		identifiedAODFrequencies = []

		if aodClient.connected:

			for i in range(len(self.cameraCornerPositions)):
				targetY, targetX = self.cameraCornerPositions[i]

				# Adjust brightness for different coordinates due to inhomogeneous transmission through mirror
				#amp = [1.0, .7, 1.0, .7][i] DB commented out 12/10/2022
				amp = [3, 3, 3, 3][i] #DB added 12/10/2022

				xFreq = 98.0
				yFreq = 98.0

				unable_to_converge = False

				while True:
					aodClient.setStaticWaveform(xFreq, yFreq, amp, amp)

					
					time.sleep(0.2)

					traps = self.thorCamInterface.updateImage(num_traps=1, useGaussian=True)
					self.app.processEvents()

					if len(traps) == 0:
						print("CALIBRATION ERROR: Unable to find beam!")
						unable_to_converge = True
						break
					else:
						currentY, currentX = traps[0][:2]

					if (xFreq < 70) or (xFreq > 130) or (yFreq < 70) or (yFreq > 130):
						print("Beams didn't converge properly")
						unable_to_converge = True
						break

					print("Current pos:", currentX, currentY)
					print("\t Relative to target pos:", targetX, targetY)

					xIsGood = False
					yIsGood = False

					if np.abs(currentX - targetX) > tolerance:
						xFreq += (targetX - currentX) / pixels_per_MHz
					else:
						xIsGood = True


					if np.abs(currentY - targetY) > tolerance:
						yFreq -= (targetY - currentY) / pixels_per_MHz
					else:
						yIsGood = True

					if xIsGood and yIsGood:
						identifiedAODFrequencies.append([xFreq, yFreq])
						# print("%d. FOUND AOD FREQS:" %i, xFreq, yFreq)
						break

				if unable_to_converge:
					break

			if len(identifiedAODFrequencies) == 4:
				for i in range(4):
					print("Corner %d: " %i, identifiedAODFrequencies[i])
				print()
				lowerXFreq = 0.5*(identifiedAODFrequencies[0][0] + identifiedAODFrequencies[2][0])
				upperXFreq = 0.5*(identifiedAODFrequencies[1][0] + identifiedAODFrequencies[3][0])
				lowerYFreq = 0.5*(identifiedAODFrequencies[2][1] + identifiedAODFrequencies[3][1])
				upperYFreq = 0.5*(identifiedAODFrequencies[0][1] + identifiedAODFrequencies[1][1])

				print("X freqs:", lowerXFreq, upperXFreq)
				print("Y freqs:", lowerYFreq, upperYFreq)
				print()

				print("Lower X diff:", identifiedAODFrequencies[0][0] - identifiedAODFrequencies[2][0])
				print("Upper X diff:", identifiedAODFrequencies[1][0] - identifiedAODFrequencies[3][0])
				print("Lower Y diff:", identifiedAODFrequencies[2][1] - identifiedAODFrequencies[3][1])
				print("Upper Y diff:", identifiedAODFrequencies[0][1] - identifiedAODFrequencies[1][1])
				print()

				#IF SAVE TO FILE: IMPLEMENT HERE TO SAVE TO REARRANGEMENTSETTINGS.TXT

				if True: #DB ADDED 11/12/2022
					rearrangement_settings_file = '../TrapController2D/RearrangementSettings.txt'
					with open(rearrangement_settings_file, 'r') as file:
					    # read a list of lines into data
					    filedata = file.readlines()

					lineindex = 0
					for line in filedata:
						keywords = ["Y_MIN_AOD_FREQ", "Y_MAX_AOD_FREQ", "X_MAX_AOD_FREQ", "X_MIN_AOD_FREQ"]
						values = [lowerYFreq, upperYFreq, upperXFreq, lowerXFreq]
						wordindex = 0
						for word in keywords:
							if line.startswith(word):
								filedata[lineindex] = f"{word} {values[wordindex]:.3f} # in MHz \n"
								# word + " " + str(values[wordindex]) + " # in MHz \n"
							wordindex += 1
						lineindex += 1

					# and write everything back
					with open(rearrangement_settings_file, 'w') as file:
					    file.writelines(filedata)

			else:
				print("Unable to find all corners!")



			aodClient.setStaticWaveform(98, 98, 0, 0)


		self.thorCamInterface.resumeUpdateThread()




	def setBlazeGrating(self, xBlaze, yBlaze, apertureSize):
		phases = np.zeros(self.dims)
		xs = np.arange(self.dims[0])
		ys = np.arange(self.dims[1])

		phases[:, :] += ys * yBlaze * 2.0*np.pi
		phases[:, :] = (phases.T + xs * xBlaze * 2.0*np.pi).T

		phases = np.mod(phases + np.pi, 2.0*np.pi) - np.pi

		x0 = self.dims[0] // 2
		y0 = self.dims[0] // 2
		for x in xs:
			phases[x, ys] *= (x0 - x)**2.0 + (y0-ys)**2.0 <= apertureSize**2.0


		self.setPhaseProfile(phases)
		self.updateSLMDisplay()

		# phaseProfile_bytes = np.array(np.modf(phases/(2.0*np.pi))[0] * 256, dtype=np.uint8)
		# self.display.setImage(phaseProfile_bytes, self.slmSettings.getPhaseCurvature())
		# phaseProfile_bytes = np.array(phaseProfile_2pi * 256, dtype=np.uint8)
		# self.display.setImage(phaseProfile_bytes)

	def setTargetTrapPositions(self, trapPositions, oversamplingFactor=1):
		self.oversampling_factor = oversamplingFactor			
		self.targetTrapPositions = np.array(trapPositions)

		# Does this target arrangement factorize fully along X and Y coordinates?
		# If so, we can dramatically speed up WGS by solving it separately in 1D along X and Y.
		unique_x_coords = np.unique(self.targetTrapPositions[:, 0])
		unique_y_coords = np.unique(self.targetTrapPositions[:, 1])

		# Criterion for factorizability:
		# All x coordinates that are present should pair with all y coordinates that are present
		# That is, we should have [x,y] in targetTrapPositions for each x in unique_x_coords and y in unique_y_coords
		# There are precisely |unique_x_coords| * |unique_y_coords| such pairs [x,y]
		# There's no way that |unique_x_coords| * |unique_y_coords| > |targetTrapPositions|
		# So the condition for factorizability is |unique_x_coords| * |unique_y_coords| = |targetTrapPositions|
		if len(unique_x_coords) * len(unique_y_coords) == len(self.targetTrapPositions):
			self.target_arrangement_factorizes = True
		else:
			self.target_arrangement_factorizes = False
		print("Target arrangement factorizes:", self.target_arrangement_factorizes)


	def getDeflectionEfficiency(self, x, y, shouldCorrectDeflectionEfficiency=True): # In SLM diffraction units from origin
		if shouldCorrectDeflectionEfficiency:
			r = np.sqrt(x**2.0 + y**2.0)
			angle_scaling_1 = 0.85 #rescaling diffraction angle used to be called fudge factor
			angle_scaling_2 = 0.75 #rescaling diffraction angle
			max_diffraction_unit_capping = 1000 #to prevent overcompensation at large distance
			# Assume efficiency goes as sinc of radial distance:
			if r < max_diffraction_unit_capping:
				efficiency = (np.sinc(r/(self.dims[0]/2)*angle_scaling_1))**2.0

			else:
				efficiency = (np.sinc(r/(self.dims[0]/2)*angle_scaling_2))**2.0

			# Assume efficiency goes as the product of sinc for x coordinate and y coordinate
			 #This is to correct for the scaling
			# efficiency = (np.sinc(x/(self.dims[0]/2) *fudge_factor))**2.0 * (np.sinc(y/(self.dims[0]/2)*fudge_factor))**2.0
		else:
			efficiency = 1.0

		return efficiency


	def updateTargetIntensityProfile(self, xOffset, yOffset):
		self.targetIntensityProfile = np.zeros(np.array(self.dims) * self.oversampling_factor)

		self.targetIntensityProfileIndexing = np.zeros(self.targetIntensityProfile.shape, dtype=np.int32)


		zeroOrderX = int(self.dims[0]/2)
		zeroOrderY = int(self.dims[1]/2)

		slmTrapPositions = np.zeros((len(self.targetTrapPositions), 2), dtype=np.float)
		slmTrapPositions[:, 0] = self.targetTrapPositions[:, 0] + xOffset
		slmTrapPositions[:, 1] = self.targetTrapPositions[:, 1] + yOffset


		coords = []

		for i in range(len(self.targetTrapPositions)):
			t = self.targetTrapPositions[i]

			efficiency = self.getDeflectionEfficiency(t[0] + xOffset, t[1] + yOffset)

			targetCoordX = np.round((zeroOrderX + xOffset + t[0]) * self.oversampling_factor).astype(np.int)
			targetCoordY = np.round((zeroOrderY + yOffset + t[1]) * self.oversampling_factor).astype(np.int)

			#print(slmTrapPositions[i])

			coords.append([xOffset + t[0], yOffset + t[1]])

			if targetCoordX < 0 or targetCoordX >= self.dims[0] * self.oversampling_factor:
				print("Error: Target coordinates out of bounds.")
				continue
			if targetCoordY < 0 or targetCoordY >= self.dims[1] * self.oversampling_factor:
				print("Error: Target coordinates out of bounds.")
				continue


			self.targetIntensityProfile[targetCoordX, targetCoordY] = 1.0 / efficiency
			self.targetIntensityProfileIndexing[targetCoordX, targetCoordY] = i


		# Save current trap positions to disk
		if self.shouldEnableSLMDisplay:
			# np.savetxt("C:\\Users\\Public\\TrapArrangement\\CurrentTrapArrangement.txt",
			# 	slmTrapPositions, fmt="%.2f")
			# np.savetxt("A:\\SharedProgramFiles\\Beast\\TrapArrangement\\CurrentTrapArrangement.txt",
			# 	slmTrapPositions, fmt="%.2f") ####SAVE TO A: DRIVE, DOLEV ADDED 7/21/2022
			pass # DISABLED on 02/22/23 

		self.thorCamInterface.markSLMCoordinates(coords)



	# Update image views according to current instance variables
	# which represent the phase profile and output intensity profile
	def refreshImageViews(self):
		self.phaseImageView.setImage(self.phaseProfile, levels=(-np.pi, np.pi))
		self.simulatedImagePlaneView.setImage(self.outputIntensityProfile)


	# Update parameters according to settings table.
	def retrieveSettings(self):
		self.setGaussianIncidentIntensity(self.slmSettings.getGaussianWidth())

		zeroOrderOffsetX, zeroOrderOffsetY = self.slmSettings.getZeroOrderOffset()
		self.updateTargetIntensityProfile(zeroOrderOffsetX, zeroOrderOffsetY)

		self.numIter = self.slmSettings.getNumNumericalIterations()


	# Simply calculates exp(I * x), where x is an array of values.
	def complexExponential(self, array):
		# method = "numpy" # Standard np.exp()
		method = "numexpr" # Seems to be > 2x faster

		if method == "numpy":
			return np.exp(1.0j * array)
		elif method == "numexpr":
			return numexpr.evaluate('exp(1j * array)')

	# Update the simulated phase profile on the SLM
	# and calculate the new output field
	def setPhaseProfile(self, phases, calculateOutput=True):
		if phases.shape != self.phaseProfile.shape:
			print("Unable to set phase profile due to shape mismatch.")
			return
		self.phaseProfile = phases

		if calculateOutput:
			# TODO: APERTURE MASK

			incidentField = self.incidentAmplitudeProfile * self.complexExponential(self.phaseProfile)

			padded = np.zeros((np.array(self.dims) * self.oversampling_factor), dtype=np.complex64)
			padded[self.dims[0] * self.oversampling_factor//2  - self.dims[0]//2:self.dims[0] * self.oversampling_factor//2  + self.dims[0]//2,
				   self.dims[1] * self.oversampling_factor//2  - self.dims[1]//2:self.dims[1] * self.oversampling_factor//2  + self.dims[1]//2] = incidentField

			
			self.fftw_input_field = np.fft.fftshift(incidentField)
			#padded[:self.dims[0], :self.dims[1]] = self.fftw_input_field

			inp = np.fft.fftshift(padded)
			out = np.fft.fft2(inp)
			self.outputAmplitudeProfile = np.fft.ifftshift(out)


			self.outputIntensityProfile = np.abs(self.outputAmplitudeProfile)**2.0


		self.refreshImageViews()


	def feedbackToSimulatedField(self):
		self.thorCamInterface.pauseUpdateThread() # Stop processing images on the camera while we do this calculation

		t1 = time.time()
		self.retrieveSettings()


		if False and self.target_arrangement_factorizes:
			max_num_attempts = 10
			num_iterations_1d = 500
			on_the_right_track_iteration = 50
			on_the_right_track_threshold = 1e-2
			successful_threshold = 1e-5




			unique_x_coords = np.unique(self.targetTrapPositions[:, 0])
			unique_y_coords = np.unique(self.targetTrapPositions[:, 1])

			zeroOrderOffsetX, zeroOrderOffsetY = self.slmSettings.getZeroOrderOffset()

			input_field_along_axes = np.zeros((2, self.dims[0]), dtype=np.complex64) # Assuming dims[0] == dims[1]
			axes_were_successful = np.zeros(2, dtype=np.bool)

			print("Solving problem in separate coordinates...")
			for axis in range(2): # X and Y axis calculations
				print("Coordinate", axis)
				sys.stdout.flush()
				self.numericalFeedbackProgressBar.label.setText("Calculating for axis %d" %axis)
				self.app.processEvents() # Allows image views to redraw

				targetOutputIntensities = np.zeros(self.dims[axis] * self.oversampling_factor)
				unique_coords = [unique_x_coords, unique_y_coords][axis]
				offset_from_zero_order = [zeroOrderOffsetX, zeroOrderOffsetY][axis]
				zeroOrderPosition = self.dims[axis] // 2

				for coord in unique_coords:
					index_in_oversampled_output = np.round((zeroOrderPosition + offset_from_zero_order + coord) * self.oversampling_factor).astype(np.int)
					
					if len(unique_coords) > 2:
						deflectionEfficiency = self.getDeflectionEfficiency(coord, 0)

						targetOutputIntensities[index_in_oversampled_output] = 1.0 / deflectionEfficiency
					else:
						targetOutputIntensities[index_in_oversampled_output] = 1.0


				targetMask = np.abs(targetOutputIntensities) > 0

				padded_input_field_1d = np.zeros((self.dims[axis] * self.oversampling_factor), dtype=np.complex64)
				pixel_indices = np.arange(self.dims[axis] * self.oversampling_factor)
				relevant_pixels = (pixel_indices) >= (self.dims[axis] * self.oversampling_factor // 2) - self.dims[axis]//2
				relevant_pixels *= (pixel_indices) < (self.dims[axis] * self.oversampling_factor // 2) + self.dims[axis]//2

				padded_input_field_1d[relevant_pixels] = 1.0


				for attempt in range(max_num_attempts):
					print("Axis %d: Attempt %d" %(axis, attempt))
					target_output_field = np.zeros(self.dims[axis] * self.oversampling_factor, dtype=np.complex64)
					target_output_field[targetMask] = 1.0

					# Assign random initial phases:
					target_output_field[targetMask] *= np.exp(1.0j * (2.0*np.pi) * np.random.rand(np.sum(targetMask)))


					frac_stds = []
					for iteration in range(num_iterations_1d):
						if iteration == on_the_right_track_iteration:
							if frac_stds[-1] > on_the_right_track_threshold:
								break

						shifted_target_output_field = np.fft.ifftshift(target_output_field)
						shifted_ideal_input_field = np.fft.ifft(shifted_target_output_field, norm='ortho')
						ideal_input_field = np.fft.fftshift(shifted_ideal_input_field)

						#padded_input_field_1d[relevant_pixels] = np.exp(1.0j * np.angle(ideal_input_field[relevant_pixels]))
						padded_input_field_1d = np.exp(1.0j * np.angle(ideal_input_field))
						shifted_input = np.fft.ifftshift(padded_input_field_1d)
						shifted_output = np.fft.fft(shifted_input, norm='ortho')
						output = np.fft.fftshift(shifted_output)


						relative_intensities = np.abs(output[targetMask])**2.0 / targetOutputIntensities[targetMask]
						frac_std = np.std(relative_intensities) / np.mean(relative_intensities)
						frac_stds.append(frac_std)

						if frac_std < successful_threshold:
							break

						# Replace target phases with most recent output of FFT.
						target_output_field[targetMask] = np.abs(target_output_field[targetMask])
						target_output_field[targetMask] *= np.exp(1.0j * np.angle(output[targetMask]))
						target_output_field[targetMask] *= (np.mean(relative_intensities) / relative_intensities)**0.5
					

					if frac_stds[-1] < successful_threshold:
						break


					# This correction should technically be a sqrt (exponent 0.5), but slightly undercorrecting
					# seems to make things more stable, so we are using 0.45.
				if frac_stds[-1] < successful_threshold:
					axes_were_successful[axis] = True

				input_field_along_axes[axis] = padded_input_field_1d[relevant_pixels]

				if False: # To view the evolution of the std.dev over iterations:
					plt.subplot(2,2, 2*axis+1)
					plt.plot(frac_stds)
					plt.axhline(0.01)
					plt.yscale('log')

					plt.subplot(2,2, 2*axis+2)
					plt.plot(np.abs(output)**2.0)
			#plt.show()

			# At this point, we have solved the 1D problem along both axes. The ideal phase profile is therefore the outer
			# product of these two solutions.
			if axes_were_successful[0] and axes_were_successful[1]:
				self.numericalFeedbackProgressBar.label.setText("Done with 1D calculations!")
			else:
				self.numericalFeedbackProgressBar.label.setText("ERROR: 1D calculations failed to converge! (Run again!)")

			self.app.processEvents() # Allows image views to redraw

			print("Done with 1D problem!")
			sys.stdout.flush()

			print("Computing outer product of solutions...")
			input_field_2d = np.outer(input_field_along_axes[0], input_field_along_axes[1])
			print("Done!")

			self.setPhaseProfile(np.angle(input_field_2d), calculateOutput = False)


		else:
			self.numericalFeedbackProgressBar.setValue(0, 0)
			self.numericalFeedbackProgressBar.setRange(0, self.numIter)


			# Initial guess for phase profile
			# self.setPhaseProfile(np.random.rand(*self.dims))
			self.setPhaseProfile(np.zeros(self.dims))

			targetMask = self.targetIntensityProfile > 0
			numTargets = np.count_nonzero(targetMask)

			print("NUM TARGETS:", numTargets)

			targetOutputField = np.zeros(np.array(self.dims) * self.oversampling_factor, dtype=np.complex64)
			targetOutputField[targetMask] = 1.0

			targetTrapAmplitudes = np.zeros(numTargets, dtype=np.complex64)

			# method = 0 # Gerchberg-Saxton algorithm
			# method = 1 # Weighted Gerchberg-Saxton
			method = 2 # Weighted Gerchberg-Saxton with phase fixing (Donggyu's idea)
			phaseFixingCutoffIteration = 30 #2023-04-25. changed from 15 to 30. Seems to allow for better convergence. Sepehr

			for it in (1 + np.arange(self.numIter)):
				fieldAtTargetPositions = self.outputAmplitudeProfile[targetMask]


				if method == 2 and it >= phaseFixingCutoffIteration:
					pass # Don't update output phase profile
				else:
					targetOutputField[targetMask] = np.abs(targetOutputField[targetMask])
					targetOutputField[targetMask] *= np.exp(1.0j * np.angle(fieldAtTargetPositions))

				relativeIntensities = np.abs(fieldAtTargetPositions)**2.0 / self.targetIntensityProfile[targetMask]
				targetOutputField[targetMask] *= np.sqrt(np.mean(relativeIntensities) / relativeIntensities)

				self.outputPhases = np.angle(targetOutputField)

				shiftedTargetOutputField = np.fft.ifftshift(targetOutputField)
				out = np.fft.ifft2(shiftedTargetOutputField)
				correctedInputField = np.fft.fftshift(out)

				# Remove padding at border. Only the center of this field corresponds to the actual SLM.
				correctedInputField = correctedInputField[self.dims[0] * self.oversampling_factor//2  - self.dims[0]//2:self.dims[0] * self.oversampling_factor//2  + self.dims[0]//2,
														  self.dims[1] * self.oversampling_factor//2  - self.dims[1]//2:self.dims[1] * self.oversampling_factor//2  + self.dims[1]//2]


				self.setPhaseProfile(np.angle(correctedInputField))
				
				simulatedTrapIntensities = self.outputIntensityProfile[targetMask]
				fractionalVariation = np.std(relativeIntensities) / np.mean(relativeIntensities)
				

				self.numericalFeedbackProgressBar.setValue(it, fractionalVariation*100)


				self.app.processEvents() # Allows image views to redraw


		self.updateSLMDisplay()


		t2 = time.time()
		print("Took %.3f sec to perform feedback." %(t2-t1))



		self.thorCamInterface.resumeUpdateThread() # Resume processing images on the camera now that we are done with this calibration


	def setUniformIncidentIntensity(self):
		self.incidentIntensityProfile[:, :] = 1.0

		self.updatedIncidentIntensity()

	# Gaussian beam of waist 'size', where 'size' is in units of the SLM display size
	def setGaussianIncidentIntensity(self, size):
		# self.setUniformIncidentIntensity()
		# TODO: GO BACK TO GAUSSIAN INTENSITY PROFILE
		# return

		xs = np.arange(self.dims[1])
		ys = np.arange(self.dims[0])

		xx, yy = np.meshgrid(xs, ys)

		self.incidentIntensityProfile = gaussian_2d(xx, yy, self.dims[0]/2, self.dims[0]/2, size * self.dims[0])

		# plt.imshow(self.incidentIntensityProfile)
		# plt.show()
		self.updatedIncidentIntensity()



	def updatedIncidentIntensity(self):
		self.incidentAmplitudeProfile = np.sqrt(self.incidentIntensityProfile)
		self.incidentIntensityView.setImage(self.incidentIntensityProfile, levels=(0, 1))


class SLMDisplay(QtWidgets.QLabel): #(QtGui.QLabel):
	def __init__(self, enable=False, displayOnSLM=True):
		super().__init__()

		self.enable = enable

		if not enable:
			return

		geometryMain = QtWidgets.QDesktopWidget().screenGeometry(0)
		geometryScreen1 = QtWidgets.QDesktopWidget().screenGeometry(1)
		geometryScreen2 = QtWidgets.QDesktopWidget().screenGeometry(2)
		geometrySLM = QtWidgets.QDesktopWidget().screenGeometry(3)
		numberOfScreens =  QtWidgets.QDesktopWidget().screenCount()
		print("Main geometry:", geometryMain)
		print("SLM Geometry:", geometrySLM)
		print("Screen 1:", geometryScreen1)
		print("Screen 2:", geometryScreen2)
		print("Number of screens:", numberOfScreens)

		self.dims = [1024, 1024]
		self.xs = (np.arange(self.dims[0]) - self.dims[0]//2) / (self.dims[0]//2)
		self.ys = (np.arange(self.dims[1]) - self.dims[1]//2) / (self.dims[1]//2)

		xx, yy = np.meshgrid(self.xs, self.ys)
		self.rhos = np.sqrt(xx**2.0 + yy**2.0)
		self.phis = np.arctan2(yy, xx)

		self.zernikePolynomials = np.zeros((len(zernike.ordered_polynomials), self.dims[0], self.dims[1]))
		for i in range(len(zernike.ordered_polynomials)):
			n, m = zernike.ordered_polynomials[i][0]
			self.zernikePolynomials[i, :, :] = zernike.zernike(self.rhos, self.phis, m, n)


		if displayOnSLM:
			self.width = geometrySLM.width()
			self.height = geometrySLM.height()
			print([geometryMain.width(), 0, self.width, self.height])
			self.setGeometry(geometryMain.width()+self.width, 0, self.width, self.height)
		else: # Testing purposes, dipslay on main screen
			self.width = geometryMain.width()
			self.height = geometryMain.height()

			self.setGeometry(0, 0, self.width, self.height)

		
		# The image to display is a rectangular grid with 8-bit grayscale
		# phase values.
		self.finalOutputPhaseProfile_radians = np.zeros((self.height, self.width), dtype=np.float)
		self.finalOutputPhaseProfile_8bit = np.zeros((self.height, self.width), dtype=np.uint8)
		self.loadHardwareCalibration()

		print("Setting image")
		self.setImage(np.zeros((self.width, self.height)))
		print("Done!")

		# Display in full screen to avoid borders, toolbars, etc.
		# Tell Windows to keep this window on top of others
		# (To avoid extra windows being dragged onto the SLM display).
		self.showFullScreen()
		self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)


		self.show()


	def loadHardwareCalibration(self):
		# Hamamatsu provides a phase profile to correct for
		# the calibrated curvature of the SLM.
		# We just add it to whatever phase profile we want to
		# display.
		curvature = imageio.imread('CAL_LSH0802922_790nm.bmp')
		self.phaseCorrection_radians = curvature.astype(np.float) / 141 * 2.0*np.pi
		# self.V_2pi = 213 # From technical spec documents
		self.V_2pi = 141

	# phaseProfile encodes phase at each pixel in radians, represented in floating points.
	def setImage(self, phaseProfile_arg, zernikeCoefficients=None):
		if not self.enable:
			return

		phaseProfile = np.copy(phaseProfile_arg).T

		#print("Phase profile:", phaseProfile)


		if type(zernikeCoefficients) != type(None):
			print("Adding Zernike polynomials...")
			for i in range(len(zernike.ordered_polynomials)):
				n, m = zernike.ordered_polynomials[i][0]

				zernike_poly_radians = 2.0*np.pi * (0.001 * zernikeCoefficients[i]) * self.zernikePolynomials[i]
				phaseProfile += zernike_poly_radians

			print("Done!")


		# Initialize a new output array starting with the curvature correction,
		# and adding the target phase pattern on top.
		self.finalOutputPhaseProfile_radians[:, :] = 0
		self.finalOutputPhaseProfile_radians[:self.phaseCorrection_radians.shape[0],
											 :self.phaseCorrection_radians.shape[1]] = self.phaseCorrection_radians[:, :]


		if phaseProfile.shape[1] < self.finalOutputPhaseProfile_radians.shape[1]:
			padding = self.finalOutputPhaseProfile_radians.shape[1] - phaseProfile.shape[1]

			self.finalOutputPhaseProfile_radians[:, padding//2:-padding//2] += phaseProfile
		else:
			self.finalOutputPhaseProfile_radians[:phaseProfile.shape[0], :phaseProfile.shape[1]] += phaseProfile


		# Ensure the phases are non-negative by subtracting min value. Then take the phases modulo 2pi.
		self.finalOutputPhaseProfile_radians -= np.min(self.finalOutputPhaseProfile_radians)
		self.finalOutputPhaseProfile_radians += 2.0*np.pi # (To completely ensure non-negative phases)
		self.finalOutputPhaseProfile_radians = np.fmod(self.finalOutputPhaseProfile_radians, 2.0*np.pi)

		#print("Final output phase profile:", self.finalOutputPhaseProfile_radians)


		#Rotating hologram here!! Sepehr 9/8/2022
		temp = self.finalOutputPhaseProfile_radians
		if type(zernikeCoefficients) != type(None):
			self.finalOutputPhaseProfile_radians = rotate(temp, zernikeCoefficients[-3])#The -2nd order to be the rotation

		#Zooming hologram here!! Sophie 3/16/2023 
		temp = self.finalOutputPhaseProfile_radians
		if type(zernikeCoefficients) != type(None): 
			factor = zernikeCoefficients[-1]
			resized = np.zeros_like(temp)
			zoomed = cv2.resize(temp, None, fx=factor, fy=factor)
			
			h, w = temp.shape
			zh, zw = zoomed.shape
			
			if factor == 1: 
				resized = zoomed
			elif factor<1:    # zero padded
				# resized[int((h-zh)/2):int(-(h-zh)/2), int((w-zw)/2):int(-(w-zw)/2)] = zoomed
				resized = zoomed
			else:               # clip out
				# resized = zoomed[int((zh-h)/2):int(-(zh-h)/2), int((zw-w)/2):int(-(zw-w)/2)]
				resized = zoomed[:h,:w]

			# print("rescaled phases: " + str(resized.shape))
			# plt.subplot(131), imshow(temp)
			# plt.title('Original Image')
			# plt.subplot(132), imshow(resized)
			# plt.title('Resized Image')
			# plt.show()
			#sqrt_num_squared = np.sqrt(np.sum(resized)**2.0)
			# resized /= sqrt_num_squared
			# resized = resized / np.max(resized) * 2*np.pi
			temp-= np.min(temp)
			temp += 2.0*np.pi # (To completely ensure non-negative phases)
			temp = np.fmod(temp, 2.0*np.pi)
			self.finalOutputPhaseProfile_radians = resized
		
		#Aperturing hologram!! Sepehr 9/8/2022
		temp = self.finalOutputPhaseProfile_radians
		
		if type(zernikeCoefficients) != type(None):
			if zernikeCoefficients[-2]>3:
				print("APERTURED")
				xs = np.arange(temp.shape[0])
				ys = np.arange(temp.shape[1])

				x0 = temp.shape[0]//2
				y0 = temp.shape[1]//2
				aperture_mask = np.ones_like(temp)
				for x in xs:
					aperture_mask[x, ys] *= (x0 - x)**2.0 + (y0-ys)**2.0 <= zernikeCoefficients[-2]**2.0#The -1st is the aperture size in pixels

				temp*= aperture_mask
				self.finalOutputPhaseProfile_radians = temp


		#ALWAYS HAVING 512 SLM Radius #Added by Sepehr 03-11-2022
		#This is to ensure we know where the aperture is and we don't burn things
		temp = self.finalOutputPhaseProfile_radians

		print("Mandatory Aperture of %i imposed"%MANDATORY_APERTURE)
		xs = np.arange(temp.shape[0])
		ys = np.arange(temp.shape[1])

		x0 = temp.shape[0]//2
		y0 = temp.shape[1]//2
		aperture_mask = np.ones_like(temp)
		for x in xs:
			aperture_mask[x, ys] *= (x0 - x)**2.0 + (y0-ys)**2.0 <= MANDATORY_APERTURE**2.0

		temp*= aperture_mask
		self.finalOutputPhaseProfile_radians = temp

		############

		# At this point, finalOutputPhaseProfile_radians contains the final phase profile to display to SLM, as a floating point in units of radians,
		# and in the range [0, 2pi).

		self.finalOutputPhaseProfile_radians = self.finalOutputPhaseProfile_radians * self.V_2pi / (2.0*np.pi)

		########## ADDED 04/21/23 SOPHIE FLIPPING THE ENTIRE PHASE PROFILE HORZ AND VERT 
		self.finalOutputPhaseProfile_radians = np.flip(self.finalOutputPhaseProfile_radians, 0)

		self.finalOutputPhaseProfile_radians = np.flip(self.finalOutputPhaseProfile_radians, 1)

		## Finally make into an 8bit 
		self.finalOutputPhaseProfile_8bit = (self.finalOutputPhaseProfile_radians).astype(np.uint8)
		#print("Final output phase profile (8 bit):", self.finalOutputPhaseProfile_8bit)

		# Convert data to a 'pixmap' for display.
		im = QtGui.QImage(self.finalOutputPhaseProfile_8bit.data, self.width, self.height, self.width,
			QtGui.QImage.Format_Indexed8)

		pixmap = QtGui.QPixmap(im)


		self.setPixmap(pixmap)


	def setRandomImage(self):
		self.setImage(np.random.rand(self.height, self.width) * 256)


def main():
	app = QtWidgets.QApplication(sys.argv)

	slmController = SLMController(app, shouldEnableSLMDisplay=True, shouldEnableThorcam=True)

	targetIntensities = np.zeros(slmController.dims)
	
	#slmController.feedbackToSimulatedField()

	slmController.show()

	app.exec_()

def main2():
	app = QtWidgets.QApplication(sys.argv)

	slmController = SLMController(app, shouldEnableSLMDisplay=True, shouldEnableThorcam=True)
	slmController.calibrateLA()
	#slmController.show()
	app.exec_()


if __name__ == "__main__":
	main2()
	# app = QtWidgets.QApplication(sys.argv)

	# print(QtWidgets.QDesktopWidget().screenGeometry(1))

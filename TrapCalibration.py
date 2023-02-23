from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from ThorCam import ThorCam
from scipy.optimize import curve_fit
import time
import socket

class Button(QtGui.QPushButton):
	def __init__(self, params):
		super(Button, self).__init__(params)

		font = QtGui.QFont()
		font.setPixelSize(25)
		self.setFont(font)

class TrapCalibration:
	def __init__(self, use_camera):
		self.USE_CAMERA = use_camera
		self.initWindow()

		if self.USE_CAMERA:
			self.cam = ThorCam("trap")


	def initWindow(self):
		background_color = "#ccc"

		self.app = QtGui.QApplication([])

		self.win = QtGui.QWidget()
		self.win.resize(800,800)
		self.app.setActiveWindow(self.win)

		p = self.win.palette()
		p.setColor(self.win.backgroundRole(), QtGui.QColor(background_color))
		self.win.setPalette(p)

		self.layout = QtGui.QGridLayout()
		self.win.setLayout(self.layout)
		self.win.setAutoFillBackground(True)


		pg.setConfigOption('background', background_color)
		pg.setConfigOption('foreground', '#000')




		self.cameraView = pg.ImageView()
		self.cameraView.ui.roiBtn.hide()
		self.cameraView.ui.menuBtn.hide()

		defaultImage = np.random.rand(5,5)
		self.cameraView.setImage(defaultImage)

		self.refreshButton = Button("Update Image")
		self.refreshButton.clicked.connect(self.updateImage)

		self.establishLandscapeButton = Button("Establish Landscape")
		self.establishLandscapeButton.clicked.connect(self.establishLandscape)


		self.feedbackButton = Button("Apply feedback")
		self.feedbackButton.clicked.connect(self.applyFeedback)

		fittedPointsPen = pg.mkPen({'color':'#000', 'width':0})
		interpolationPen = pg.mkPen({'color':'#f00', 'width':2})


		self.fitPlot = pg.PlotWidget()
		self.fitPlotData = self.fitPlot.plot([], [], symbol='o', symbolSize=10, symbolBrush='k')
		self.landscape = self.fitPlot.plot([], [], pen=interpolationPen)


		self.layout.addWidget(self.cameraView, 0, 0, 1, 2)
		self.layout.addWidget(self.refreshButton, 1, 0)
		self.layout.addWidget(self.establishLandscapeButton, 1, 1)
		self.layout.addWidget(self.fitPlot, 2, 0, 1, 2)
		self.layout.addWidget(self.feedbackButton, 3, 1, 1, 1)
		

	def find_traps(self, image, num_traps, local_maximum_size=1, blockade_size=3):
		dtype = [('x', int), ('y', int), ('val', int)]
		local_maxima = []

		image_dup = np.array(np.copy(image), dtype=np.uint16)

		for x in range(blockade_size, len(image) - blockade_size):
			for y in range(blockade_size, len(image[0]) - blockade_size):
				if image[x][y] < 100:
					continue
				is_local_max = True

				for a in range(-local_maximum_size, local_maximum_size + 1):
					for b in range(-local_maximum_size, local_maximum_size + 1):

						if image_dup[x + a][y+b] > image[x][y]:
							is_local_max = False
							break

					if not is_local_max:
						break

				if is_local_max:
					local_maxima.append((x,y, image[x][y]))

					for a in range(-blockade_size, blockade_size + 1):
						for b in range(-blockade_size, blockade_size + 1):
							image_dup[x+a][y+b] = 256

		maxima = np.array(local_maxima, dtype=dtype)
		sorted_maxima = np.sort(maxima, order='val')[::-1]
		trap_positions = sorted_maxima[:num_traps]

		return np.sort(trap_positions, order='y')


	def find_traps2(self, image, num_traps, local_maximum_size=1, blockade_size=3):
		max_y_ind_per_column = []
		max_pixel_per_column = []

		for x in range(len(image)):
			max_y_index = np.argmax(image[x])
			max_y_ind_per_column.append(max_y_index)
			max_pixel_per_column.append(image[x][max_y_index])

		local_maxima = []

		for x in range(1, len(image) - 1):
			y = max_y_ind_per_column[x]

			if max_pixel_per_column[x] < 15:
				continue

			if len(local_maxima) > 0 and x < local_maxima[-1][0] + 3:
				continue

			if max(max_pixel_per_column[x], max_pixel_per_column[x+1]) > max(max_pixel_per_column[x-1], max_pixel_per_column[x+2]):
				if max_pixel_per_column[x] > max_pixel_per_column[x+1]:
					brightest_x = x
				else:
					brightest_x = x+1


				local_maxima.append((brightest_x, max_y_ind_per_column[brightest_x], max_pixel_per_column[brightest_x]))


				# image[brightest_x, y+10:y+20] = 255
		

		dtype = [('x', int), ('y', int), ('val', int)]
		maxima = np.array(local_maxima, dtype=dtype)
		return maxima

		# print max_pixel_per_column


	def gaussian(self, pos, amplitude, x_0, y_0, sigma):
		offset = 1
		return self.gaussian_2D(pos, amplitude, x_0, y_0, sigma, offset)

	def gaussian_2D(self, pos, amplitude, x0, y0, sigma, offset):
		y = pos[0]
		x = pos[1]

		if amplitude < 10.0 or sigma < 0.5 or sigma > 4:
			return -100000

		dist = np.sqrt((x - x0)**2.0 + (y - y0)**2.0)

		ret_val = offset + amplitude * np.exp(-(dist / sigma)**2.0)

		return ret_val.ravel()


	def fit_traps_to_gaussians(self, image, trap_positions):
		opt_params = []
		param_errors = []
		x_positions = []
		for trap_position in trap_positions:
			approximate_x0, approximate_y0 = trap_position[0], trap_position[1]

			approximate_amplitude = 100
			approximate_sigma = 2.8
			approximate_offset = 2

			p0=(approximate_amplitude,
			approximate_x0,
			approximate_y0,
			approximate_sigma)

			region_radius = 5;
			bounds = [[int(approximate_x0 - region_radius), int(approximate_x0 + region_radius)],
			        [int(approximate_y0 - region_radius), int(approximate_y0 + region_radius)]]

			x = np.arange(bounds[0][0], bounds[0][1])
			y = np.arange(bounds[1][0], bounds[1][1])
			y,x = np.meshgrid(y,x)

			image_region = np.array(image)[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]]

			popt, pcov = curve_fit(self.gaussian, (y,x), np.ndarray.flatten(image_region), p0=p0)
			opt_params.append(popt)
			param_errors.append(pcov)

			x_positions.append(popt[0])

		opt_params = np.array(opt_params)
		param_errors = np.array(param_errors)

		return opt_params, param_errors

	def updateImage(self, createLandscape=False):
		image = np.transpose(self.cam.getImage())
		self.cameraView.setImage(image)
		self.fitPlotData.setData([0, 1], [0, np.max(image)])

		starting_time = time.time()

		successful_fit = False
		try:
			local_maximum_size = 1
			blockade_size = 3
			trap_positions = self.find_traps2(image, 101, local_maximum_size, blockade_size)
			middle_time = time.time()


			# print trap_positions
			opt_params, param_errors = self.fit_traps_to_gaussians(image, trap_positions)

			ending_time = time.time()

			successful_fit = True
		except:
			pass



		if not successful_fit:
			print "Unable to fit."


		if successful_fit:
			x_pixels = opt_params[:, 2]
			waists = opt_params[:, 3]
			waist_errors = np.sqrt(param_errors[:, 3,3])
			amplitudes = opt_params[:, 0]
			amplitude_errors = np.sqrt(param_errors[:, 0,0])
			

			powers = amplitudes * waists * waists
			# Naive uncorrelated error analysis. Should probably use the actual covariance matrix to compute this more properly.
			power_errors = waists * np.sqrt((waists * amplitude_errors)**2.0 + (2 * amplitudes * waist_errors)**2.0)


			sorted_indices = np.argsort(x_pixels)
			x_pixels = x_pixels[sorted_indices]
			powers = powers[sorted_indices]


			self.fitPlot.setTitle(title="%d Traps" %len(trap_positions), size='24pt')

			if createLandscape:
				smoothed_x_pixels = 0.5 * (x_pixels[1:] + x_pixels[:-1])
				smoothed_powers = 0.5 * (powers[1:] + powers[:-1])

				dense_x_pixels = np.arange(np.min(x_pixels)-1.0, np.max(x_pixels)+1.0, 0.01)
				dense_powers = np.interp(dense_x_pixels, smoothed_x_pixels, smoothed_powers)

				self.landscape_x_pixels = dense_x_pixels
				self.landscape_powers = dense_powers

				self.landscape.setData(self.landscape_x_pixels, self.landscape_powers)
			self.fitPlotData.setData(x_pixels, powers)

		return x_pixels, powers

	def establishLandscape(self):
		self.updateImage(True)

	def applyFeedback(self):
		x_pixels, powers = self.updateImage(False)

		desired_powers = []

		for x_pixel in x_pixels:
			landscape_index = np.searchsorted(self.landscape_x_pixels, x_pixel)
			desired_powers.append(self.landscape_powers[landscape_index])


		desired_powers = np.array(desired_powers)
		amplitude_factors = np.sqrt(desired_powers / powers)

		feedback_command = "traps multiply_amplitudes"
		for a in amplitude_factors:
			feedback_command += " %.5f" %a
		feedback_command += "\n"

		try:
			# Send amplitude factors over network
			server_IP = "192.168.10.7"
			server_PORT = 1235

			s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s.connect((server_IP, server_PORT))

			print "Trying to send command:"
			print feedback_command
			s.send(feedback_command)

			s.close()
		except:
			print "Error sending feedback."


	def shutdown(self):
		if self.USE_CAMERA:
			self.cam.shutDown()
	


	def start(self):
		self.win.show()
		self.app.exec_()

def main():
	trapCalibration = TrapCalibration(True)

	trapCalibration.start()

	trapCalibration.shutdown()


if __name__ == '__main__':
	main()

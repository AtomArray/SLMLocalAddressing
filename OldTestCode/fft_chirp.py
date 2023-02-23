import numpy as np
import scipy.fftpack
import sys
import pyfftw
import time
import pyqtgraph as pg
from PyQt5 import QtWidgets

def main():
    # Time units are microseconds.
    sweepDuration = 1000.0
    dt = 0.001
    times = np.arange(0.0, sweepDuration, dt)

    f0 = 10.0 # in units of MHz
    f1 = 10.0

    ts = (times / sweepDuration)

    order = 2

    fs = (ts < 0.5) * (f0 + (ts/0.5)**order * (f1-f0)/2.0)
    fs += (ts >= 0.5) * (f1 + ((1.0-ts)/0.5)**order * (f0-f1)/2.0)

    dphis = fs * dt
    phis = np.cumsum(dphis)

    wfm = np.cos(2.0*np.pi * phis)

    mask = times < sweepDuration/2.0
    wfm[mask] = 0.0


    fft = np.fft.rfft(wfm)
    fftfreqs = np.fft.rfftfreq(wfm.shape[0]) / dt
    #fftfreqs = np.fft.rfftfreq(windowsize) / dt

    abs_fft = np.abs(fft)
    angles = np.angle(fft)


    app = QtWidgets.QApplication(sys.argv)

    plot = pg.PlotWidget()
    #plot.plot(times, fs)
    plot.plot(fftfreqs, np.real(fft), pen='#ff0000')
    plot.plot(fftfreqs, np.imag(fft), pen='#00ff00')
    #plot.setXRange(5, 25)
    #plot.plot(fftfreqs, abs_fft, pen='#ff0000')
    #plot.plot(fftfreqs, 1000*np.cos(angles), pen='#00ff00')


    #wfm2 = np.fft.irfft((fft))
    #plot.plot(times, wfm)

    plot.show()



    app.exec_()

    

if __name__ == "__main__":
    main()

import numpy as np
import scipy.fftpack
import sys
import pyfftw
import time
import pyqtgraph as pg
import multiprocessing 
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import matplotlib.pyplot as plt
import timeit
import multiprocessing as mp


def main():
    times = np.arange(100000)
    datapoints = np.zeros((len(times), 1))
    datapoints[100, 0] = 1.0
    datapoints[150, 0] = 1.0

    t1 = time.time()
    #fft = np.fft.fft(datapoints, axis=0)
    fft = scipy.fftpack.fft(datapoints, axis=0)
    t2 = time.time()

    print('Time to calculate fft:', t2-t1)
    #plt.plot(times, fft)
    #plt.show()

def main2():
    numSections = 1
    numPointsPerSection = 131072
    a = pyfftw.empty_aligned((numSections, numPointsPerSection), dtype='complex64', n=16)
    b = pyfftw.empty_aligned((numSections, numPointsPerSection), dtype='complex64', n=16)

    fft_object = pyfftw.FFTW(a, b, axes=(1,))
    
    for i in range(numSections):
        a[i, 200] = 1.0
        a[i, 210] = 1.0
        a[i, 220] = -1.0

    t1 = time.time()
    fft_object()
    t2 = time.time()
    print('Time to calculate fft:', t2-t1)


    app = QtWidgets.QApplication(sys.argv)

    plot = pg.PlotWidget()
    plot.plot(np.arange(1024*128), np.ndarray.flatten(np.real(b)))
    #plot.plot(np.arange(numPointsPerSection), np.real(b[10]))
    plot.show()

    app.exec_()



def addArrays(arg):
    arrs, indices = arg
    output = np.zeros(arrs.shape[1], dtype=np.int32)

    for i in indices:
        output = np.add(arrs[i], output)

    return output

def f(x):
    return 2*x

def g(x, y):
    return x*y

def fastAddArrays(arrs, indices):
    num = len(indices)
    """
    pool = multiprocessing.Pool(8)
    

    indices1 = indices[:int(num/2)]
    indices2 = indices[int(num/2):]

    args = [(arrs, indices1), (arrs, indices2)]

    t1 = time.time()
    

    res = list(pool.map(addArrays, [(arrs, indices1), (arrs, indices2)]))

    t2 = time.time()
    print("Time to calculate sum:", t2-t1)
    return res[0]+res[1]
    """



    t1 = time.time()
    res1 = addArrays((arrs, indices[:int(num/2)]))
    res2 = addArrays((arrs, indices[int(num/2):]))

    result = res1 + res2

    t2 = time.time()
    print("Time to calculate sum:", t2-t1)

    return result
    

def main3():
    times = np.arange(int(1e-3 / 5e-9))

    fs = np.arange(0.1, 0.2, 0.001)
    print("Number of frequencies:", len(fs))

    t1 = time.time()
    phis = np.outer(2.0*np.pi * fs, times)
    ws = np.array(1000.0 * np.cos(phis), dtype=np.int16)
    t2 = time.time()
    print("Time to calculate individual wfms:", t2-t1)

    masks = np.zeros((len(fs), len(times)), dtype=np.bool)
    for i in range(len(fs)):
        masks[i, i*1000:(i+1)*1000] = True


    mask = np.ones(len(fs), dtype=np.bool)
    #mask[40:] = False

    output = np.zeros(len(times), dtype=np.int32)


    output = fastAddArrays(ws, indices=np.arange(len(fs))[mask])
    #for i in np.arange(len(fs))[mask]:
        #output = np.add(ws[i], output)

    w = output
    #w = np.sum(ws[mask], axis=0, dtype=np.int16)
    #w = ws.sum(axis=0)

    #print("Time to calculate sum of wfms:", t2-t1)


    app = QtWidgets.QApplication(sys.argv)

    plot = pg.PlotWidget()
    plot.plot(times, w)
    #plot.plot(np.arange(numPointsPerSection), np.real(b[10]))
    plot.show()

    app.exec_()

def adding_vs_fft():
    times = np.arange(int(3e-3 / 2.5e-9))

    fs = np.arange(0.1, 0.2, 0.001)
    print("Number of frequencies:", len(fs))

    t1 = time.time()
    phis = np.outer(2.0*np.pi * fs, times)
    ws = np.array(1000.0 * np.cos(phis), dtype=np.int16)
    t2 = time.time()
    print("Time to calculate individual wfms:", t2-t1)

    t1 = time.time()
    w = np.sum(ws, axis=0)
    t2 = time.time()
    print("Time to calculate sum:", t2-t1)


    freqs = np.fft.rfftfreq(len(w))

    for i in range(10):

        t1 = time.time()
        fft = np.fft.rfft(w)
        t2 = time.time()

        print("Time to calculate fft:", t2-t1)


    # Test pyFFTW

    n = 1
    numPointsPerSection = len(times)
    a = pyfftw.empty_aligned((n, numPointsPerSection), dtype='float32', n=16)
    b = pyfftw.empty_aligned((n, int(numPointsPerSection//2+1)), dtype='complex64', n=16)
    pyfftw.interfaces.cache.enable()

    for i in range(n):
        a[i, :] = w[:]
    """

    for i in range(10):
        t1 = time.time()
        #fft = pyfftw.interfaces.numpy_fft.rfft(a, avoid_copy=True)
        fft = pyfftw.interfaces.numpy_fft.rfft(a, threads=1)
        t2 = time.time()

        print('Time to calculate pyfft (1 thread):', t2-t1)


    for i in range(10):
        t1 = time.time()
        #fft = pyfftw.interfaces.numpy_fft.rfft(a, avoid_copy=True)
        fft = pyfftw.interfaces.numpy_fft.rfft(a, threads=2)
        t2 = time.time()
        print('Time to calculate pyfft (2 thread):', t2-t1)

    for i in range(10):
        t1 = time.time()
        #fft = pyfftw.interfaces.numpy_fft.rfft(a, avoid_copy=True)
        fft = pyfftw.interfaces.numpy_fft.rfft(a, threads=4)
        t2 = time.time()
        print('Time to calculate pyfft (4 thread):', t2-t1)
    """


    # """ THIS WORKS WELL """
    for i in range(10):
        t1 = time.time()
        #fft = pyfftw.interfaces.numpy_fft.rfft(a, avoid_copy=True)
        fft = pyfftw.interfaces.numpy_fft.rfft(a, threads=8)
        t2 = time.time()
        print('Time to calculate pyfft (8 thread):', t2-t1)

    return



    fft_object = pyfftw.FFTW(a, b, threads=8)
    a[:]  = w[:]
    for i in range(10):
        t1 = time.time()
        fft = fft_object()
        t2 = time.time()
        print('Time to calculate pyfft 2 (8 thread):', t2-t1)


    return
    


    app = QtWidgets.QApplication(sys.argv)

    plot = pg.PlotWidget()
    #plot.plot(times, w)
    plot.plot(freqs, np.abs(fft)**2.0)
    #plot.plot(np.arange(numPointsPerSection), np.real(b[10]))
    plot.show()

    app.exec_()


def square_pulses():
    samplestep = 10e-3
    times = np.arange(int(1e3 /samplestep)) * samplestep
    #fs = np.arange(0.1, 0.2, 0.001)

    fs = np.array([10])

    phis = np.outer(2.0*np.pi * fs, times)
    ws = np.array(1000.0 * np.cos(phis), dtype=np.int16)

    w = np.sum(ws, axis=0)

    w[times > 0.5e3] = 0
    w[times < 0.2e3] = 0

    freqs = np.fft.rfftfreq(len(w)) / samplestep
    fft = np.fft.rfft(w)



    app = QtWidgets.QApplication(sys.argv)

    widget = QtGui.QWidget()
    layout = QtGui.QGridLayout(widget)

    plot1 = pg.PlotWidget()
    plot2 = pg.PlotWidget()
    plot3 = pg.PlotWidget()
    plot4 = pg.PlotWidget()
    plot5 = pg.PlotWidget()
    plot6 = pg.PlotWidget()

    plot1.plot(times, w)
    plot2.plot(freqs, np.abs(fft)**2.0)



    #fft[freqs < 9.7] = 0
    #fft[freqs > 10.3] = 0
    fft *= np.exp(1.0j * 2.0*np.pi * 0.5)


    w = np.fft.irfft(fft)
    plot3.plot(times, w)
    plot4.plot(freqs, np.abs(fft)**2.0)




    analytic_fft = np.zeros(len(fft), dtype=np.complex)
    ks = np.fft.rfftfreq(len(w))
    a, b = 1e4, 2e4
    f = 0.01

    #analytic_fft[:] = np.exp(-2.0j*np.pi * (1 + a + b) * (f + ks))
    #analytic_fft[:] *= -np.exp(-2.0j*np.pi * (f + a*f + a*ks)) - np.exp(2.0j*np.pi * ((1 + a + 2*b)*f + a*ks)) + np.exp(2.0j*np.pi * ((2 + a + 2*b)*f + ks + a*ks)) + np.exp(2.0j*np.pi *((2+b)*f + ks + b*ks)) - np.exp(2.0j*np.pi * ((1+b)*f + (2 + b)*ks)) - np.exp(2.0j*np.pi *((1 + 2*a + b)*f + (2 + b)*ks)) + np.exp(2.0j*np.pi * (ks+a*(f+ks))) + np.exp(2.0j*np.pi * (2*a*f + ks + b * (f + ks)))
    #
    #analytic_fft[:] /= 4.0 * (np.cos(2.0*np.pi * f) - np.cos(2.0*np.pi * ks))


    def analytic_square_pulse_fft(ks, f, a, b, N, truncateFraction=-1, output=None):
        if output is None:
            result = np.zeros(len(ks), dtype=np.complex)
        else:
            result = output

        num_indices_to_keep = int(len(ks) * truncateFraction)
        i_center = int(f * N)
        i1 = i_center - num_indices_to_keep
        i2 = i_center + num_indices_to_keep


        result[:] = 0.5 * np.exp(-2.0j*np.pi/N * (ks + f*N)) * (1 + np.exp(4.0j*np.pi * f))
        return result
        

        prefactor = 2.0j*np.pi/N

        for z in range(2):
            if z == 0:
                good_ks = ks[i1:i_center]
            else:
                good_ks = ks[i_center+1:i2]
            t1 = np.exp(-prefactor * (1 + a + b) * (good_ks + f*N))
            t2 = np.exp( prefactor * (good_ks + a*good_ks + a*f*N))
            t3 = np.exp( prefactor * (a*good_ks + (1 + a)*f*N))
            t4 = np.exp( prefactor * ((2+b)*good_ks+(1+b)*f*N))
            t5 = np.exp( prefactor * ((1+b)*good_ks+(2+b)*f*N))
            t6 = np.exp( prefactor * ((1+b)*good_ks+(2*a+b)*f*N))
            t7 = np.exp( prefactor * ((2+b)*good_ks+(1+2*a+b)*f*N))
            t8 = np.exp( prefactor * (a*good_ks + (1 + a + 2*b) * f * N))
            t9 = np.exp( prefactor * ((1+a)*good_ks+(2+a+2*b)*f*N))
            t10 = 4.0 * (np.cos(2.0*np.pi*f) - np.cos(2.0*np.pi * good_ks / N))

            if z == 0:
                result[i1:i_center] = (t1/t10) * (t2 - t3 - t4 + t5 + t6 - t7 - t8 + t9)
            else:
                result[i_center+1:i2] = (t1/t10) * (t2 - t3 - t4 + t5 + t6 - t7 - t8 + t9)


        result[i_center] = (1.0 + b - (1 + b) * np.exp(4.0j*np.pi * f) - np.exp(-4.0j*np.pi * (a-1)*f) + np.exp(-4.0j*np.pi * b * f) + a*(-1 + np.exp(4.0j*np.pi * f))) / (2.0 - 2.0*np.exp(4.0j*np.pi * f))


        return result

    t1 = time.time()
    output = np.zeros(len(ks), dtype=np.complex)
    for i in range(100):
        analytic_fft = analytic_square_pulse_fft(np.arange(len(ks)), f, a, b, len(w), truncateFraction=0.01, output=output)
        #analytic_fft = analytic_square_pulse_fft(np.arange(len(ks)), f, a, b, len(w))
    t2 = time.time()
    print("time to compute analytic fft:", t2-t1)

    analytic_square_pulse = np.fft.irfft(analytic_fft)
    print(analytic_fft)
    print(analytic_square_pulse)


    plot5.plot(np.arange(len(analytic_square_pulse)), analytic_square_pulse)
    plot6.plot(np.arange(len(analytic_fft)), np.abs(analytic_fft)**2.0)


    layout.addWidget(plot1, 1, 1)
    layout.addWidget(plot2, 2, 1)
    layout.addWidget(plot3, 1, 2)
    layout.addWidget(plot4, 2, 2)
    layout.addWidget(plot5, 1, 3)
    layout.addWidget(plot6, 2, 3)

    widget.show()

    app.exec_()



def simple_square_pulse():
    samplestep = 10e-3
    times = np.arange(int(1e3 /samplestep)) * samplestep

    fs = np.array([10])

    phis = np.outer(2.0*np.pi * fs, times)
    ws = np.array(1000.0 * np.cos(phis), dtype=np.int16)

    w = np.sum(ws, axis=0)

    freqs = np.fft.rfftfreq(len(w)) / samplestep
    fft = np.fft.rfft(w)

    analytic_fft = np.zeros(len(fft), dtype=np.complex)
    ks = np.fft.rfftfreq(len(w))
    a, b = 0e4, 1e3
    f = 0.1


    def analytic_square_pulse_fft(ks, f, a, b, N, truncateFraction=-1, output=None):
        if output is None:
            result = np.zeros(len(ks), dtype=np.complex)
        else:
            result = output

        num_indices_to_keep = int(len(ks) * truncateFraction)
        i_center = int(f * N)
        i1 = i_center - num_indices_to_keep
        i2 = i_center + num_indices_to_keep


        #result[:] = 0.5 * np.exp(-2.0j*np.pi/N * (ks + f*N)) * (1 + np.exp(4.0j*np.pi * f))
        

        prefactor = 2.0j*np.pi/N

        for z in range(2):
            if z == 0:
                good_ks = ks[i1:i_center]
            else:
                good_ks = ks[i_center+1:i2]
            t1 = np.exp(-prefactor * (1 + a + b) * (good_ks + f*N))
            t2 = np.exp( prefactor * (good_ks + a*good_ks + a*f*N))
            t3 = np.exp( prefactor * (a*good_ks + (1 + a)*f*N))
            t4 = np.exp( prefactor * ((2+b)*good_ks+(1+b)*f*N))
            t5 = np.exp( prefactor * ((1+b)*good_ks+(2+b)*f*N))
            t6 = np.exp( prefactor * ((1+b)*good_ks+(2*a+b)*f*N))
            t7 = np.exp( prefactor * ((2+b)*good_ks+(1+2*a+b)*f*N))
            t8 = np.exp( prefactor * (a*good_ks + (1 + a + 2*b) * f * N))
            t9 = np.exp( prefactor * ((1+a)*good_ks+(2+a+2*b)*f*N))
            t10 = 4.0 * (np.cos(2.0*np.pi*f) - np.cos(2.0*np.pi * good_ks / N))

            if z == 0:
                result[i1:i_center] = (t1/t10) * (t2 - t3 - t4 + t5 + t6 - t7 - t8 + t9)
            else:
                result[i_center+1:i2] = (t1/t10) * (t2 - t3 - t4 + t5 + t6 - t7 - t8 + t9)


        result[i_center] = (1.0 + b - (1 + b) * np.exp(4.0j*np.pi * f) - np.exp(-4.0j*np.pi * (a-1)*f) + np.exp(-4.0j*np.pi * b * f) + a*(-1 + np.exp(4.0j*np.pi * f))) / (2.0 - 2.0*np.exp(4.0j*np.pi * f))


        return result

    result = analytic_square_pulse_fft(np.arange(len(ks)), f, a, b, len(w), truncateFraction=0.1)

    ks = np.arange(len(ks))

    analytic = 1 / (((ks - 1e4)/30)**2.0 + 1)
    x = 10.0/np.pi
    analytic2 = (np.sin((ks - 1e4)/x) / ((ks-1e4)/x)) * np.exp(2.0j*np.pi * ks/2)
    analytic2[int(1e4)]=1
    
    wfm = np.fft.irfft(analytic2)
    plt.plot(wfm)
    plt.show()

    return


    plt.plot(np.abs(result)**2.0)
    #plt.plot(3e5 * analytic)
    plt.plot(2.5e5 * np.abs(analytic2)**2.0)
    plt.yscale('log')
    plt.ylim(0.1, 1e6)
    plt.xlim(8000, 12000)


    plt.show()


    plt.plot(np.angle(result))
    plt.show()


def design_square_pulse():
    samplestep = 10e-3
    times = np.arange(int(1e3 /samplestep)) * samplestep

    fs = np.array([10])

    phis = np.outer(2.0*np.pi * fs, times)
    ws = np.array(1000.0 * np.cos(phis), dtype=np.int16)

    w = np.sum(ws, axis=0)

    freqs = np.fft.rfftfreq(len(w)) / samplestep
    fft = np.fft.rfft(w)

    #analytic_fft = np.zeros(len(fft), dtype=np.complex)
    analytic_fft = pyfftw.empty_aligned(len(fft), dtype='complex64', n=16)
    wfm = pyfftw.empty_aligned(len(times), dtype='float32', n=16)

    analytic_fft[:] = 0

    ks = np.fft.rfftfreq(len(w))

    ks = np.arange(len(ks))


    np.seterr(divide='ignore', invalid='ignore')


    shift_amounts = np.arange(-1.0, 1.0, 0.02)
    shift_phases = 2.0*np.pi * np.outer(ks, shift_amounts)
    shift_matrix = np.exp(1.0j * shift_phases)


    shift_dictionary = {'shift_amounts':shift_amounts, 'shift_matrix':shift_matrix}



    traps = np.array([
        [0.01, 0.1, 0.0, 0e4, 2e4],
        [0.02, 0.1, 0.0, 2.5e4, 4e4],
        [0.03, 0.1, 0.0, 6e4, 9e4],
        [0.04, 0.1, 0.0, 0e4, 3e4],
        [0.05, 0.1, 0.0, 1e4, 2e4],
        [0.06, 0.1, 0.0, 4e4, 9e4],
        [0.07, 0.1, 0.0, 3e4, 5e4],
        [0.08, 0.1, 0.0, 3e4, 6e4],
        [0.09, 0.1, 0.0, 2e4, 8e4],
        [0.1,  0.1, 0.0, 1e4, 7e4],
    ])

    traps = np.array([
        [0.01, 0.1, 0.0, 0e4, 2e4],
        [0.01, 0.1, 0.0, 4e4, 6e4]
    ])


    #pyfftw.interfaces.cache.enable()

    fftw = pyfftw.FFTW(analytic_fft, wfm, threads=1, direction='FFTW_BACKWARD', flags=('FFTW_ESTIMATE',))
    fftw()
    #pyfftw.interfaces.numpy_fft.irfft(analytic_fft, threads=8)


    #print(multiprocessing.cpu_count())

    t1 = time.time()

    for i in range(1):
        analytic_fft[:] = 0
        

        # Construct Fourier transform analytically
        analytic_square_pulse(traps, ks, len(times), analytic_fft, shift_dictionary)


        
        # Compute inverse Fourier transform

        if True:
            fftw()
        elif False:
            wfm = pyfftw.interfaces.numpy_fft.irfft(analytic_fft, threads=8)
        else:
            wfm = np.fft.irfft(analytic_fft)


        wfm = np.array(2**15 + 2**15 * wfm, dtype=np.uint16)

    t2 = time.time()
    print("Took total time:", t2-t1)






    ### Plot resulting Fourier transform and time-domain waveform

    plt.subplot(211)
    plt.plot(np.real(analytic_fft))
    plt.plot(np.imag(analytic_fft))
    
    plt.subplot(212)
    plt.plot(times, wfm)
    plt.show()



def analytic_square_pulse(traps, ks, N, output, shift_dictionary):
    fs = traps[:, 0]
    amps = traps[:, 1]
    phis = traps[:, 2]

    a_s = traps[:, 3]
    b_s = traps[:, 4]



    widths = b_s-a_s

    prefactors = amps * widths/2.0 * np.exp(2.0j*np.pi * -phis)

    centers = fs * N

    xs = N / widths
    shifts = -(0.5*widths + a_s)/N



    truncation_range = 200
    starts = np.array(centers - truncation_range, dtype=np.int)
    ends = np.array(centers + truncation_range, dtype=np.int)

    starts[starts < 0] = 0


    closest_shift_indices = np.searchsorted(shift_dictionary['shift_amounts'], shifts)

    actual_shifts = shift_dictionary['shift_amounts'][closest_shift_indices]


    relevant_ks = np.array([ks[starts[i]:ends[i]] for i in range(len(traps))])

    
    args = ((relevant_ks.T - centers)/xs).T
    sincs = np.sinc(args)


    for i in range(len(traps)):
        shift_factors = shift_dictionary['shift_matrix'][:, closest_shift_indices[i]]

        output[starts[i]:ends[i]] += prefactors[i] * shift_factors[starts[i]:ends[i]] * sincs[i]



if __name__ == "__main__":


    #adding_vs_fft()
    #square_pulses()
    #simple_square_pulse()
    design_square_pulse()
    #multiprocessing_test()

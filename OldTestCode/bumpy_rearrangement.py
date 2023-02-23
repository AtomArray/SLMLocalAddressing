import numpy as np
import matplotlib.pyplot as plt


# All units:
# Position -- microns
# Time -- microseconds
# Energy -- MHz


hbar = 1.055e-28 # (micron^2 * kg) / (microsecond)
m_rb = 87 * 1.66e-27 # mass in kg

def trapPotential(x, x0, A, waist):
    return -A * np.exp(-2 * (x-x0)**2.0/(waist**2.0))

def getPotential(t, x, tMax=500):
    staticTrapA = 6.0 # MHz

    trapPositions = np.arange(0, 100.1, 5)

    staticTraps = sum([trapPotential(x, pos, staticTrapA, 1) for pos in trapPositions])


    movingTrapA = 30.0
    movingTrapA *= (t >= 0) * (t < tMax)
    center = trapPositions[0] + (t / tMax) * (trapPositions[-1] - trapPositions[0])
    movingTrap = trapPotential(x, center, movingTrapA, 1)

    return staticTraps + movingTrap



def main():
    x_axis = np.linspace(-5, 15, 1000)
    #fullPotential = getPotential(0, x_axis)
    #plt.plot(x_axis, fullPotential)
    #plt.show()

    dt = 0.05
    tMax = 800

    ts = np.arange(-100, tMax + 100, dt)


    positions = [0.0]
    velocities = [0.0]

    for t in ts:
        x = positions[-1]
        v = velocities[-1]

        dx = 0.01

        dUdx = (getPotential(t, x + dx, tMax) - getPotential(t, x - dx, tMax)) / (2*dx)
        F = -dUdx

        acc = F / (m_rb / (2.0*np.pi * hbar))

        v += acc * dt
        x += v * dt


        positions.append(x)
        velocities.append(v)

    plt.plot(ts, positions[:-1])
    plt.show()

    




if __name__ == "__main__":
    main()

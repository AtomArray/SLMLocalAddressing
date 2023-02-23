import numpy as np
from scipy.special import eval_jacobi

import matplotlib.pyplot as plt

### Previous values for zernike polynomials, from before 10/14/2022 ###
# ordered_polynomials = [
#         [(0, 0),  "Piston",                 0.0],
#         [(1, -1), "Y-Tilt",                 0.0],
#         [(1, 1),  "X-Tilt",                 0.0],
#         [(2, -2), "Oblique_astigmatism",    28.0],
#         [(2, 0), "Defocus",                 -20.0],
#         [(2, 2), "Vertical_astigmatism",    22.4],
#         [(3, -3), "Vertical_trefoil",       8.54],
#         [(3, -1), "Vertical_coma",          -13.3],
#         [(3, 1), "Horizontal_coma",         40.0],
#         [(3, 3), "Oblique_trefoil",         5.3],
#         [(4, -4), "Oblique_quadrafoil",     0.0],
#         [(4, -2), "Horizontal_secondary_astigmatism",   -6.3],
#         [(4, 0), "Primary_spherical",       46.3],
#         [(4, 2), "Vertical_secondary_astigmatism",  3.5],
#         [(4, 4), "Vertical_quadrafoil",     0.0],
#         [(6, 0), "Secondary_Spherical",     1.36]
#     ]

ordered_polynomials = [
        [(0, 0),  "Piston",                 0.0],
        [(1, -1), "Y-Tilt",                 0.0],
        [(1, 1),  "X-Tilt",                 0.0],
        [(2, -2), "Oblique_astigmatism",    20.5],
        [(2, 0), "Defocus",                 -40.0],
        [(2, 2), "Vertical_astigmatism",    27.3],
        [(3, -3), "Vertical_trefoil",       -5.0],
        [(3, -1), "Vertical_coma",          26.6],
        [(3, 1), "Horizontal_coma",         7.0],
        [(3, 3), "Oblique_trefoil",         6.0],
        [(4, -4), "Oblique_quadrafoil",     0.0],
        [(4, -2), "Horizontal_secondary_astigmatism",   -6.3],
        [(4, 0), "Primary_spherical",       12.4],
        [(4, 2), "Vertical_secondary_astigmatism",  3.5],
        [(4, 4), "Vertical_quadrafoil",     0.0],
        [(6, 0), "Secondary_Spherical",     1.36],
    ]

# See: http://mathworld.wolfram.com/ZernikePolynomial.html
# and https://en.wikipedia.org/wiki/Zernike_polynomials
def radial_polynomial(rho, m, n):
    if (n - m) % 2 == 1:
        print("ODD")
        return 0

    eff_n = (n-m)//2
    alpha = m
    beta = 0
    x = 1 - 2 * (rho**2.0)
    return (-1)**(eff_n) * (rho**m) * eval_jacobi(eff_n, alpha, beta, x)

def zernike(rho, phi, m, n):
    # Zernike polynomials are normalized (according to Wikipedia: OSA/ANSI) such that the integral of the
    # polynomial squared over the unit disk is equal to Pi. Here we enforce this normalization.
    # See opt.indiana.edu/vsg/library/vsia/vsia-2000_taskforce/tops4_2.html
    normalization = np.sqrt(2 * (n+1))
    if m == 0:
        normalization /= np.sqrt(2)

    if m == 0:
        return normalization * radial_polynomial(rho, m, n)
    elif m > 0:
        return normalization * radial_polynomial(rho, m, n) * np.cos(m * phi)
    elif m < 0:
        return normalization * radial_polynomial(rho, -m, n) * np.sin(m * phi)



def main():
    dims = [1024, 1024]

    xs = (np.arange(dims[0]) - dims[0]//2) / (dims[0]//2)
    ys = (np.arange(dims[1]) - dims[1]//2) / (dims[1]//2)

    xx, yy = np.meshgrid(xs, ys)

    rhos = np.sqrt(xx**2.0 + yy**2.0)
    phis = np.arctan2(yy, xx)

    # print(np.max(rhos))

    for i in range(5):
        n, m = ordered_polynomials[i][0]
        label = ordered_polynomials[i][1]

        phase_profile = zernike(rhos, phis, m, n) # Phase profile calculated over entire unit square

        # Mask phase profile outside of unit disk? This is generally not good unless we aperture the SLM phase mask.
        # phase_profile[rhos >= 1.0] = 0.0

        # Check normalization of Zernike polynomial?
        print("Normalization factor:", np.sqrt(1 / np.mean(phase_profile[rhos <= 1.0]**2.0)))
        
        plt.imshow(phase_profile, cmap='rainbow', vmin=-1, vmax=1)
        plt.title("n=%d, m=%d: %s" %(n, m, label))
        plt.show()
    



if __name__ == "__main__":
    main()

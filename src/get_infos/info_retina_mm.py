import math
import numpy as np


def compute_rf_gang(d_rf_bip, radius_gang, spacing_gang):
    return (d_rf_bip*3*2)+(radius_gang//spacing_gang//2)*spacing_gang*2


def distance(point1, point2):
    return math.sqrt(abs(point1[0] - point2[0])**2 + abs(point1[1] - point2[1])**2)


def um_to_deg(point, ratio):
    return f"{point[0] * ratio}°, {point[1] * ratio}°"


def gaussian(radius, dist, length, n_cell):
    r2 = radius*radius
    sigma2 = radius/3 * radius/3
    dist_sqr = dist * dist
    norm_g = (1 - np.exp(-r2/2))*(2*np.pi*sigma2)
    w = np.exp(-1 * dist_sqr / (2 * sigma2)) / norm_g
    return w

def compute_cell_id(x,y,z, X, Y):
    return X*Y*z + (X*(x-1)+(y-1))

print(compute_rf_gang(90, 300, 60))
print(compute_rf_gang(50, 170, 95))
print(um_to_deg((13.1, 10.3), 0.3))
print(gaussian(3, 1.05, 20, 20))
print(gaussian(2.7, 18/19, 18, 20) * 0.33)
print(gaussian(4.5, 30/19, 30, 20)* 0.2)

print(compute_cell_id(19,19, 3,37,37))
print(compute_cell_id(19,19, 4,37,37))
print(compute_cell_id(7,8, 3,14,14))
print(compute_cell_id(7,8, 4,14,14))
print(compute_cell_id(5,6, 3,10,10))

print()
print(compute_cell_id(3,19, 4,37,37))
print(compute_cell_id(35,19, 4,37,37))
print(compute_cell_id(3,19, 3,37,37))
print(compute_cell_id(35,19, 3,37,37))

print()
print(compute_cell_id(21,21, 3,40,40))
print(compute_cell_id(21,21, 4,40,40))
print(compute_cell_id(1,6, 3,10,10))
print(compute_cell_id(10,10, 3,10,10))




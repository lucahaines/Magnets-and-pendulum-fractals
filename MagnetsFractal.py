#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:42:57 2024

@author: lucahaines
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
from PIL import Image


# Magnet positions

R = 1  # Circumradius
n = 7 # number of vertices

# Finding coordinates of vertices for n-sided polygon

def npolygoncoords(R, n):
    coords = []
    for i in range(n):
        ncoord = []
        ncoord.append(R * np.sin((2*np.pi*i)/n))
        ncoord.append(R * np.cos((2*np.pi*i)/n))
        coords.append(ncoord)
    return np.array(coords)

magnets = npolygoncoords(R, n)

# Defining coordinates

x1 = magnets[0][0]
y1 = magnets[0][1]
x2 = magnets[1][0]
y2 = magnets[1][1]
x3 = magnets[2][0]
y3 = magnets[2][1]
x4 = magnets[3][0]
y4 = magnets[3][1]
x5 = magnets[4][0]
y5 = magnets[4][1]
x6 = magnets[5][0]
y6 = magnets[5][1]
x7 = magnets[6][0]
y7 = magnets[6][1]



# Custom coordinates option

# a = 1
# x1 = a/2
# y1 = a/2
# x2 = -a/2
# y2 = a/2
# x3 = -a/2
# y3 = -a/2
# x4 = a/2
# y4 = -a/2
# x5 = 0 
# y5 = 0
# x6 = 3a/2
# y6 = 4
# x7 = a**2
# y7 = 99

# Height of pendulum plane above magnets
d = 0.1

# Drag coefficient
b = 0.5

# Gravity
c = 0.5

# charge of magnets
f = 5

# Time
dt = 0.01
end_time = 5
start_time = 0
span = (start_time, end_time)
time = np.arange(start_time, end_time, dt)

# Function that finds differential velocity and acceleration
def motion(t, state):
    
    """ 
    Function returns the differential velocities and accelerations in both x
    and y when given some xy position and xy velocity input
    """
    x, vx, y, vy = state
    
    
    dxdt = vx
    
    dvxdt = f*((x1-x)/(((x1-x)**2+(y1-y)**2+(d)**2)**(3/2)) +\
    (x2-x)/(((x2-x)**2+(y2-y)**2+(d)**2)**(3/2)) +\
    (x3-x)/(((x3-x)**2+(y3-y)**2+(d)**2)**(3/2)) +\
    (x4-x)/(((x4-x)**2+(y4-y)**2+(d)**2)**(3/2)) +\
    (x5-x)/(((x5-x)**2+(y5-y)**2+(d)**2)**(3/2)) +\
    (x6-x)/(((x6-x)**2+(y6-y)**2+(d)**2)**(3/2))+\
    (x7-x)/(((x7-x)**2+(y7-y)**2+(d)**2)**(3/2))) - b*vx - c*x
    
    dydt = vy
    
    dvydt = f*((y1-y)/(((x1-x)**2+(y1-y)**2+(d)**2)**(3/2)) +\
    (y2-y)/(((x2-x)**2+(y2-y)**2+(d)**2)**(3/2)) +\
    (y3-y)/(((x3-x)**2+(y3-y)**2+(d)**2)**(3/2)) +\
    (y4-y)/(((x4-x)**2+(y4-y)**2+(d)**2)**(3/2))+\
    (y5-y)/(((x5-x)**2+(y5-y)**2+(d)**2)**(3/2))+\
    (y6-y)/(((x6-x)**2+(y6-y)**2+(d)**2)**(3/2))+\
    (y7-y)/(((x7-x)**2+(y7-y)**2+(d)**2)**(3/2)))- b*vy - c*y
    
    return [dxdt, dvxdt, dydt, dvydt]


def integrate(x0, vx0, y0, vy0):
    """
    Returns an array describing the motion of a ball in the magnetic field
    when given an initial x and y position
    """
    output = sp.solve_ivp(motion, span, [x0, vx0, y0, vy0], t_eval = time,
                          method = 'RK45',rtol=1e-13)
    return output.y[0], output.y[2]


# plot result

x0 = 0.01
y0 = 0
vx0 = 0
vy0 = 0

plt.figure(figsize=(8, 8))
plt.scatter([x1,x2, x3, x4, x5, x6, x7], [y1, y2, y3, y4, y5, y6, y7], 
            color='red', zorder=5, label='Magnets')
x_traj, y_traj = integrate(x0, vx0, y0, vy0)
plt.plot(x_traj, y_traj, label='Particle Path')
plt.plot(x_traj[0], y_traj[0], 'go', label='Start')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('Particle Motion in Magnetic Field')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()


# Distance function
def dist(a, b):
    """
    Finds distance between two points
    """
    return np.sqrt(np.sum((a - b) ** 2, axis=0))

# Create image
W, H = 2000, 2000
im = Image.new("RGB", (W, H))
pixels = im.load()

colors = [(255, 87, 51),  # Red-Orange
    (75, 192, 192),  # Teal
    (255, 206, 86),  # Yellow
    (153, 102, 255),  # Purple
    (255, 159, 64),  # Orange
    (255, 99, 132),  # Pink-Red
    (101, 143, 255),  # Periwinkle
    (103, 210, 91),]  # Green

for x in range(W):
    print(x)
    for y in range(H):
        pos_y = 4 * (y / H) - 2
        pos_x = 4 * (x / W) - 2

        xs, ys = integrate(pos_x, 0, pos_y, 0)
        final_pos = np.array([xs[-1], ys[-1]])

        distances = []
        for magnet in magnets:
            distances.append(dist(final_pos, magnet))

        closest_magnet = np.argmin(distances)
        pixels[x, y] = colors[closest_magnet % len(colors)]

im.show()
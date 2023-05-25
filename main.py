#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from collections import Counter

# Adapted from https://www.youtube.com/watch?v=oYnfDcU33i4
# https://github.com/lukepolson/youtube_channel/blob/main/Python%20Metaphysics%20Series/vid37.ipynb

# # Theory

# For a vibrating 2D plate, let the height at some point on the plate during vibration be $u(x,y,t)$. The differential equation governing the motion is given by
# 
# $$u_{tt} = c^2\nabla^2 u$$
# 
#  where $c$ is the wave speed. The *eigenmodes* of vibration occur when the plate is vibrated at very specific frequencies:
# 
# $$f(n,m) = \frac{c}{2\pi} \sqrt{\frac{n^2}{L_x^2} + \frac{m^2}{L_y^2}}$$
# 
# where $L_x$ and $L_y$ are the dimensions of the rectangular plate and $n$ and $m$ are integers. In particular, if one knows $c$, $L_x$, and $L_y$, one can plug in different integer values of $n$ and $m$ to determine precisely what these special frequencies are.
# 
#  At these special frequencies, we have that $u(x,y,t) = U(x,y)G(t)$ (the solution in space and time is seperable) where $G(t)$ is a periodic oscillating function. If the plate is suspended in the center, then the boundary conditions are such that maximum amplitude occurs at the boundary of the plates, and 
# 
# $$U(x,y) \propto |\sin(n \pi x/L_x)\sin(m \pi y/L_y) - \sin(m \pi x/L_x)\sin(n \pi y/L_y)|$$
# 
# where $n$ and $m$ are **odd** integers corresponding to a particular driving frequency that gives a particular mode of vibration.
# 
# 

# Lets create the coordinates of the rectangular grid


x = y = np.linspace(-1, 1, 1000)
xv, yv = np.meshgrid(x,y)

# Define a function the represents the amplitude $U(x,y)$

def amplitude(xv, yv, n, m):
    return np.abs(np.sin(n*np.pi*xv/2)*np.sin(m*np.pi*yv/2)-np.sin(m*np.pi*xv/2)*np.sin(n*np.pi*yv/2))

# This `Sand` class will include
# 1. The number of points `N_points`
# 2. The locations of the points `points`
# 3. The distance `delta` a grain of sand moves when at a position of maximum vibrating amplitude
# 4. A function `move`, which will move all the sand grains at once. Each grain moves in random direction specified by `angle`
# 
# In addition, the previous locations of the points `prev_points` will be tracked so that smooth animations can be produced later (can plot intermediate locations through interpolation)

# In[9]:

CLIP = 0.99
RANDOM_AMPLITUDE = 0.05


class Sand:
    def __init__(self, N_points, amplitude, delta=0.05):
        self.N_points = N_points
        self.pointgroups = {}
        for g in [-1, 1]:
            self.pointgroups[g] = np.random.uniform(-1, 1, size=(2,self.N_points))
        self.prev_points = {}
        self.amplitude = amplitude
        self.delta = delta

    def moverandom(self):
        for colorindex, points in self.pointgroups.items():
            angles = np.random.uniform(0, 2 * np.pi,
                                       size=self.N_points)  # assuming they move randomly, so this will only work with _many_ particles or over long time frames?
            dr = self.delta * np.array([np.cos(angles), np.sin(angles)])# \
                 #* self.amplitude(points[0], points[1], n, m) / 2
                # well, will still have an amplitude distribution of some sort, but maybe we can omit that here

            self.prev_points[colorindex] = np.copy(points)
            self.pointgroups[colorindex] += dr * RANDOM_AMPLITUDE
            self.pointgroups[colorindex] = np.clip(self.pointgroups[colorindex], -CLIP, CLIP)

    def move(self, n=None, m=None):#, **amplitude_params):
        if n is None:
            paramstats = Counter()
            # pairs = [(3,5), (5,3)]
            pairs = []
            P = 13
            for n in range(3,3*P+1,2):
                for m in range(3,3*P+1,2):
                    if n != m:
                        pairs.append((n,m))
            # can work with same pairs? (3,3), (5,5) etc.?
            for n, m in pairs:
                paramstats[(n,m)] = 0
                for colorindex, points in self.pointgroups.items():
                    angles = np.random.uniform(0, 2*np.pi, size=self.N_points)#assuming they move randomly, so this will only work with _many_ particles or over long time frames?
                    dr = self.delta * np.array([np.cos(angles), np.sin(angles)]) \
                          * self.amplitude(points[0], points[1], n, m) / 2
                    newpos = points + dr
                    newpos = np.clip(newpos, -CLIP, CLIP)
                    paramstats[(n,m)] += colorindex*np.average(newpos[0]) #assuming colorindex in {-1,1}, separate by x-position

            # simulation optimization, keep best result previously - NOPE: particles actually move randomly, can't just choose the best, have to simulate movement independence of simulation
            bestn, bestm = list(paramstats.most_common())[0][0]
            #print(bestn, bestm, paramstats)
        else:
            bestn, bestm = n, m

        for colorindex, points in self.pointgroups.items():
            angles = np.random.uniform(0, 2 * np.pi, size=self.N_points)
            dr = self.delta * np.array([np.cos(angles), np.sin(angles)]) \
                 * self.amplitude(points[0], points[1], bestn, bestm) / 2
            self.prev_points[colorindex] = np.copy(points)
            self.pointgroups[colorindex] += dr
            self.pointgroups[colorindex] = np.clip(self.pointgroups[colorindex], -CLIP, CLIP)

        return bestn, bestm


colordict = {-1:"red",1:"blue"}#{-1:[255,0,0],1:[0,0,255]}
#colordictv = np.vectorize(colordict.get)

SPLITAT = 250

NUMPARTICLES = 50
NUMFRAMES = 1000
ensemble = Sand(NUMPARTICLES, amplitude, delta=0.075)

for colorindex, points in ensemble.pointgroups.items():
    plt.plot(*points, marker='o', ms=2, color=colordict[colorindex])
plt.grid()
plt.xlim(-1,1)
plt.ylim(-1,1)


# Here we make an animation:
# * Every 5 frames, we move all the points
# * For the in between frames, we use linear interpolation to get the position as each grain of sand is going from its initial position to its moved position (this enables a smooth animation)

fig, ax = plt.subplots(1,1, figsize=(10,10))
lines = {}
for colorindex, points in ensemble.pointgroups.items():
    ln1, = plt.plot([], [], "o", ms=2, color=colordict[colorindex])
    lines[colorindex] = ln1
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_facecolor('black')
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

"""
# move a bit
for i in range(10000):
    if i % 100 == 0:
        print(i)
    ensemble.move()
"""

def animate(i):

    print(i)
    n, m = ensemble.move()
    ax.set_title(f"n={n} m={m}")
    ensemble.moverandom()

    for colorindex, points in ensemble.pointgroups.items():
        points = ensemble.prev_points[colorindex] + (i%5)/5 *(points-ensemble.prev_points[colorindex])
        lines[colorindex].set_data(*points)

ani = animation.FuncAnimation(fig, animate, frames=NUMFRAMES, interval=50)
#ani.save('sand.gif',writer='pillow',fps=25,dpi=200)
writer = animation.FFMpegWriter(fps=60)
ani.save("sand.mp4", writer=writer)

# XXX possible to only consider sample/subset when choosing best function?




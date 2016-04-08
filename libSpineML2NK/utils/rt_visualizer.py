#!/usr/bin/env python

"""
Real Time Visualisation 
"""
import pdb
import numpy as np1i

from vispy import app, scene

class visualizer(object):
    """
	A class for the real-time visualisation
    """

    def __init__(self,N=1,M=400,scale=1):

	self.N = N
	self.M = M
	self.scale=scale

        self.canvas = scene.SceneCanvas(keys='interactive', show=True, size=(400, 400))

        self.grid = self.canvas.central_widget.add_grid()
        self.view = self.grid.add_view(0, 1)
        self.view.camera = 'panzoom'

        cols = int(self.N**0.5)
        
        self.view.camera.rect = (0, -(self.N/cols)/2., cols, self.N/cols)

        self.lines = scene.ScrollingLines(n_lines=self.N, line_size=M, columns=cols, dx=0.8/M,
                             cell_size=(1, 8), parent=self.view.scene)
	self.lines.transform = scene.STTransform(scale=(1, 1/8.))

    def update_image(self,data):
        self.lines.roll_data(self.scale*data)
        app.process_events()

from dolfin import *
from mshr import *
from matplotlib import cm, pyplot as pp
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import numpy as np, sys, os
import networkx as nx
import random as rnd


def result_directory():
    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M%S")
    rd = os.path.join(os.path.dirname(__file__), './results/' + dt_string)
    if not os.path.isdir(rd): os.makedirs(rd)
    return rd


def create_mesh(coord, n):
    domain = RectangleMesh(Point(coord[0][0], coord[0][1]), Point(coord[1][0], coord[1][1]),
                           n[0], n[1], 'crossed')
    return domain

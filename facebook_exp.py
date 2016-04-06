import networkx
import math
import scipy.optimize
import numpy
import sys
from lib.vis import *
from lib.graph_signal_proc import *
from lib.optimal_cut import *
from lib.time_graph import *
from lib.io import *

from scipy import linalg
import matplotlib.pyplot as plt
from IPython.display import Image
from numpy.linalg import eigh

G = read_time_graph('/home/arlei/Phd/research/network_process/data/facebook/facebook_dyn.graph', 10)

x = eigen_vec_cut_inv(G)
c = sweep(G, x)

print(c)

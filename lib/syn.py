import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import linalg
import matplotlib.pyplot as plt
from IPython.display import Image
import pywt
import scipy.fftpack
import random
import operator
import copy
from collections import deque
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering

def compute_distances(center, graph):
	distances = networkx.shortest_path_length(graph, center)
    
	return distances

def compute_embedding(distances, radius, graph):
	B = []
	s = 0
	nodes = {}
	for v in graph.nodes():
		if distances[v] <= radius:
			B.append(1)
			s = s + 1
		else:
			B.append(0)
            
	return numpy.array(B)

def generate_dyn_cascade(G, diam, duration, n):
	Fs = []
    
	for j in range(n):
		v = random.randint(0, len(G.nodes())-1)
		distances = compute_distances(G.nodes()[v], G)

		if diam > duration:
			num_snaps = diam
		else:
			num_snaps = duration
     
		for i in range(num_snaps):
			r = int(i * math.ceil(float(diam)/duration))
        
			F = compute_embedding(distances, r, G)
			Fs.append(F)
        
	return numpy.array(Fs)

def generate_dyn_heat(G, s, jump, n):
	Fs = []
	L = networkx.normalized_laplacian_matrix(G)
	L = L.todense()
	F0s = []	
	seeds = []

	for i in range(s):
		F0 = numpy.zeros(len(G.nodes()))
		v = random.randint(0, len(G.nodes())-1)
		seeds,append(v)
		F0[v] = 1.0
		F0s.append(F0)

	Fs.append(numpy.sum(F0s, axis=0))

	for j in range(n):
		FI = []
		for i in range(s):
			FI = numpy.multiply(linalg.expm(-i*jump*L), F0s[i])[:,seeds[i]]
			FIs.append(FI)
		
		Fs.append(numpy.sum(FIs, axis=0))

	return numpy.array(Fs)

def generate_dyn_gaussian_noise(G, n):
	Fs = []
	
	for j in range(n):
		F = numpy.random.rand(len(G.nodes()))
		Fs.append(F)

	return numpy.array(Fs)

def generate_dyn_sync_smooth_gaussian_noise(G, freq, n):
	Fs = []
	t = numpy.arange(0, n, float(1) / n)
	y = numpy.sin(2*numpy.pi*freq*t)

	for j in range(n):
		F = y[j]*numpy.random.rand(len(G.nodes()))
		Fs.append(F)

	return numpy.array(Fs)

def generate_dyn_bursty_noise(G, n):
	Fs = []
	bursty_beta = 1
	non_bursty_beta = 1000
	bursty_bursty = 0.7
	non_bursty_non_bursty = 0.9
	bursty = False

	for j in range(n):
		r = random.random()

		if not bursty:
			if r > non_bursty_non_bursty:
				bursty = True
		else:
			if r > bursty_bursty:
				bursty = False

		if bursty:	
			F = numpy.random.exponential(bursty_beta, len(G.nodes()))
		else:
			F = numpy.random.exponential(non_bursty_beta, len(G.nodes()))
			
		Fs.append(F)

	return numpy.array(Fs)

def generate_dyn_indep_cascade(G, s, p):
	Fs = []
	
	seeds = numpy.random.choice(len(G.nodes()), s, replace=False)
	
	F0 = numpy.zeros(len(G.nodes()))
	
	ind = {}
	i = 0

	for v in G.nodes():
		ind[v] = i
		i = i + 1
	
	for s in seeds:
		F0[s] = 2.0

	while True:
		F1 = numpy.zeros(len(G.nodes()))
		new_inf = 0
		for v in G.nodes():
			if F0[ind[v]] > 1.0:
				for u in G.neighbors(v):
					r = random.random()
					if r <= p and F0[ind[u]] < 1.0:
						F1[ind[u]] = 2.0
						new_inf = new_inf + 1
				F1[ind[v]] = 1.0
				F0[ind[v]] = 1.0
			elif F0[ind[v]] > 0.0:
				F1[ind[v]] = 1.0
		
		Fs.append(F0)
		
		if new_inf == 0 and len(Fs) > 1:
			break

		F0 = numpy.copy(F1)
	
	return numpy.array(Fs)
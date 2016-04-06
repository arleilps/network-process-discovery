import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import sparse

from lib.vis import *
from lib.graph_signal_proc import *
from lib.optimal_cut import *
from scipy import linalg
import matplotlib.pyplot as plt
from IPython.display import Image
from numpy.linalg import eigh

class TimeGraph(object):
	def __init__(self, swp_cost=1., t_max=0):
		self.graphs = [networkx.Graph()]
		self.swp_cost = swp_cost
	
	def swap_cost(self):
		return self.swp_cost

	def extend(self, t):
		while len(self.graphs) <= t:
			self.graphs.append(networkx.Graph())

			for v in self.graphs[0].nodes():
				self.graphs[-1].add_node(v)
		            
	def add_edge(self, v1,v2,t,w=1.):
		self.extend(t)
		
		if v1 not in self.graphs[0]:
			self.add_node(v1)
						                
		if v2 not in self.graphs[0]:
			self.add_node(v2)
										           
		self.graphs[t].add_edge(v1, v2, weight=w)

	def add_node(self, v):
		for t in range(len(self.graphs)):
			self.graphs[t].add_node(v)

	def add_snap(self, G, t):
		for (v1,v2) in G.edges():
			self.add_edge(v1,v2,t)

	def window(self, tb, te):
		G = TimeGraph()
		for t in range(tb, te+1):
			G.add_snap(self.snap(t), t-tb)

		return G
	
	def size(self):
		return len(self.graphs[0])
	     
	def num_snaps(self):
		return len(self.graphs)

	def snap(self,t):
		return self.graphs[t]

	def norm_cut(self):
		c = normalized_cut(self.g)
		                    
		return c
	
	def index(self, p):
		return self.graphs[0].nodes()[p % self.size()], int(p / self.size())

	def set_values(self, f):
		i = 0
		for t in range(len(self.graphs)):
			for v in self.graphs[t].nodes():
				self.graphs[t].node[v]["value"] = f[i]
				i = i + 1

def sweep(G, x):
	best_score = sys.float_info.max
	sorted_x = numpy.argsort(x)
	size_one = 0
	edges_cut = 0
	swaps = 0
	nodes_one = []
	sizes_one = []
	den = 0

	for t in range(G.num_snaps()):
		nodes_one.append({})
		sizes_one.append(0)

	for i in range(x.shape[0]):
		(v,t) = G.index(sorted_x[i])
		den = den - sizes_one[t] * (G.size() - sizes_one[t])
		sizes_one[t] = sizes_one[t] + 1
		den = den + sizes_one[t] * (G.size() - sizes_one[t])

		nodes_one[t][v] = True
		
		for u in G.graphs[t].neighbors(v):
			if u not in nodes_one[t]:
				edges_cut = edges_cut + G.graphs[t].edge[v][u]["weight"]
			else:
				edges_cut = edges_cut - G.graphs[t].edge[v][u]["weight"]

		if t+1 < G.num_snaps():
			if v not in nodes_one[t+1]:
				swaps = swaps + G.swap_cost()
			else:
				swaps = swaps - G.swap_cost()

		if t > 0:
			if v not in nodes_one[t-1]:
				swaps = swaps + G.swap_cost()
			else:
				swaps = swaps - G.swap_cost()
		
		if den > 0:
			score = float(edges_cut + swaps) / den
		else:
			score = sys.float_info.max

		if score <= best_score:
			best_score = score
			best = i
			best_edges_cut = edges_cut
			best_swaps = swaps

	vec = numpy.zeros(G.size() * G.num_snaps())

	for i in range(x.shape[0]):
		if i <= best:
			vec[sorted_x[i]] = -1.
		else:
			vec[sorted_x[i]] = 1.

	return {"cut": vec, "score": best_score, "edges": best_edges_cut, "swaps": swaps}

def create_laplacian_matrix(G):
	row = []
	column = []
	value = []

	for t in range(G.num_snaps()):
		Lg = networkx.laplacian_matrix(G.snap(t))
		for (i,j) in zip(*scipy.nonzero(Lg)):
			row.append(G.size()*t + i)
			column.append(G.size()*t + j)
			
			if i != j:
				value.append(Lg[i,j])
			else:
				if t > 0 and t < G.num_snaps() - 1:
					value.append(Lg[i,j] + 2 * G.swap_cost())
				else:	
					value.append(Lg[i,j] + 1 * G.swap_cost())
	
	for t in range(G.num_snaps()-1):
		for v in range(G.size()):
			row.append(t*G.size() + v)
			column.append((t+1)*G.size() + v)
			value.append(-1 * G.swap_cost())
			
			column.append(t*G.size() + v)
			row.append((t+1)*G.size() + v)
			value.append(-1 * G.swap_cost())


	sz = G.num_snaps() * G.size()
	return scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)

def create_c_matrix(G):
	row = []
	column = []
	value = []
	
	for t in range(G.num_snaps()):
		for i in range(G.size()):
			for j in range(G.size()):
				row.append(t*G.size() + i)
				column.append(t*G.size() + j)
				
				if i == j:
					value.append(G.size()-1)
				else:
					value.append(-1.)
	
	sz = G.num_snaps() * G.size()
	
	return scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)

#	n = G.size()
#	C = numpy.zeros((G.num_snaps()*n,G.num_snaps()*n))

#	C = C + (n-1) * numpy.diag(numpy.ones(n*G.num_snaps()))

#	for ti in range(G.num_snaps()):
#		for tj in range(G.num_snaps()):
#			for vi in range(n):
#				for vj in range(n):
#					if ti != tj:
#						C[ti*n+vi][tj*n+vj] = 0
#					else:
#						if vi != vj:
#							C[ti*n+vi][tj*n+vj] = -1.

#	return C

def eigen_vec_cut_prod(G):
	L = create_laplacian_matrix(G)
	C = create_c_matrix(G)
	
	M = scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(C, L), C)
	(eigvals, eigvecs) = scipy.sparse.linalg.eigs(M, k=G.num_snaps()+1, which='SM')
	x = scipy.sparse.csr_matrix.dot(eigvecs[:,0], C)
	
#	M = scipy.dot(scipy.dot(C, L), C)
#	(eigvals, eigvecs) = scipy.linalg.eigh(M)
#	x = scipy.dot(eigvecs[:,numpy.argsort(eigvals.real)[G.num_snaps()]], C)
	
	return x.real

def eigen_vec_cut_inv(G):
	#L = networkx.laplacian_matrix(G.g, G.nodes()).todense().astype(float)
	L = create_laplacian_matrix(G).todense()
	C = create_c_matrix(G).todense()
	sqrtC = sqrtm(C)
	isqrtC = sqrtm(scipy.linalg.pinv(C))
	M = numpy.dot(numpy.dot(isqrtC, L), isqrtC)
	(eigvals, eigvecs) = scipy.linalg.eigh(M, eigvals=(G.num_snaps(), G.num_snaps()))

	x = numpy.asarray(numpy.dot(eigvecs[:,0], sqrtC))[0]
	
	return x.real

def power_method(mat, maxit):
	vec = numpy.ones(mat.shape[0])
	vec = vec/numpy.linalg.norm(vec)

	for i in range(maxit):
		vec = scipy.sparse.csr_matrix.dot(vec, mat)
		vec = vec/numpy.linalg.norm(vec)
 
	return numpy.asarray(vec)

def fast_eigen_vec_cut(G,niter=0):
	L = create_laplacian_matrix(G)
	C = create_c_matrix(G)
	
	M = C-L

	if niter == 0:
		(eigvals, eigvecs) = scipy.sparse.linalg.eigs(M, k=1, which='LM')
		x = scipy.sparse.csr_matrix.dot(eigvecs[:,0], C)
	else:
		x = power_method(M, niter)
		x = scipy.sparse.csr_matrix.dot(x, C)
	
	return x.real


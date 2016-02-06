import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lib.netpros import *
from lib.static import *
from mpl_toolkits.mplot3d import axes3d
import numpy
from matplotlib.mlab import griddata
from matplotlib import cm
from lib.syn import *
import time
import sys

def size_time_experiment(sizes, balance, sparsity, energy, noise, num):
	res_time = []

	for s in range(len(sizes)):
		res_t = []
		for i in range(num):
			(G,F,cut) = synthetic_graph(sizes[s], 3*sizes[s], sparsity, energy, balance, noise)

			j = 0
			ind = {}
			for v in G.nodes():
				ind[v] = j
				j = j + 1

			k = int(len(G.edges())*sparsity)
			start_time = time.time()
			c = one_d_search(G, F, k, ind)
			time_slow = time.time()-start_time

			start_time = time.time()
			c = fast_search(G, F, k, 5, ind)
			time_5 = time.time()-start_time
			
			start_time = time.time()
			c = fast_search(G, F, k, 20, ind)
			time_20 = time.time()-start_time
			
			start_time = time.time()
			c = fast_search(G, F, k, 50, ind)
			time_50 = time.time()-start_time

			res_t.append([time_slow, time_5, time_20, time_50])

		r = numpy.mean(numpy.array(res_t), axis=0)
		res_time.append(r)

	return numpy.array(res_time)

def sparsity_acc_experiment(sparsity, size, balance, energy, noise, num):
	res = []

	for s in range(len(sparsity)):
		res_a = []
		for i in range(num):
			(G,F,k) = synthetic_graph(size, 3*size, sparsity[s], energy, balance, noise)
			
			j = 0
			ind = {}
			for v in G.nodes():
				ind[v] = j
				j = j + 1

			c = one_d_search(G, F, k, ind)
			acc_slow = c["energy"]
			
			c = fast_search(G, F, k, 5, ind)
			acc_5 = c["energy"]
			
			c = fast_search(G, F, k, 20, ind)
			acc_20 = c["energy"]
			
			c = fast_search(G, F, k, 50, ind)
			acc_50 = c["energy"]

			res_a.append([acc_slow, acc_5, acc_20, acc_50])
		
		r = numpy.mean(numpy.array(res_a), axis=0)
		res.append(r)

	return numpy.array(res)

def noise_acc_experiment(noise, size, sparsity, energy, balance, num):
	res = []

	for s in range(len(noise)):
		res_a = []
		for i in range(num):
			(G,F,k) = synthetic_graph(size, 3*size, sparsity, energy, balance, noise[s])
			
			j = 0
			ind = {}
			for v in G.nodes():
				ind[v] = j
				j = j + 1
			
			c = one_d_search(G, F, k, ind)
			acc_slow = c["energy"]
			
			L = networkx.laplacian_matrix(G)
			c = fast_search(G, F, k, 5, ind)
			acc_5 = c["energy"]
			
			c = fast_search(G, F, k, 20, ind)
			acc_20 = c["energy"]
			
			c = fast_search(G, F, k, 50, ind)
			acc_50 = c["energy"]

			res_a.append([acc_slow, acc_5, acc_20, acc_50])
		
		r = numpy.mean(numpy.array(res_a), axis=0)
		res.append(r)

	return numpy.array(res)

def energy_acc_experiment(energy, size, sparsity, noise, balance, num):
	res = []

	for s in range(len(energy)):
		res_a = []
		for i in range(num):
			(G,F,k) = synthetic_graph(size, 3*size, sparsity, energy[s], balance, noise)
			
			j = 0
			ind = {}
			for v in G.nodes():
				ind[v] = j
				j = j + 1
			
			c = one_d_search(G, F, k, ind)
			acc_slow = c["energy"]
			
			c = fast_search(G, F, k, 5, ind)
			acc_5 = c["energy"]
			
			c = fast_search(G, F, k, 20, ind)
			acc_20 = c["energy"]
			
			c = fast_search(G, F, k, 50, ind)
			acc_50 = c["energy"]

			res_a.append([acc_slow, acc_5, acc_20, acc_50])
		
		r = numpy.mean(numpy.array(res_a), axis=0)
		res.append(r)

	return numpy.array(res)

def plot_size_time_experiment(results, sizes, output_file_name):
	plt.clf()
	
	ax = plt.subplot(111)
	
	ncol=2
	ax.plot(sizes, results[:,0], marker="x", color="b", label="SWT", markersize=15)
	ax.plot(sizes, results[:,1], marker="o", color="r", label="FSWT-5", markersize=15)
	ax.plot(sizes, results[:,2], marker="o", color="g", label="FSWT-20", markersize=15)
	ax.plot(sizes, results[:,3], marker="o", color="k", label="FSWT-50", markersize=15)
	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper center', prop={'size':20}, ncol=ncol)
	ax.set_ylabel('time (sec.)', fontsize=30)
	ax.set_xlabel('#vertices', fontsize=30)
	plt.rcParams['xtick.labelsize'] = 20 
	plt.rcParams['ytick.labelsize'] = 20
	ax.set_xlim([180,1020])
	ax.set_ylim([0.01,50000])
	ax.set_yscale('log')

	plt.savefig(output_file_name, dpi=200, bbox_inches='tight')

def plot_sparsity_acc_experiment(results, sparsity, output_file_name):
	plt.clf()

	ncol=2
	ax = plt.subplot(111)
	width = 0.04       # the width of the bars)
	ax.bar(numpy.array(sparsity)-2*width, results[:,0], width, color='b', label="SWT")
	ax.bar(numpy.array(sparsity)-width, results[:,1], width, color='r', label="FSWT-5")
	ax.bar(numpy.array(sparsity), results[:,2], width, color='g', label="FSWT-20")
	ax.bar(numpy.array(sparsity)+width, results[:,3], width, color='k', label="FSWT-50")
	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper center', prop={'size':20}, ncol=ncol)
	ax.set_ylabel(r'L$_2$ energy', fontsize=30)
	ax.set_xlabel('sparsity', fontsize=30)
	plt.rcParams['xtick.labelsize'] = 20 
	plt.rcParams['ytick.labelsize'] = 20
	ax.set_xlim([0,1])
	ax.set_ylim([0,150])
	
	plt.savefig(output_file_name, dpi=200, bbox_inches='tight')

def plot_noise_acc_experiment(results, noise, output_file_name):
	plt.clf()
	
	ax = plt.subplot(111)
	ncol=2
	width = 0.04       # the width of the bars)
	rects1 = ax.bar(numpy.array(noise)-2*width, results[:,0], width, color='b', label="SWT")
	rects2 = ax.bar(numpy.array(noise)-width, results[:,1], width, color='r', label="FSWT-5")
	rects3 = ax.bar(numpy.array(noise), results[:,2], width, color='g', label="FSWT-20")
	rects4 = ax.bar(numpy.array(noise)+width, results[:,3], width, color='k', label="FSWT-50")

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper center', prop={'size':20}, ncol=ncol)
	ax.set_ylabel(r'L$_2$ energy', fontsize=30)
	ax.set_xlabel('noise', fontsize=30)
	plt.rcParams['xtick.labelsize'] = 20 
	plt.rcParams['ytick.labelsize'] = 20
	ax.set_xlim([0,1])
	ax.set_ylim([0,150])
	
	plt.savefig(output_file_name, dpi=200, bbox_inches='tight')

def plot_energy_acc_experiment(results, energy, output_file_name):
	plt.clf()
	ncol=2
	ind = numpy.array(list(range(5)))
	ax = plt.subplot(111)
	width = 0.2      # the width of the bars)
	rects1 = ax.bar(ind-width, results[:,0], width, color='b', label="SWT", log=True)
	rects2 = ax.bar(ind, results[:,1], width, color='r', label="FSWT-5", log=True)
	rects3 = ax.bar(ind+width, results[:,2], width, color='g', label="FSWT-20", log=True)
	rects4 = ax.bar(ind+2*width, results[:,3], width, color='k', label="FSWT-50", log=True)

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper left', prop={'size':20}, ncol=ncol)
	ax.set_ylabel(r'L$_2$ energy', fontsize=30)
	ax.set_xlabel(r'L$_2$ energy (data)', fontsize=30)
	plt.rcParams['xtick.labelsize'] = 20 
	plt.rcParams['ytick.labelsize'] = 20
	ax.set_yscale('log')
	ax.set_xticks(ind + width)
	ax.set_xticklabels((r'10$\mathregular{^1}$', r'10$\mathregular{^2}$', r'10$\mathregular{^3}$', r'10$\mathregular{^4}$', r'10$\mathregular{^5}$'))
	ax.set_ylim(0.1,1000000)
	
	plt.savefig(output_file_name, dpi=200, bbox_inches='tight')

def compression_experiment_dynamic(G, FT, algs):
	comp_ratios = (.05, .10, .15, .20, .25, .30)

	results = {}
	for alg in algs:
		results[a.name()] = []

		for i in range(len(FT)):
			results[alg.name()].append([])
			alg.set_graph(G)
			tr = alg.transform(FT[i])

			for r in comp_ratios:
				size = int(FT[i].size * r)
				appx_tr = alg.drop_frequency(tr, size)
				appx_FT = alg.inverse(appx_tr)

				results[alg.name()][i].append([L2(FT[i], appx_FT), L1(FT[i], appx_FT)])

		results[alg.name()] = numpy.array(results[alg.name()]) 
		results[alg.name()] = numpy.mean(results[alg.name()], axis=0)
              
	return results

def plot_compression_experiments(results, comp_ratios, output_file_name, max_y):
	plt.clf()
	
	ax = plt.subplot(111)
	ncol = 3
	ax.plot(comp_ratios, results["FSWT"], marker="o", color="r", label="FSWT", markersize=15)
	ax.plot(comp_ratios, results["FT"], marker="*", color="c", label="FT", markersize=15)
	
	if "SWT" in results:
		ax.plot(comp_ratios, results["SWT"], marker="x", color="b", label="SWT", markersize=15)
	
	ax.plot(comp_ratios, results["GWT"], marker="s", color="g", label="GWT", markersize=15)
	
	if "HWT" in results:
		ax.plot(comp_ratios, results["HWT"], marker="v", color="y", label="HWT", markersize=15)
	
	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc='upper center', prop={'size':20}, ncol=ncol)
	ax.set_ylabel(r'L$_2$ error', fontsize=30)
	ax.set_xlabel('size', fontsize=30)
	plt.rcParams['xtick.labelsize'] = 20 
	plt.rcParams['ytick.labelsize'] = 20
	ax.set_xlim(0.,0.35)
	ax.set_ylim(0., max_y)

	plt.savefig(output_file_name, dpi=200, bbox_inches='tight')

def compression_experiment_static(G, F, algs, comp_ratios):
	results = {}
	times = {}

	for alg in algs:
		results[alg.name()] = []
		times[alg.name()] = []
			
		for i in range(len(F)):
			results[alg.name()].append([])
			times[alg.name()].append([])
			
			for r in range(len(comp_ratios)):
				print("alg = ", alg, " i = ", i, " r = ", r)
				sys.stdout.flush()
				start_time = time.time()
				alg.set_graph(G)
				tr = alg.transform(F[i])
				size = int(F[i].size * comp_ratios[r])
				appx_tr = alg.drop_frequency(tr, size)
				appx_F = alg.inverse(appx_tr)
				t = time.time()-start_time

				times[alg.name()][i].append([t])
				results[alg.name()][i].append(L2(F[i], appx_F))

		results[alg.name()] = numpy.array(results[alg.name()]) 
		results[alg.name()] = numpy.mean(results[alg.name()], axis=0)

		times[alg.name()] = numpy.array(times[alg.name()]) 
		times[alg.name()] = numpy.mean(times[alg.name()], axis=0)
              
	return results, times

def compression_experiment_static_weighted(G, F, algs):

	results = {}
	for alg in algs:
		results[alg.name()] = []

		for i in range(len(F)):
			results[alg.name()].append([])
			set_weight_graph(G, F[i])
			alg.set_graph(G)
			tr = alg.transform(F[i])

			for r in comp_ratios:
				size = int(F[i].size * r)
				appx_tr = alg.drop_frequency(tr, size)
				appx_F = alg.inverse(appx_tr)

				results[alg.name()][i].append([L2(F[i], appx_F), L1(F[i], appx_F)])

		results[alg.name()] = numpy.array(results[alg.name()]) 
		results[alg.name()] = numpy.mean(results[alg.name()], axis=0)
              
	return results

def experiment_lambda_k(G, F, lambdas, edges, ratio):
	X = []
	Y = []
	Z = []

	for l in range(len(lambdas)):
		for e in range(len(edges)):
			for i in range(len(F)):
				res = []
				alg = OptWavelets(edges[e], lambdas[l])
				alg.set_graph(G)
				tr = alg.transform(F[i])
				size = int(F[i].size * ratio)
				appx_tr = alg.drop_frequency(tr, size)
				appx_F = alg.inverse(appx_tr)
				res.append(L2(F[i], appx_F))
			
			res = numpy.array(res)
			Z.append(numpy.mean(res))
			Y.append(lambdas[l])
			X.append(edges[e])
	return X,Y,Z

def plot_experiment_lambda_k(X,Y,Z, output_file_name):
	plt.clf()
	ax = plt.subplot(111, projection='3d')
	
	ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	ax.set_zlabel(r'L$_2$ error', fontsize=15)
	ax.set_xlabel('k', fontsize=15)
	ax.set_ylabel('lambda', fontsize=15)
	box = ax.get_position()
	
	Xi = numpy.linspace(min(X), max(X))
	Yi = numpy.linspace(min(Y), max(Y))
	
	x, y = numpy.meshgrid(Xi, Yi)
	z = griddata(X, Y, Z, Xi, Yi)
	
#	surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
#	                       linewidth=1, antialiased=False)
	surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)

	ax.set_zlim3d(numpy.min(Z), numpy.max(Z))
	plt.colorbar(surf)

	plt.savefig(output_file_name)

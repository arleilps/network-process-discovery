import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lib.netpros import *

def compression_experiment(G, FT, algs):
	comp_ratios = (.05, .10, .15, .20, .25, .30)

	results = {}
	for alg in algs:
		results[alg.name()] = []

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

def plot_compression_experiments(results, output_file_name):
	comp_ratios = (.05, .10, .15, .20, .25, .30)
	plt.clf()
	ax = plt.subplot(111)
	i = 0
	markers = ["o", "x", "*", "+", "s", "v"]
	for alg in results.keys():
		ax.semilogy(comp_ratios, results[alg][:,0], label=alg, marker=markers[i], markersize=10, basey=2)
		i = i + 1

#plt.loglog(comp_ratios, results[alg][:,0], basex=2, basey=2, label=alg)

	ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
	ax.set_ylabel('L2 error', fontsize=20)
	ax.set_xlabel('size', fontsize=20)
	#ax.set_ylim([math.pow(2,-6), 2])
	ax.set_xlim([0,.35])
	box = ax.get_position()
	plt.rcParams['xtick.labelsize'] = 10 
	plt.rcParams['ytick.labelsize'] = 10
	ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])

	plt.savefig(output_file_name)

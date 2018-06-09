import matplotlib.pyplot as plt
import numpy as np

def colourPlot(nnets):
	colors = np.zeros((13,13,3))
	for i in range(13):
		for j in range(i,13):
			for k in range(4):
				for l in range(4):
					if not k==l:
						cards = np.zeros(52)
						cards[4*i+k] = 1
						cards[4*j+l] = 1
						colors[12- i,12 - j,:] += nnets.policyValue(cards,np.zeros((2,4,5,2)),np.zeros(52))/12

	for i in range(13):
		for j in range(i):
			for k in range(4):
				cards = np.zeros(52)
				cards[4*i+k] = 1
				cards[4*j+k] = 1
				colors[12- i,12 - j,:] += nnets.policyValue(cards,np.zeros((2,4,5,2)),np.zeros(52))/4

	plt.imshow(colors)
	ax = plt.gca()
	ax.set_xticks(np.arange(13))
	ax.set_yticks(np.arange(13))
	ax.set_xticklabels(("A","K","Q","J","10","9","8","7","6","5","4","3","2"))
	ax.set_yticklabels(("A","K","Q","J","10","9","8","7","6","5","4","3","2"));
	ax.set_xticks(np.arange(-.5, 13, 1), minor=True);
	ax.set_yticks(np.arange(-.5, 13, 1), minor=True);
	ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
	plt.savefig("colour.png")

import matplotlib.pyplot as plt
import numpy as np

def colourPlot(nnets):
	colours = np.zeros((13,13,3))
	average = np.zeros(3)
	for i in range(52):
		for j in range(52):
			cards = np.zeros(52)
			cards[i] = 1
			cards[j] = 1
			p,_ = nnets.policyValue(cards,np.zeros((2,4,5,2)),np.zeros(52))
			average += p/(52*52)
	for i in range(13):
		for j in range(i,13):
			avColour = np.zeros(3)
			for k in range(4):
				for l in range(4):
					if not k==l:
						cards = np.zeros(52)
						cards[4*i+k] = 1
						cards[4*j+l] = 1
						colour,_ = nnets.policyValue(cards,np.zeros((2,4,5,2)),np.zeros(52))
						avColour += colour/12
			avColour = avColour/average
			avColour = avColour**2
			colours[12- i,12 - j,:] = avColour/np.sum(avColour)

	for i in range(13):
		for j in range(i):
			avColour = np.zeros(3)
			for k in range(4):
				cards = np.zeros(52)
				cards[4*i+k] = 1
				cards[4*j+k] = 1
				colour,_= nnets.policyValue(cards,np.zeros((2,4,5,2)),np.zeros(52))
				avColour += colour/4
			avColour = avColour/average
			avColour = avColour**2
			colours[12- i,12 - j,:] = avColour/np.sum(avColour)

	plt.imshow(colours)
	ax = plt.gca()
	ax.xaxis.tick_top()
	ax.set_xlabel('Suited')    
	ax.xaxis.set_label_position('top') 
	ax.set_ylabel('Unsuited')    
	ax.yaxis.set_label_position('left') 
	ax.set_xticks(np.arange(13))
	ax.set_yticks(np.arange(13))
	ax.set_xticklabels(("A","K","Q","J","10","9","8","7","6","5","4","3","2"))
	ax.set_yticklabels(("A","K","Q","J","10","9","8","7","6","5","4","3","2"));
	ax.set_xticks(np.arange(-.5, 13, 1), minor=True);
	ax.set_yticks(np.arange(-.5, 13, 1), minor=True);
	ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
	plt.savefig("colour.png")

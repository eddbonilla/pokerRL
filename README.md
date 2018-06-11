# pokerRL

To run the code:
1.- compile the cython files by running:

  python3 setup.py build_ext --inplace
  
2.- inside the python 3 console import the Training clas from learning Grounds

  from learningGrounds import Training
  
3.- Create an instance of the class:

  agentInstance= Training(game='leduc')  #For leduc poker
  
  agentInstance= Training(game='holdem') #for Limit texas hold em
  
4.- To train the neural networks for some number of episodes use:

  agentInstance.doTraining(numEpisodes)
  

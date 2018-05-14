#Code adapted from https://github.com/suragnair/alpha-zero-general (needs heavy adaptation for imperfect information games)

import math
import numpy as np
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, fnet,gnet, args):
        self.game = game
        self.fnet = fnet
        self.gnet = gnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def treeStrategy(self, playerCards, publicHistory, publicCards, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        initialState.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        estimOpponentCards= self.gnet.predict(playerCards, publicHistory, publicCards) # gives a guess of the opponent cards, we can change this to be the actual cards
        
        for i in range(self.args.numMCTSSims):
            self.search(estimOpponentCards, playerCards, publicHistory, publicCards)

        s = self.game.stringRepresentation(initialState) #This is to get a representation of the state of the game
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())] #Here you count the number of times that action a was taken in state s

        counts = [x**(1./temp) for x in counts] #Add a temperature factor to emphasize max_a(Nsa) or to sample uniform
        probs = [x/float(sum(counts)) for x in counts] #normalize
        return probs #return pi,tree strategy


    def search(self, opponentCards, playerCards, publicHistory, publicCards):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current initialState
        """

        s = self.game.stringRepresentation(opponentCards, playerCards, publicHistory, publicCards) #gives a code for the state of the game, it is a unique string of characters that can be placed in a python dictionary

        if s not in self.Es: # check if s is a known terminal state
            self.Es[s] = self.game.getGameEnded(opponentCards, playerCards, publicHistory, publicCards, 1)
        if self.Es[s]!=0: # This means that the game has terminated (what about this return value though?)
            # terminal node
            return -self.Es[s]

        if s not in self.Ps: #Have we been on this state during the search? if yes, then no need to reevaluate it
            # leaf node
            self.Ps[s], v = self.nnet.predict(opponentCards, playerCards, publicHistory, publicCards)
            valids = self.game.getValidMoves(publicHistory, 1)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        opponentCards, playerCards, publicHistory, publicCards = self.game.getNextState(opponentCards, playerCards, publicHistory, publicCards, 1, a)
        #next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(opponentCards, playerCards, publicHistory, publicCards)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
return -v
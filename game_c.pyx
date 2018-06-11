import numpy as np
cimport numpy as np

cdef class Game:
	cpdef void resetGame(self):
		pass
	cpdef object copy(self):
		pass
	cdef str playerInfoStringRepresentation(self):
		pass
	cdef str publicInfoStringRepresentation(self):
		pass
	cdef void finishGame(self,int playerfolded = 0):
		pass
	cpdef int getPlayer(self):
		pass
	cdef void endRound(self):
		pass
	cpdef np.ndarray getPlayerCards(self):
		pass
	cpdef np.ndarray getOpponentCards(self):
		pass
	cpdef np.ndarray getPlayerStates(self):
		pass
	cpdef np.ndarray getPublicCards(self):
		pass
	cpdef int getPot(self):
		pass
	cdef void setPlayer(self,int player):
		pass
	cpdef int getRound(self):
		pass
	cpdef np.ndarray getPublicHistory(self):
		pass
	cpdef np.ndarray getOutcome(self):
		pass
	cpdef int isFinished(self):
		pass
	cpdef int action(self,int action=-1, np.ndarray strategy=None):
		pass
	cpdef int sampleOpponent(self,object nnets):
		pass
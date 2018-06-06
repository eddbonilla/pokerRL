import numpy as np
cimport numpy as np

cdef class Game:
	cdef int pot,player,round,bet,raisesInRound,finished
	cdef np.ndarray winnings, playersCardsArray, publicCardsArray, history
	cdef int[:] winnings_view,  publicCardsArray_view
	cdef int[:,:] playerCardsArray_view
	cdef int[:,:,:,:] history_view

	cpdef void resetGame(self)
	cpdef object copy(self)
	cdef str playerInfoStringRepresentation(self)
	cdef str publicInfoStringRepresentation(self)
	cdef void finishGame(self,int playerfolded = *)
	cpdef int getPlayer(self)
	cdef void endRound(self)
	cpdef np.ndarray getPlayerCards(self)
	cpdef np.ndarray getOpponentCards(self)
	cpdef np.ndarray getPlayerStates(self)
	cpdef np.ndarray getPublicCards(self)
	cpdef int getPot(self)
	cdef void setPlayer(self,int player)
	cpdef int getRound(self)
	cpdef np.ndarray getPublicHistory(self)
	cpdef np.ndarray getOutcome(self)
	cpdef int isFinished(self)
	cpdef int action(self,int action=*, np.ndarray strategy=*)
	cpdef int sampleOpponent(self,object nnets)
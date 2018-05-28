import numpy as np
cimport numpy as np

cdef class LeducGame:
	cdef int dealer,pot,player,round,bet,raisesInRound,finished
	cdef np.ndarray cards, winnings, playersCardsArray, publicCardArray, history
	cdef int[:] cards_view, winnings_view, publicCardArray_view
	cdef int[:,:] playersCardsArray_view,
	cdef int[:,:,:,:] history_view

	cpdef void resetGame(self)
	cdef object copy(self)
	cdef str playerInfoStringRepresentation(self)
	cdef str publicInfoStringRepresentation(self)
	cdef void finishGame(self,int playerfolded = *)
	cpdef int getPlayer(self)
	cdef void endRound(self)
	cpdef np.ndarray getPlayerCard(self)
	cpdef np.ndarray getOpponentCard(self)
	cpdef np.ndarray getPlayerStates(self)
	cpdef np.ndarray getPublicCard(self)
	cpdef int getPot(self)
	cpdef void setOpponentCard(self,int card)
	cpdef void setPlayerCard(self,int card)
	cdef void setPlayer(self,int player)
	cpdef np.ndarray getPublicHistory(self)
	cpdef np.ndarray getOutcome(self)
	cpdef int isFinished(self)
	cpdef np.ndarray regulariseOpponentEstimate(self, np.ndarray estimate)
	cpdef int action(self,int action=*, np.ndarray strategy=*)
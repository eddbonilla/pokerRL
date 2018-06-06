import numpy as np
cimport numpy as np
from game_c cimport Game

cdef class LeducGame(Game):
	cdef np.ndarray cards
	cdef int[:] cards_view

	cpdef void setOpponentCard(self,int card)
	cpdef void setPlayerCard(self,int card)
	cpdef void setPublicCard(self,int card)
	cpdef np.ndarray regulariseOpponentEstimate(self,np.ndarray estimate)


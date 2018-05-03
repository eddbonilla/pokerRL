def SelfPlay(eta):
	game = LeducGame()
 	treeStrategies = []
 	rangeMemory = []
 	while game.isUnfinished():
 		player = game.getPlayer()
 		publicState = game.getPublicState()
 		playerState = game.getPlayerState()
 		opponentsState = game.getOpponentsState()
 		rangeMemory.append((publicState,playerState,opponentsState))
 		if np.random.random() > eta:
 			strategy = averageStrategy(player,playerState,publicState)
 		
 		else:
 			strategy = treeStrategy(playerState,publicState)
 			treeStrategies.append((strategy,playerState,publicState))
 		action = sampleFromStrategy(strategy)
 		game.playMove(action)
 	v = game.getOutcome()
 	#Add rangeMemory to g_Reservoir
 	#Add treeStrategies,v to f_Reservoir
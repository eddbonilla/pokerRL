def SelfPlay(eta):
	game = LeducGame()
 	treeStrategies = []
 	rangeMemory = []
 	playerStates = game.getPlayerStates()
 	v = [0,0]
 	while not game.isFinished():
 		player = game.getPlayer()
 		publicHistory = game.getPublicHistory()
 		playerState = playerStates[player]
 		opponentState = playerStates[(player+1)%2]
 		rangeMemory.append((publicHistory,playerState,opponentsState))
 		if np.random.random() > eta:
 			strategy = averageStrategy(player,playerState,publicHistory)
 		
 		else:
 			strategy = treeStrategy(playerState,publicHistory)
 			treeStrategies.append((strategy,playerState,publicHistory))
 		action,bet = game.action(strategy)
 		v[player]-= bet
 	v += game.getOutcome()
 	#Add rangeMemory to g_Reservoir
 	#Add treeStrategies,v to f_Reservoir
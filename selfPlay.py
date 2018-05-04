def averageStrategy(player,playerState,publicHistory,publicCards):
	#Test code. To be replaced by neural net
	return [0.5,0.4,0.1] 

def SelfPlay(eta):
	game = LeducGame()
	treeStrategies = []
	fReservoir = []
	gReservoir = []
	allPlayersCards = game.getPlayerStates()
	ante = game.getAnte()
	v = np.array([-ante,-ante],dtype = float)
	while not game.isFinished():
		player = game.getPlayer()
		publicHistory, publicCards = game.getPublicState()
		playerCards = allPlayersCards[player]
		opponentCards = np.delete(allPlayersCards,player,axis=0)
		gReservoir.append((publicHistory,publicCards,playerCards,opponentCards))
		if np.random.random() > eta:
			strategy = averageStrategy(player,playerCards,publicHistory,publicCards)
		
		else:
			strategy = treeStrategy(playerState,publicHistory)
			treeStrategies.append((strategy,playerCards,publicHistory,publicCards))
		action,bet = game.action(strategy)
		v[player]-= bet
		#print(action,bet,v)
	v += game.getOutcome()
	for strt,plC,pbH,pbC in treeStrategies:
	 fReservoir = (strt,plC,pbH,pbC,v)
	return fReservoir,gReservoir
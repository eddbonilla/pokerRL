def averageStrategy(player,playerState,publicHistory,publicCards):
	#Test code. To be replaced by neural net
	return [0.5,0.4,0.1] 

def SelfPlay(eta,nnets):
	game = LeducGame()
	cache = []
	allPlayersCards = game.getPlayerStates()
	ante = game.getAnte()
	v = np.array([-ante,-ante],dtype = float)
	while not game.isFinished():

		player = game.getPlayer()

		treeStrategy = treeStrategy(game)
		averageStrategy,_ = nnets.policyValue(game.getPlayerCard(),game.getPublicHistory(),game.getPublicCard())

		strategy = (1-eta)*averageStrategy + eta * treeStrategy 

		cache.append((treeStrategy,player,game.getPublicHistory(),game.getPublicCard(),game.getPlayerCard(),getOpponentCard(),game.getPot(),v[player]))
		action,bet = game.action(strategy)
		v[player]-= bet
		#print(action,bet,v)
	v += game.getOutcome()
	for strt,plr,pbH,pbC,plC,opC,pot,v_0 in cache:
		fReservoir.append(pbH,pbC,plC,strt,(v[plr]-v_0)/pot)			#Store (publicHistory, publicCard, playerCard, treeStrategy, v)
		gReservoir.append(pbH,pbC,opC)									#Store (publicHistory, publicCard, opponentCard)
	return fReservoir,gReservoir

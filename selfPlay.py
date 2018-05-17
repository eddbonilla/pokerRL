def averageStrategy(player,playerState,publicHistory,publicCards):
	#Test code. To be replaced by neural net
	return [0.5,0.4,0.1] 

def SelfPlay(eta, numMCTSSims,nnets,cpuct):
	game = LeducGame()
	cache = []
	allPlayersCards = game.getPlayerStates()
	ante = game.getAnte()
	v = np.array([-ante,-ante],dtype = float)
	trees = [MCTS(nnets, numMCTSSims, cpuct), MCTS(nnets, numMCTSSims, cpuct)]             #Index labels player
	while not game.isFinished():

		player = game.getPlayer()

		averageStrategy, treeStrategy = trees[player].Strategy(game)
		print("avStrat =" + str(averageStrategy) + "\n treeStrat =" + str(treeStrategy))
		strategy = (1-eta)*averageStrategy + eta * treeStrategy 
		dict = {
					"treeStrategy" :treeStrategy,
					"player": player,
					"publicHistory": game.getPublicHistory(),
					"publicCard"  : game.getPublicCard(),
					"playerCard"  : game.getPlayerCard(),
					"opponentCard"    : game.getOpponentCard(),
					"pot"         : game.getPot(),
					"moneyBet"    : v[player]
				}
		cache.append(dict)
		action,bet = game.action(strategy)
		v[player]-= bet
		print(action,bet,v)
	v += game.getOutcome()
	
	inputs = []
	opponentCards = []
	policies = []
	vs = []


	for dict in cache:
		v = (v[dict["player"]] - dict["moneyBet"])/float(dict["pot"])
		inputs.append(nnets.preprocessInput(dict["playerCard"],dict["publicHistory"],dict["publicCard"]))
		opponentCards.append(dict["opponentCard"])
		policies.append(dict["treeStrategy"])
		vs.append[v]

	return inputs, opponentCards, policies, vs

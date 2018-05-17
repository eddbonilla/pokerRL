
class SelfPlay:


	def runGame(eta, numMCTSSims,nnets,cpuct):
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
	
		inputs = np.zeros((len(cache), game.params["historySize"] + game.params["handSize"] + game.params["publicCardSize"]))
		opponentCards = np.zeros((len(cache),game.params["handSize"]))
		policies = np.zeros((len(cache),game.params["actionSize"]))
		vs = np.zeros((len(cache),1))


		for i in len(cache):
			dict = cache[i]
			v = (v[dict["player"]] - dict["moneyBet"])/float(dict["pot"])
			inputs[i,:] = nnets.preprocessInput(dict["playerCard"],dict["publicHistory"],dict["publicCard"]))
			opponentCards[i,:] = dict["opponentCard"])
			policies[i,:] = dict["treeStrategy"])
			vs[i,:]= v

		return inputs, opponentCards, policies, vs

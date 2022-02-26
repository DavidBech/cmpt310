# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        #util.pause()
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        if successorGameState.isWin():
            return 1 << 31
        elif successorGameState.isLose():
            return -1 * (1 << 30)

        newPos = successorGameState.getPacmanPosition()
        currentPos = currentGameState.getPacmanPosition()

        newFood = successorGameState.getFood()
        currentFood = currentGameState.getFood()

        newGhostStates = successorGameState.getGhostStates()
        currentGhostStates = currentGameState.getGhostStates()

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currentScaredTimes = currentGameState.getGhostStates()

        # Weights
        ghostDistance_weight = 500
        eatDot_weight = 100         # Weight added when dot can be eaten
        eastPrefference_weight = 1  # Weight added when going east and food is east
        dontStop_weight = 10        # Weight subtracted when stop action is taken

        score = 0                   # Base score that is updated with weights
        # Avoid Death
        for ghost in currentGhostStates:
            distToPacman = manhattanDistance(ghost.getPosition(), newPos)
            if distToPacman <= 1: # ghost on pacman
                score -= ghostDistance_weight # do not pick this 
            if distToPacman < 4: # avoid tiles near ghost
                score -= distToPacman

        # Eat Food
        distToClosestFood = 1 << 30 # Used to bring packman towards food
        foodEastOfPacman = 0 # Used to keep trak of food to the east of packman
        for foodPellet in currentFood.asList():
            if foodPellet[0] > currentPos[0]: # food is farther east
                foodEastOfPacman +=1
            # find distance to food pellet
            distToFood = manhattanDistance(newPos, foodPellet)
            # update min distance if neccessary
            distToClosestFood = min(distToClosestFood, distToFood)
            if distToFood == 0: # move eats food
                score += eatDot_weight
        score -= distToClosestFood # move pacman closer to closest food

        # Go EAST First to prevent pacman going back and forth 
        if foodEastOfPacman and action == Directions.EAST:
            score += eastPrefference_weight

        # Avoid stopping
        if action == Directions.STOP:
            score -= dontStop_weight
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        PACMAN = 0
        def minmax(state, depth, currentAgent) -> int:
            # if leaf/end of search return the evaluation of the state
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            # set up values based on current agent 
            if currentAgent == PACMAN:
                curValue = float("-inf")
                newDepth = depth
            else:
                curValue = float("inf")
                if (currentAgent + 1) % state.getNumAgents() == PACMAN: # check if this is last min agent
                    newDepth = depth -1 # decrease depth if new agent is pacman
                else:
                    newDepth = depth

            pacmanAction = None
            # loop over actions on current state
            for action in state.getLegalActions(currentAgent):
                # get the new value
                actionValue = minmax(state.generateSuccessor(currentAgent, action), # new state
                                                    newDepth, # new depth add one if it is pacmans turn again
                                                    (currentAgent + 1) % state.getNumAgents()
                                    )
                if currentAgent == PACMAN and actionValue > curValue: # max agent
                    curValue = actionValue
                    pacmanAction = action
                elif currentAgent != PACMAN and actionValue < curValue: # min agent
                    curValue = actionValue

            if depth == self.depth and currentAgent == PACMAN: # first call finished
                return pacmanAction
            else: # return value calculated
                return curValue
        return minmax(gameState, self.depth, PACMAN)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        PACMAN=0
        def alphaBetaPrune(state, alpha, beta, depth, currentAgent):
            # if leaf/end of search return the evaluation of the state
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            # set up values based on current agent 
            if currentAgent == PACMAN:
                curValue = float("-inf")
                newDepth = depth
            else:
                curValue = float("inf")
                if (currentAgent + 1) % state.getNumAgents() == PACMAN: # check if this is last min agent
                    newDepth = depth -1 # decrease depth if new agent is pacman
                else:
                    newDepth = depth

            pacmanAction = None
            # loop over actions on current state
            for action in state.getLegalActions(currentAgent):
                # get the new value
                actionValue = alphaBetaPrune(state.generateSuccessor(currentAgent, action), # new state
                                                    alpha,
                                                    beta,
                                                    newDepth, # new depth add one if it is pacmans turn again
                                                    (currentAgent + 1) % state.getNumAgents()
                                    )
                if currentAgent == PACMAN: # max agent
                    if actionValue > beta:
                        curValue = actionValue
                        break
                    else:
                        alpha = max(alpha, actionValue)
                    if actionValue > curValue: 
                        curValue = actionValue
                        pacmanAction = action
                else:
                    if actionValue < alpha:
                        curValue = actionValue
                        break
                    else:
                        beta = min(beta, actionValue)
                    if actionValue < curValue: # min agent
                        curValue = actionValue

            if depth == self.depth and currentAgent == PACMAN: # first call finished
                #print(f"depth{depth} agent{currentAgent} value{curValue} action{pacmanAction}")
                return pacmanAction
            else: # return value calculated
                #print(f"depth{depth} agent{currentAgent} value{curValue}")
                return curValue
        return alphaBetaPrune(gameState, float("-inf"), float("inf"), self.depth, PACMAN)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

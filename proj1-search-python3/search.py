# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from hashlib import new
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class searchTree:
    class State:
        def __init__(self, stateTuple, _parent = None):
            self.name = stateTuple[0]
            self.delta = stateTuple[1]
            self.cost = stateTuple[2]
            self.parent = _parent
            self.children = []

        def __repr__(self) -> str:
            return self.__str__()

        def __str__(self):
            return f"({self.name} {self.delta} {self.cost})"

    def __init__(self, frontier, problem):
        self.frontier = frontier
        self.problem = problem
        initState = (self.problem.getStartState(), None, None)
        self.root = self.State(initState)
        
    def addState(self, _parentState, _newState):
        newState = self.State(_newState, _parentState)
        _parentState.children.append(newState)
        self.frontier.push(newState)
        
    def getPath(self, endState):
        path = []
        curState = endState
        while curState != None:
            path.append(curState)
            curState = curState.parent
        path = reversed(path)
        return path

    def search(self):
        expandedStates = set()
        currentState = self.root
        while not self.problem.isGoalState(currentState.name):
            for state in self.problem.getSuccessors(currentState.name):
                self.addState(currentState, state)

            expandedStates.add(currentState.name)

            while currentState.name in expandedStates:
                try:
                    currentState = self.frontier.pop()
                except IndexError:
                    print("No Path Found")
                    return None
        
        moveList = []
        for state in self.getPath(currentState):
            moveList.append(state.delta)

        # remove null from root delta
        return moveList[1:]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    return searchTree(util.Stack(), problem).search()
    
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return searchTree(util.Queue(), problem).search()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    def totalCost(item):
        if item.parent.cost:
            item.cost += item.parent.cost
            return item.cost
        else: 
            return item.cost
    return searchTree(util.PriorityQueueWithFunction(totalCost), problem).search()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # TODO
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

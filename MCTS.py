import random
import math
import time
import copy

class MCTS():
    def __init__(self, game, time):
        '''For selecting a single move using Monte-Carlo Tree Search.
        
        Arguments:
            game:
                object - instance of a class for a game e.g. Connect_Four.
                Must have methods rollout, actions_available, set_print_game,
                play, print_board, get_reward and the attribute player_turn.
            time:
                float - seconds allowed to decide on a move.
        
        Attributes:
            seconds_allowed:
                float - seconds allowed to decide on a move.
            root:
                Node - First node in the tree, and not associated with an
                action. This nodes direct children will be for the actions
                currently available in the game (i.e, after time has expired,
                one of them will be selected).
            total_sims:
                int - for tracking total number of games simulated.
            max_depth:
                int - max number of generations in the tree.
            print_tree, print_tree_sparse:
                boolean - for toggling diagnostic printouts. Print_tree_sparse
                prints less.
            game:
                object - instance of a class for a game e.g. Connect_Four.
                Must have methods rollout, actions_available, set_print_game,
                play, print_board, get_reward and the attribute player_turn.
            player_num:
                int - which player is selecting a move.
            sim_game:
                object of same type as game. The current simulation.
        '''
        self.seconds_allowed = time
        self.root = Node(99,1-game.player_turn)
        self.total_sims = 0
        self.max_depth = 9
        self.print_tree = False
        self.print_tree_sparse = False
        self.game = game
        self.player_num = self.game.player_turn
        self.sim_game = copy.deepcopy(game)
        self.root.expand(self.sim_game.actions_available(),
                         self.sim_game.player_turn)
    
    def run_MCTS(self):
        '''Runs MCTS to completion, returning the (int) action selected.'''
        start_time = time.time()
        actions = self.game.actions_available()
        if len(actions) == 1:
            return actions[0]
        i = 0
        while time.time() - start_time < self.seconds_allowed:
            self.iteration()
            i += 1
            if self.print_tree_sparse:
                if i % 100 == 0:
                    print("iteration ", i)
                    print("Root child UCBs: ", self.root.children_UCB)
                    print("Root child Avgs: ", self.root.children_avg)
        return self.choose_move()
    
    def iteration(self):
        '''Runs one complete iteration of MCTS. Specifially it does:
        1. Selection (see 'select' method)
        2. Expansion (see 'Node.expand' method)
        3. Rollout (see 'rollout' method)
        4. Back Propogation (see 'back_prop' method)
        '''
        found_node = False
        curr_node = self.root
        self.sim_game = copy.deepcopy(self.game)
        self.sim_game.set_print_game(False)
        
        if self.print_tree: print("Selection--------------------------")
        
        while found_node == False:
            curr_node = self.select(curr_node)
            if curr_node.action < 0 or (curr_node.action >6):
                self.sim_game.print_board()
                print(curr_node)
            self.sim_game.play(curr_node.action) #self.sim_game.player_turn)
            
            if curr_node.depth >= self.max_depth:
                found_node = True
                if self.print_tree: print("-reached max depth: ", self.max_depth, curr_node.depth)
            elif curr_node.children == []:
                found_node = True
                actions = self.sim_game.actions_available()
                if actions.size != 0:
                    curr_node.expand(actions, self.sim_game.player_turn)
                    
        sim_result = self.rollout(curr_node)
        self.back_prop(curr_node, sim_result)
    
    def rollout(self, curr_node):
        '''Rollout of the rest of the game from current play state.'''
        diagnostics = False
        self.total_sims += 1
        self.sim_game.rollout()
        if diagnostics:
            print("sim end - player ", self.player_num, " gets ", self.sim_game.get_reward())
            self.sim_game.print_board()
        return self.sim_game.get_reward()
    
    def UCB(self, node):
        '''calculates upper confidence bound for a given node, using an
        optimistic high value for nodes with no rollouts.
        '''    
        if node.num_sims == 0:
            return 9999999 + random.random()
        else:
            return (node.reward_sum / node.num_sims) + math.sqrt(
                (2 * math.log(self.total_sims)) / node.num_sims
            )
    
    def update_children_UCB(self, node):
        node.children_UCB = []
        for child in node.children:
            node.children_UCB.append(self.UCB(child))
    
    def update_children_avg(self, node):
        node.children_avg = []
        for child in node.children:
            if child.num_sims == 0:
                node.children_avg.append(None)
            else:
                node.children_avg.append(child.reward_sum / child.num_sims)
    
    def select(self, node):
        '''For a given node, returns the child node with greatest upper confidence bound.'''
        self.update_children_UCB(node)
        return node.children[ #returning child with best UCB
            max(zip(node.children_UCB, range(len(node.children_UCB))))[1] #this is same as argmax
        ]
    
    def back_prop(self, curr_node, result):
        '''Increments the number of simulations and total reward for each ancestor of curr_node.
        
        Arguments:
            curr_node:
                Node - Node from which to back_propogate from (inclusive).
            result:
                iterable of length 2 - reward for back propogating for players 1, 2 respectively.
        '''
        curr_node.num_sims += 1
        curr_node.reward_sum += result[curr_node.player_num]
        
        while curr_node.parent != None:
            curr_node = curr_node.parent
            curr_node.num_sims += 1
            curr_node.reward_sum += result[curr_node.player_num]
    
    def choose_move(self):
        '''From the actions available to the player (i.e., the root nodes children), returns the
        action with the greatest average reward in simulations.'''
        node = self.root
        self.update_children_avg(node)
        if self.print_tree:
            print("avg values of actions: ", node.children_avg)
            print()
        return node.children[ #returning child with best avg
            max(zip(node.children_avg, range(len(node.children_avg))))[1] #this is same as argmax
        ].action
    
class Node():
    def __init__(self, action, player_num):
        '''Nodes in a Monte Carlo Tree Search Tree.

        Arguments:
            action:     int - action taken to get to this node
            player_num: int - which player takes the action to get to this node (need to confirm)
        '''
        if action < 0 or (action >6 and action != 99):
            raise Exception('wtf: {}'.format(action))
        self.action = action
        self.num_sims = 0
        self.reward_sum = 0
        self.children = []
        self.parent = None
        self.depth = 0
#             self.is_terminal = False
        self.children_UCB = []
        self.children_avg = []
        self.player_num = player_num

    def child_node(self, action, player_num):
        '''Add a child node from this node.'''
        self.children.append(Node(action, player_num))
        self.children[-1].parent = self
        self.children[-1].depth = self.depth + 1

    def expand(self, actions_available, player_num):
        '''Perform expand step of MCTS, creating a new child node for each available action.'''
        for action in actions_available:
            self.child_node(action.item(), player_num)
    
    @staticmethod
    def print_tree_floor(node, max_depth):
        '''For printing all the leaf nodes, starting from node, and cutting
        off the nodes beyond a max depth'''
        if node.children == [] or node.depth == max_depth:
            print(node)
        else:
            for obj in node.children:
                print_tree_bottom(obj, max_depth)
    
    def __str__(self):
        return 'Node with action={}, player_turn={},num_sims={} reward_sum={}, depth={}'.format(
            self.action,
            self.player_num,
            self.num_sims,
            self.reward_sum,
            self.depth
        )

import random
import torch
import numpy as np

class Connect_Four():
    '''Instances are a particular game of connect four.
    
    Attributes:
        num_rows, num_cols:
            Int - Dimensions of the board
        board:
            numpy array of ints in range 0-2 - 0 indicates an empty space, 1
            and 2 indicate a player's piece.
        column_heights:
           numpy array of ints -  number of pieces in a particular column
        game_over:
            boolean - whether the game is over
        winner:
            int - np.nan (pytorch does not handle None well), if game is not
            over or game is over and a tie, otherwise indicates winner (player
            0 or 1)
        player_turn:
            int - which player's turn it is (0 or 1)
        print_game:
            boolean - flag for controlling if game should be output to console
        last_action:
            int - column index of last piece played
        num_plays:
            int - total number of turns taken so far (i.e., number of pieces
            on the board)
    '''
    def __init__(self):
        self.num_rows = 6
        self.num_cols = 7
        self.board = torch.zeros([self.num_rows,self.num_cols]).long()
        self.column_heights = torch.zeros(self.num_cols).long()
        self.game_over = False
        self.winner = np.nan
        self.player_turn = 0
        self.print_game = False
        self.last_actions = [None, None]
        self.num_plays = 0
    
    def reset(self):
        '''Reset the game to beginning.'''
        self.__init__()
    
    def get_reward(self):
        '''Return tuple of rewards associated w/ current game state for
        players (0, 1) respectively.
        '''
        if self.game_over:
            if self.winner == 0:
                return (1,-1)
            elif self.winner == 1:
                return (-1,1)
            else:
                return (0,0) #draw
        elif np.isnan(self.winner):
            return (0,0) #game not done
        Exception("Did not recognize winner: {}".format(self.winner))
    
    def random_move(self, player_num):
        '''Player indicated by player_num takes action at random.'''
        actions = self.actions_available()
        rand_action = actions[random.randint(0, len(actions)-1)].item()
        self.play(rand_action)
        if self.print_game: print("player ", player_num, " played ", rand_action)
    
    def rollout(self):
        '''Plays out the rest of the game according to the rollout policy (currently random moves by
        both players).
        '''
        while not self.game_over:
            self.random_move(self.player_turn)
    
    def check_win(self):
        '''Returns np.nan if no players has won, otherwise returns player number of winner.
        
        Works by checking for consecutive sets of 4 pieces that match the one that was most recently
        played, and only checks in the vicinity of that piece.
        '''
        
        def check_axis(row, col, row_incr, col_incr, board):
            '''Checks for winning connections of 4 starting from the piece at
            (row, col), but only along the axis specified by row_incr,
            col_incr. Board is a tensor, others are ints.
            '''
            player_num = board[row,col].item()
            counter = 1
            for i in [1, -1]: #-1 inverts direction
                row_cur, col_cur = row, col
                search = True
                while search:
                    row_cur += row_incr * i
                    col_cur += col_incr * i
                    if row_cur < 0 or col_cur < 0 or row_cur >= 6 or col_cur >= 7:
                        search = False
                    elif board[row_cur, col_cur] == player_num:
                        counter += 1
                        if counter >= 4:
                            return True
                    else:
                        search = False
            return counter >= 4
            
        if self.num_plays < 7: #the earliest a player can win is on the first mover's 4th move, i.e. turn 7
            return np.nan
        else:
            last_action = self.last_actions[1-self.player_turn]
            last_action_col = last_action
            last_action_row = self.num_rows - self.column_heights[last_action].item()
            assert(self.column_heights[last_action] > 0) #otherwise there is no piece in last action col
            player = self.board[last_action_row][last_action_col].item()
            assert(player > 0)
            
            axes = [(1,0),(1,1),(0,1),(1,-1)]
            for axis in axes:
                win = check_axis(last_action_row, last_action_col, axis[0], axis[1], self.board)
                if win:
                    self.game_over = True
                    return player-1
            #board full
            if torch.sum(self.column_heights == self.num_rows) == self.num_cols and np.isnan(self.winner):
                self.game_over = True
                return np.nan
        return np.nan # game not over
    
    def play(self, action):
        '''Play a piece for current player in column 'action'. Also change player_turn and check
        for a winner.
        '''
        if action < 0 or action >= self.num_cols:
            raise Exception('invalid action: {}'.format(action))
        if self.column_heights[action] < self.num_rows:
            self.board[(self.num_rows-1) - self.column_heights[action]][action] = self.player_turn + 1
            self.column_heights[action] += 1
            self.last_actions[self.player_turn] = action
            self.num_plays += 1
        else:
            raise Exception("Board space already occupied")
        #switch who's turn it is from 1 -> 0 or 0 -> 1
        self.player_turn = 1-self.player_turn 
        self.winner = self.check_win()
    
    def actions_available(self, boolean_out=False):
        '''Returns tensor of column indices in which a legal move may be
        played. Alternatively returns 0/1 boolean mask over columns if
        boolean_out = True.
        '''
        if boolean_out:
            return (self.column_heights<self.num_rows).bool()
        else:
            if self.game_over:
                return torch.tensor([])
            else:
                return torch.arange(self.num_cols)[
                    (self.column_heights<self.num_rows)]
    
    def print_board(self):
        '''Output the current board state to console'''
        print(self.board)
        print("last action was: ", self.last_actions[1-self.player_turn])
    
    def set_print_game(self, boolean):
        self.print_game = boolean
    
    def get_state(self):
        '''Returns tuple of board (2d tensor) and turn (int)'''
        return (self.board, self.player_turn)
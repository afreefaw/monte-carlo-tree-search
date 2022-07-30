import curses
import torch
import time
from MCTS import MCTS
from connect4 import Connect_Four

def print_board(board, w, char_dict={0:' ', 1:'X', 2:'0'}, winner=None):
    '''Displays an ASCII connect 4 board in the terminal.
    Parameters:
        Board - torch.tensor
            Holds numeric indication of where each player's pieces are (0,1,2)
        w - curses window
        char_dict - Dictionary
            Maps each numeric entry in board to a character to display
    '''
    if type(board) != type(torch.tensor([])):
        raise Exception('passed board for show method must be torch.tensor'
                        'object.')
    w.clear()
    #midpoints of window
    midy = sh // 2
    midx = sw // 2

    rows = board.shape[0]
    cols = board.shape[1]

    # draw pieces
    for i, row in enumerate(board):
        for j, val in enumerate(row):
            x = midx - cols//2 + j
            y = midy - rows//2 + i
            w.addch(y, x, char_dict[int(val.item())])

    # draw border
    vpad, hpad = 1, 1 #number of spaces to leave btwn board and border
    vbord, hbord =  curses.ACS_VLINE, curses.ACS_HLINE #chars to use for vertical/horizontal borders
    bordw = cols + hpad*2 #border width
    bordh = rows + vpad*2 #border height
    
    #top and bottom
    for i in range(bordw):
        w.addch(midy - rows//2 - hpad, midx - cols//2 - hpad + i, hbord)
        w.addch(midy + rows//2 + hpad -1, midx - cols//2 - hpad + i, hbord)
    
    #column numbers
    for i in range(cols):
        w.addch(midy + rows//2 + 1, midx - cols//2 + i, str(i+1))
    
    #sides
    for i in range(bordh):
        w.addch(midy - rows//2 - vpad + i, midx - cols//2 - vpad, vbord)
        w.addch(midy - rows//2 - vpad + i, midx + cols//2 + vpad, vbord)
    w.refresh()
    
    #winner message
    if winner != None:
        msg = 'Player {} has won the game! Press enter to return to the menu.'
        w.addstr(1, midx - len(msg)//2, msg.format(winner + 1))
#         stdscr.addstr(height-1, 0, statusbarstr)
        curses.nocbreak()
        curses.flushinp()
        window.getch()
        curses.cbreak()

def ai_play(window):
    game = Connect_Four()
    print_board(game.board, window)
    while not game.game_over:
        m = MCTS(game, MCTS_time)
        game.play(m.run_MCTS())
        print_board(game.board, window)
    print_board(game.board, window, winner=game.winner)

def human_play(window):
    game = Connect_Four()
    print_board(game.board, window)
    while not game.game_over:
        #input handling
        valid_input = False
        while not valid_input:
            curses.flushinp()
            move = window.getch()
            try:
                move = int(chr(move)) - 1 # -1 to convert to 0 indexed
            except (ValueError):
                pass
            else:
                #validate column number
                if move >= 0 and move < game.board.shape[1]:
                    #can't play in filled columns
                    if game.actions_available(boolean_out=True)[move].item() == 1:
                        game.play(move)
                        valid_input = True
        
        print_board(game.board, window)
        if not game.game_over:
            m = MCTS(game, MCTS_time)
            game.play(m.run_MCTS())
        print_board(game.board, window)
    print_board(game.board, window, winner=game.winner)
    
def print_menu(w, menu, selected_row_idx):
    '''TODO Docstring
    
    Source for menu code (with modifications):
    https://github.com/nikhilkumarsingh/python-curses-tut/blob/master/
    02.%20Creating%20Menu%20Display.ipynb
    '''
    sh, sw = w.getmaxyx()
    w.clear()
    for idx, row in enumerate(menu):
        x = sw//2 - len(row)//2
        y = sh//2 - len(menu)//2 + idx
        if idx == selected_row_idx:
            w.attron(curses.color_pair(1))
            w.addstr(y, x, row)
            w.attroff(curses.color_pair(1))
        else:
            w.addstr(y, x, row)
    window.refresh()
    
def get_float(window, min_val=None, max_val=None, valid_set=None):
    '''TODO Docstring'''
    while True:
        val = window.getstr()
        try:
            val = float(val)
        except (ValueError):
            pass
        else:
            min_cond = (min_val == None) or (val >= min_val)
            max_cond = (max_val == None) or (val <= max_val)
            set_cond = (valid_set == None) or (val in valid_set)
            if min_cond and max_cond and set_cond:
                return val

def str_inp_mode(on=True):
    '''For changing from accepting single key input (on=False) to waiting
    for the enter key to be pressed to accept a full string (on=True).
    '''
    if on:
        curses.echo() # show what is typed
        curses.nocbreak() # wait for enter
        curses.curs_set(1) # show blinking cursor
    else:
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
            
#setup curses
s = curses.initscr()
s.keypad(True)
str_inp_mode(False)
curses.start_color()

#create window
sh, sw = s.getmaxyx()
window = curses.newwin(sh, sw, 0, 0)
window.keypad(True)

curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
menu = ['Human vs AI','AI vs AI', 'Set Allowed \'Thinking Time\'','Exit']
print_menu(window,menu, 0)

exit = False
current_row = 0
MCTS_time = 1.0

# Menu loop -> always return to menu after resolving menu selections
while not exit:
    key = window.getch()
    window.refresh()
    if key == curses.KEY_UP and current_row > 0:
        current_row -= 1
    elif key == curses.KEY_DOWN and current_row < len(menu)-1:
        current_row += 1
    elif key == curses.KEY_ENTER or key == 10 or key == 13:
        if current_row == 0:
            human_play(window)
        elif current_row == 1:
            ai_play(window)
        elif current_row == 2:
            window.clear()
            window.addstr('Enter the number of seconds the algorithm is '
                          'allowed to \'think\' for on each turn \r'
                          '(Min 0.2, Max 10):')
            str_inp_mode(True)
            MCTS_time = get_float(window, min_val=0.2, max_val=10)
            str_inp_mode(False)
        elif current_row == 3:
            exit = True
    print_menu(window, menu, current_row)

#return terminal to normal
str_inp_mode(True)
s.keypad(False)
curses.endwin()

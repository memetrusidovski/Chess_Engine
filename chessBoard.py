
import re
import pieces


START_PATTERN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w 0 1'

class Board(dict):
    y_axis = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
    x_axis = (1, 2, 3, 4, 5, 6, 7, 8)
    
    def __init__(self, pat=None):
        self.show(START_PATTERN)

    def alpha_notation(self, xycoord):
        if xycoord[0] < 0 or xycoord[0] > 7 or xycoord[1] < 0 or xycoord[1] > 7: 
            return
        return self.y_axis[int(xycoord[1])] + str(self.x_axis[int(xycoord[0])])

    def show(self, pat):
        self.clear()
        pat = pat.split(' ')

        def expand(match):
            return ' ' * int(match.group(0))

        pat[0] = re.compile(r'\d').sub(expand, pat[0])
        for x, row in enumerate(pat[0].split('/')):
            for y, letter in enumerate(row):
                if letter == ' ': continue
                coord = self.alpha_notation((7 - x, y))
                self[coord] = pieces.makePiece(letter)
                self[coord].place(self)
        if pat[1] == 'w':
            self.player_turn = 'white'
        else:
            self.player_turn = 'black'
        
    
    def num_notation(self, coord):
        return int(coord[1]) - 1, self.y_axis.index(coord[0])
        




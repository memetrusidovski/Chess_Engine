import sys

SHORT_NAME = {
    'R': 'Rook',
    'N': 'Knight',
    'B': 'Bishop',
    'Q': 'Queen',
    'K': 'King',
    'P': 'Pawn'
}


def makePiece(piece, colour='white'):

    if piece in (None, ' '): 
        return

    if len(piece) == 1:
        # CHecks if the the name is lower case letter, if it is then it is a black piece
        if not piece.isupper():
            colour = 'black'
        else:
            colour = 'white'

        piece = SHORT_NAME[piece.upper()]
    module = sys.modules[__name__]
    return module.__dict__[piece](colour)


class Piece(object):
    def __init__(self, colour):
        if colour == 'white':
            self.shortname = self.shortname.upper()
        
        elif colour == 'black':
            self.shortname = self.shortname.lower()
        
        self.colour = colour

    def place(self, board):
        # Tracks the board 
        self.board = board
        
class Rook(Piece):
    shortname = 'r'

class Knight(Piece):
    shortname = 'n'  

class Bishop(Piece):
    shortname = 'b'
    
class Queen(Piece):
    shortname = 'q'

class King(Piece):
    shortname = 'k'

class Pawn(Piece):
    shortname = 'p'


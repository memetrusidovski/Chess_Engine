from __future__ import annotations

import re
from bs4 import BeautifulSoup
import itertools
import typing
import requests
import collections
import copy
import dataclasses

from typing import Dict, Iterable, Callable, Tuple, \
Iterator, List, ClassVar, Hashable, Type,  Counter,\
    Mapping, Optional, SupportsInt, Generic, TypeVar, Union


isPassant = str
isTrue = bool

"""
------------------------------------------------------------------------------------------
File:    moveGen.py
Project: AI_Project_Repo
Purpose: 
==========================================================================================

Program Description:
  This will store all the moves needed to make this python engine work 
------------------------------------------------------------------------------------------
ID:      group 14
school:   wilfrid laurier 
Version  2022-11-23
-------------------------------------
"""
def pieceSymbol(pType) :
    return typing.cast(str, pieceSymbolS[pType])


def piece_name(Piecetype):
    return typing.cast(str, pieceNames[Piecetype])

# Can be used to print the board
# doesnt work with dark mode properly
asci_dict = {
    "R": "♖", "r": "♜",
    "N": "♘", "n": "♞",
    "B": "♗", "b": "♝",
    "Q": "♕", "q": "♛",
    "K": "♔", "k": "♚",
    "P": "♙", "p": "♟",
}

FileName = ["a", "b", "c", "d", "e", "f", "g", "h"]

RankName = ["1", "2", "3", "4", "5", "6", "7", "8"]

#starter
fenStart = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fenBoard = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"



class InvalidMoveError(ValueError):
    """not worded properly"""
class IllegalMoveError(ValueError):
    """this move is illegal"""
class AmbiguousMoveError(ValueError):
    """this move is no able to be made by this piece"""

"""
testing code
# Initialize the chessboard
board = [[None for _ in range(8)] for _ in range(8)]

def move_piece(start, end):
    if not (0 <= start[0] < 8 and 0 <= start[1] < 8 and 0 <= end[0] < 8 and 0 <= end[1] < 8):
        raise ValueError("Invalid starting or ending position")

    if board[start[0]][start[1]] is None:
        raise ValueError("There is no piece at the starting position")

    if board[end[0]][end[1]] is not None:
        # Check if the piece at the end position belongs to the same player
        if board[start[0]][start[1]].player == board[end[0]][end[1]].player:
            raise ValueError("You cannot capture your own piece")

    board[end[0]][end[1]] = board[start[0]][start[1]]
    board[start[0]][start[1]] = None

N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}
"""
sqrt64 = int

# Equal 1 to 64
sqrt64S = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
] = range(64)

sqrt64Name = [f + r for r in RankName for f in FileName]


def parse_sqrt64(name: str) -> sqrt64:
    return sqrt64Name.index(name)


def sqrt64_name(sqrt64: sqrt64) -> str:
    return sqrt64Name[sqrt64]


def sqrt64(file_index: int, rank_index: int) -> sqrt64:
    return rank_index * 8 + file_index


def sqrt64_file(sqrt64: sqrt64) -> int:
    return sqrt64 & 7


def sqrt64_rank(sqrt64: sqrt64) -> int:
    return sqrt64 >> 3


def sqrt64_distance(a: sqrt64, b: sqrt64) -> int:
    return max(abs(sqrt64_file(a) - sqrt64_file(b)), abs(sqrt64_rank(a) - sqrt64_rank(b)))


def sqrt64_mirror(sqrt64: sqrt64) -> sqrt64:
    return sqrt64 ^ 0x38


sqrt64S_180 = [sqrt64_mirror(sq) for sq in sqrt64S]


Board64 = int
EmptyBB = 0
BitBoardALL = 18446744073709551615

cb_SqBoardS = [
    cb_A1, cb_B1, cb_C1, cb_D1, cb_E1, cb_F1, cb_G1, cb_H1,
    cb_A2, cb_B2, cb_C2, cb_D2, cb_E2, cb_F2, cb_G2, cb_H2,
    cb_A3, cb_B3, cb_C3, cb_D3, cb_E3, cb_F3, cb_G3, cb_H3,
    cb_A4, cb_B4, cb_C4, cb_D4, cb_E4, cb_F4, cb_G4, cb_H4,
    cb_A5, cb_B5, cb_C5, cb_D5, cb_E5, cb_F5, cb_G5, cb_H5,
    cb_A6, cb_B6, cb_C6, cb_D6, cb_E6, cb_F6, cb_G6, cb_H6,
    cb_A7, cb_B7, cb_C7, cb_D7, cb_E7, cb_F7, cb_G7, cb_H7,
    cb_A8, cb_B8, cb_C8, cb_D8, cb_E8, cb_F8, cb_G8, cb_H8,
] = [1 << sq for sq in sqrt64S]

cb_CORNERS = cb_A1 | cb_H1 | cb_A8 | cb_H8
cb_CENTER = cb_D4 | cb_E4 | cb_D5 | cb_E5

cb_LIGHT_sqrt64S = 6172840429334713770
cb_DARK_sqrt64S = 0o1251255245265225325125

bitBoardFiles = [
    FileA,
    FileB,
    FileC,
    FileD,
    FileE,
    FileF,
    FileG,
    FileH,
] = [0o4010020040100200401 << i for i in range(8)]

bitBoardRanks = [
    Rank1,
    Rank2,
    Rank3,
    Rank4,
    Rank5,
    Rank6,
    Rank7,
    Rank8,
] = [0xff << (8 * i) for i in range(8)]

cb_BACKRANKS = Rank1 | Rank8


Color = isTrue
COLORS = [WHITE, BLACK] = [True, False]
Names = ["black", "white"]

isPiece = int
pieceTypes = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
pieceSymbolS = [None, "p", "n", "b", "r", "q", "k"]
pieceNames = [None, "pawn", "knight", "bishop", "rook", "queen", "king"]


def lsb(bb: Board64) -> int:
    return (bb & -bb).bit_length() - 1


def scan_forward(bb: Board64) -> Iterator[sqrt64]:
    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r


def msb(bb: Board64) -> int:
    return bb.bit_length() - 1


def scan_reversed(bb: Board64) -> Iterator[sqrt64]:
    while bb:
        r = bb.bit_length() - 1
        yield r
        bb ^= cb_SqBoardS[r]


def DownShift(b: Board64) -> Board64:
    return b >> 8


def DownShiftBy2(b: Board64) -> Board64:
    return b >> 16


def UpShift(b: Board64) -> Board64:
    return (b << 8) & BitBoardALL


def UpShiftBy2(b: Board64) -> Board64:
    return (b << 16) & BitBoardALL

def UpRightShift(b: Board64) -> Board64:
    return (b << 9) & ~FileA & BitBoardALL




def DownRightShift(b: Board64) -> Board64:
    return (b >> 7) & ~FileA

def RightShift(b: Board64) -> Board64:
    return (b << 1) & ~FileA & BitBoardALL


def LeftUpShift(b: Board64) -> Board64:
    return (b << 7) & ~FileH & BitBoardALL

def LeftShiftDown(b: Board64) -> Board64:
    return (b >> 9) & ~FileH


def LeftShift(b: Board64) -> Board64:
    return (b >> 1) & ~FileH



def SlidingAtt(sqrt64: sqrt64, occupied: Board64, deltas: Iterable[int]) -> Board64:
    attacks = EmptyBB

    for delta in deltas:
        sq = sqrt64

        while True:
            sq += delta
            if not (0 <= sq < 64) or sqrt64_distance(sq, sq - delta) > 2:
                break

            attacks |= cb_SqBoardS[sq]

            if occupied & cb_SqBoardS[sq]:
                break

    return attacks


def _step_attacks(sqrt64: sqrt64, deltas: Iterable[int]) -> Board64:
    return SlidingAtt(sqrt64, BitBoardALL, deltas)



knightAttacks = [_step_attacks(
    sq, [17, 15, 10, 6, -17, -15, -10, -6]) for sq in sqrt64S]
kingAttacks = [_step_attacks(
    sq, [9, 8, 7, 1, -9, -8, -7, -1]) for sq in sqrt64S]
pawnAttacks = [[_step_attacks(sq, deltas) for sq in sqrt64S]
               for deltas in [[-7, -9], [7, 9]]]


def _edges(sqrt64: sqrt64) -> Board64:
    return (((Rank1 | Rank8) & ~bitBoardRanks[sqrt64_rank(sqrt64)]) |
            ((FileA | FileH) & ~bitBoardFiles[sqrt64_file(sqrt64)]))


def _carry_rippler(mask: Board64) -> Iterator[Board64]:
    subset = EmptyBB
    while True:
        yield subset
        subset = (subset - mask) & mask
        if not subset:
            break


def attTableList(deltas: List[int]) -> Tuple[List[Board64], List[Dict[Board64, Board64]]]:
    mask_table = []
    attack_table = []

    for sqrt64 in sqrt64S:
        attacks = {}

        mask = SlidingAtt(sqrt64, 0, deltas) & ~_edges(sqrt64)
        for subset in _carry_rippler(mask):
            attacks[subset] = SlidingAtt(sqrt64, subset, deltas)

        attack_table.append(attacks)
        mask_table.append(mask)

    return mask_table, attack_table


cb_DIAG_MASKS, cb_DIAG_ATTACKS = attTableList([-9, -7, 7, 9])
FMasks, FileATTACKS = attTableList([-8, 8])
RMasks, RankAttacks = attTableList([-1, 1])


"""Find the rays from a diagonally moving
pieces like queens and bishops"""
def _rays() -> List[List[Board64]]:
    rays = []
    for a, cb_a in enumerate(cb_SqBoardS):
        rays_row = []
        for b, cb_b in enumerate(cb_SqBoardS):
            if cb_DIAG_ATTACKS[a][0] & cb_b:
                rays_row.append(
                    (cb_DIAG_ATTACKS[a][0] & cb_DIAG_ATTACKS[b][0]) | cb_a | cb_b)
            elif RankAttacks[a][0] & cb_b:
                rays_row.append(RankAttacks[a][0] | cb_a)
            elif FileATTACKS[a][0] & cb_b:
                rays_row.append(FileATTACKS[a][0] | cb_a)
            else:
                rays_row.append(EmptyBB)
        rays.append(rays_row)
    return rays


cb_RAYS = _rays()


def ray(a: sqrt64, b: sqrt64) -> Board64:
    return cb_RAYS[a][b]


def between(a: sqrt64, b: sqrt64) -> Board64:
    bb = cb_RAYS[a][b] & ((BitBoardALL << a) ^ (BitBoardALL << b))
    return bb & (bb - 1)


# Standard fen regex parser
Reg = re.compile(
    r"^([NBKRQ])?([a-h])?([1-8])?"
    r"[\-x]?([a-h][1-8])"
    r"(=?[nbrqkNBRQK])?[+#]?\Z")

"""
used to find a castle possibility
"""
castleReg = re.compile(
    r"^(?:-|[KQABCDEFGH]{0,2}"
    r"[kqabcdefgh]{0,2})\Z")



@dataclasses.dataclass(unsafe_hash=True)
class Move:
    from_sqrt64: sqrt64
    to_sqrt64: sqrt64
    toQueen: Optional[isPiece] = None
    drop: Optional[isPiece] = None
    '''
    fff = " "
    list(
        ('x' + 'x' * len(fff.split('/')[0]) + 'x\n') * 2 + 'x' + ''.join([
            '.' * int(c) if c.isdigit() else c for c in fff.split()[0].replace('/', 'x\nx')
        ]
        ) + 'x\n' + (
            'x' + 'x' * len(fff.split('/')[0]) + 'x\n'
        ) * 2 +
        fff.split()[1])'''

    def uci(chessboard) -> str:
        if chessboard.drop:
            return pieceSymbol(chessboard.drop).upper() + "@" + sqrt64Name[chessboard.to_sqrt64]
        elif chessboard.toQueen:
            return sqrt64Name[chessboard.from_sqrt64] + sqrt64Name[chessboard.to_sqrt64] + pieceSymbol(chessboard.toQueen)
        elif chessboard:
            return sqrt64Name[chessboard.from_sqrt64] + sqrt64Name[chessboard.to_sqrt64]
        else:
            return "0000"

    def xboard(chessboard) -> str:
        return chessboard.uci() if chessboard else "@@@@"

    def __isTrue__(chessboard) -> isTrue:
        return isTrue(chessboard.from_sqrt64 or chessboard.to_sqrt64 or chessboard.toQueen or chessboard.drop)

    def __repr__(chessboard) -> str:
        return f"Move.from_uci({chessboard.uci()!r})"

    def rotate(chessboard) -> any:
        ''' Rotates the board, preserving enpassant '''
        return chessboard.board[::-1].swapcase(), -chessboard.score, chessboard.bc, chessboard.wc, \
            119-chessboard.ep if chessboard.ep else 0, 119-chessboard.kp if chessboard.kp else 0

    def __str__(chessboard) -> str:
        return chessboard.uci()

    def test_fen2(chessboard):
        tester = None
        tools = None
        pos1 = None
        initial = tester.Position(tester.initial, 0, (True, True), (True, True), 0, 0)
        for pos in tools.flatten_tree(tools.expand_position(initial), 3):
            fen = tools.renderFEN(pos)
            chessboard.assertEqual(pos.wc, pos1.wc)
            chessboard.assertEqual(pos.bc, pos1.bc)
            ep = pos.ep if not pos.board[pos.ep].isspace() else 0
            ep1 = pos1.ep if not pos1.board[pos1.ep].isspace() else 0
            kp = pos.kp if not pos.board[pos.kp].isspace() else 0
            kp1 = pos1.kp if not pos1.board[pos1.kp].isspace() else 0
            chessboard.assertEqual(ep, ep1)
            chessboard.assertEqual(kp, kp1)

    @classmethod
    def from_uci(cls, uci: str) -> Move:
        if uci == "0000":
            return cls.null()
        elif len(uci) == 4 and "@" == uci[1]:
            try:
                drop = pieceSymbolS.index(uci[0].lower())
                sqrt64 = sqrt64Name.index(uci[2:])
            except ValueError:
                raise InvalidMoveError(f"invalid uci: {uci!r}")
            return cls(sqrt64, sqrt64, drop=drop)
        elif 4 <= len(uci) <= 5:
            try:
                from_sqrt64 = sqrt64Name.index(uci[0:2])
                to_sqrt64 = sqrt64Name.index(uci[2:4])
                toQueen = pieceSymbolS.index(
                    uci[4]) if len(uci) == 5 else None
            except ValueError:
                raise InvalidMoveError("invalid uci")
            if from_sqrt64 == to_sqrt64:
                raise InvalidMoveError("invalid uci")
            return cls(from_sqrt64, to_sqrt64, toQueen=toQueen)
        else:
            raise InvalidMoveError("expected uci")

    @classmethod
    def null(cls) -> Move:
        return cls(0, 0)


BaseBoardT = TypeVar("BaseBoardT", bound="BaseBoard")


class BaseBoard:
    """
    ------------------------------------------------------------------------------------------
    File:    moveGen.py
    Project: AI_Project_Repo
    Purpose: 
    ==========================================================================================
    
    Program Description:
      This is a helper function for creating a chess board 
      and manipulating it through different iterations of the 
      BFS search function
    ------------------------------------------------------------------------------------------
    Author:  Group 14
    ID:      CP470
    Email:   mylaurier.ca
    Version  2022-12-09
    -------------------------------------
    """
    def __init__(chessboard, board_fen: Optional[str] = fenBoard) -> None:
        chessboard.occupied_co = [EmptyBB, EmptyBB]

        if board_fen is None:
            chessboard._clear_board()
        elif board_fen == fenBoard:
            chessboard.RESETBOARD()
        else:
            chessboard.SetFenBoard(board_fen)

    def RESETBOARD(chessboard) -> None:
        chessboard.pawns = Rank2 | Rank7
        chessboard.knights = cb_B1 | cb_G1 | cb_B8 | cb_G8
        chessboard.bishop = cb_C1 | cb_F1 | cb_C8 | cb_F8
        chessboard.rk = cb_CORNERS
        chessboard.qn = cb_D1 | cb_D8
        chessboard.kings = cb_E1 | cb_E8

        chessboard.promoted = EmptyBB

        chessboard.occupied_co[WHITE] = Rank1 | Rank2
        chessboard.occupied_co[BLACK] = Rank7 | Rank8
        chessboard.occupied = Rank1 | Rank2 | Rank7 | Rank8

    def reset_board(chessboard) -> None:
        chessboard.RESETBOARD()

    def _clear_board(chessboard) -> None:
        chessboard.pawns = EmptyBB
        chessboard.knights = EmptyBB
        chessboard.bishop = EmptyBB
        chessboard.rk = EmptyBB
        chessboard.qn = EmptyBB
        chessboard.kings = EmptyBB

        chessboard.promoted = EmptyBB

        chessboard.occupied_co[WHITE] = EmptyBB
        chessboard.occupied_co[BLACK] = EmptyBB
        chessboard.occupied = EmptyBB

    def clear_board(chessboard) -> None:
        chessboard._clear_board()

    def pieces_mask(chessboard, Piecetype: isPiece, color: Color) -> Board64:
        if Piecetype == PAWN:
            bb = chessboard.pawns
        elif Piecetype == KNIGHT:
            bb = chessboard.knights
        elif Piecetype == BISHOP:
            bb = chessboard.bishop
        elif Piecetype == ROOK:
            bb = chessboard.rk
        elif Piecetype == QUEEN:
            bb = chessboard.qn
        elif Piecetype == KING:
            bb = chessboard.kings
        else:
            assert False, f"expected PieceType"

        return bb & chessboard.occupied_co[color]

    def pieceIS(chessboard, sqrt64: sqrt64) -> Optional[Piece]:
        Piecetype = chessboard.Piecetype_at(sqrt64)
        if Piecetype:
            mask = cb_SqBoardS[sqrt64]
            color = isTrue(chessboard.occupied_co[WHITE] & mask)
            return Piece(Piecetype, color)
        else:
            return None

    def Piecetype_at(chessboard, sqrt64: sqrt64) -> Optional[isPiece]:
        mask = cb_SqBoardS[sqrt64]

        if not chessboard.occupied & mask:
            return None  # Early return
        elif chessboard.pawns & mask:
            return PAWN
        elif chessboard.knights & mask:
            return KNIGHT
        elif chessboard.bishop & mask:
            return BISHOP
        elif chessboard.rk & mask:
            return ROOK
        elif chessboard.qn & mask:
            return QUEEN
        else:
            return KING

    def color_at(chessboard, sqrt64: sqrt64) -> Optional[Color]:
        mask = cb_SqBoardS[sqrt64]
        if chessboard.occupied_co[WHITE] & mask:
            return WHITE
        elif chessboard.occupied_co[BLACK] & mask:
            return BLACK
        else:
            return None

    def king(chessboard, color: Color) -> Optional[sqrt64]:
        king_mask = chessboard.occupied_co[color] & chessboard.kings & ~chessboard.promoted
        return msb(king_mask) if king_mask else None

    def attacks_mask(thisObject, sqrt64: sqrt64) -> Board64:
        cb_sqrt64 = cb_SqBoardS[sqrt64]

        if cb_sqrt64 & thisObject.pawns:
            color = isTrue(cb_sqrt64 & thisObject.occupied_co[WHITE])
            return pawnAttacks[color][sqrt64]
        elif cb_sqrt64 & thisObject.knights:
            return knightAttacks[sqrt64]
        elif cb_sqrt64 & thisObject.kings:
            return kingAttacks[sqrt64]
        else:
            attacks = 0
            if cb_sqrt64 & thisObject.bishop or cb_sqrt64 & thisObject.qn:
                attacks = cb_DIAG_ATTACKS[sqrt64][cb_DIAG_MASKS[sqrt64]
                                                       & thisObject.occupied]
            if cb_sqrt64 & thisObject.rk or cb_sqrt64 & thisObject.qn:
                attacks |= (RankAttacks[sqrt64][RMasks[sqrt64] & thisObject.occupied] |
                            FileATTACKS[sqrt64][FMasks[sqrt64] & thisObject.occupied])
            return attacks



    def _attackers_mask(acts, color: Color, sqrt64: sqrt64, occupied: Board64) -> Board64:
        rank_pieces = RMasks[sqrt64] & occupied
        file_pieces = FMasks[sqrt64] & occupied
        diag_pieces = cb_DIAG_MASKS[sqrt64] & occupied

        queens_and_rooks = acts.qn | acts.rk
        queens_and_bishops = acts.qn | acts.bishop

        attackers = (
            (kingAttacks[sqrt64] & acts.kings) |
            (knightAttacks[sqrt64] & acts.knights) |
            (RankAttacks[sqrt64][rank_pieces] & queens_and_rooks) |
            (FileATTACKS[sqrt64][file_pieces] & queens_and_rooks) |
            (cb_DIAG_ATTACKS[sqrt64][diag_pieces] & queens_and_bishops) |
            (pawnAttacks[not color][sqrt64] & acts.pawns))

        return attackers & acts.occupied_co[color]

    def attackers_mask(chessboard, color: Color, sqrt64: sqrt64) -> Board64:
        return chessboard._attackers_mask(color, sqrt64, chessboard.occupied)

    def is_attacked_by(chessboard, color: Color, sqrt64: sqrt64) -> isTrue:
        return isTrue(chessboard.attackers_mask(color, sqrt64))

    def removePiece(chessboard, sqrt64: sqrt64) -> Optional[isPiece]:
        Piecetype = chessboard.Piecetype_at(sqrt64)
        mask = cb_SqBoardS[sqrt64]

        if Piecetype == PAWN:
            chessboard.pawns ^= mask
        elif Piecetype == KNIGHT:
            chessboard.knights ^= mask
        elif Piecetype == BISHOP:
            chessboard.bishop ^= mask
        elif Piecetype == ROOK:
            chessboard.rk ^= mask
        elif Piecetype == QUEEN:
            chessboard.qn ^= mask
        elif Piecetype == KING:
            chessboard.kings ^= mask
        else:
            return None

        chessboard.occupied ^= mask
        chessboard.occupied_co[WHITE] &= ~mask
        chessboard.occupied_co[BLACK] &= ~mask

        chessboard.promoted &= ~mask

        return Piecetype

    def remove_piece_at(chessboard, sqrt64: sqrt64) -> Optional[Piece]:
        color = isTrue(chessboard.occupied_co[WHITE] & cb_SqBoardS[sqrt64])
        Piecetype = chessboard.removePiece(sqrt64)
        return Piece(Piecetype, color) if Piecetype else None

    def SetPiece(chessboard, sqrt64: sqrt64, Piecetype: isPiece, color: Color, promoted: isTrue = False) -> None:
        chessboard.removePiece(sqrt64)

        mask = cb_SqBoardS[sqrt64]

        if Piecetype == PAWN:
            chessboard.pawns |= mask
        elif Piecetype == KNIGHT:
            chessboard.knights |= mask
        elif Piecetype == BISHOP:
            chessboard.bishop |= mask
        elif Piecetype == ROOK:
            chessboard.rk |= mask
        elif Piecetype == QUEEN:
            chessboard.qn |= mask
        elif Piecetype == KING:
            chessboard.kings |= mask
        else:
            return

        chessboard.occupied ^= mask
        chessboard.occupied_co[color] ^= mask

        if promoted:
            chessboard.promoted ^= mask

    def set_piece_at(chessboard, sqrt64: sqrt64, piece: Optional[Piece], promoted: isTrue = False) -> None:
        if piece is None:
            chessboard.removePiece(sqrt64)
        else:
            chessboard.SetPiece(sqrt64, piece.Piecetype, piece.color, promoted)

    def board_fen(chessboard, *, promoted: Optional[isTrue] = False) -> str:
        builder = []
        i = 0
        empty = i

        for sqrt64 in sqrt64S_180:
            piece = chessboard.pieceIS(sqrt64)

            if not piece:
                empty += 1
            else:
                if empty:
                    builder.append(str(empty))
                    empty = i
                builder.append(piece.symbol())
                if promoted and cb_SqBoardS[sqrt64] & chessboard.promoted:
                    builder.append("~")

            if cb_SqBoardS[sqrt64] & FileH:
                if empty:
                    builder.append(str(empty))
                    empty = i

                if sqrt64 != H1:
                    builder.append("/")

        return "".join(builder)

    def SetFenBoard(chessboard, fen: str) -> None:
        # Compatibility with set_fen().
        fen = fen.strip()
        if " " in fen:
            raise ValueError(
                f"expected position part of fen, got multiple parts: {fen!r}")

        # Ensure the FEN is valid.
        rows = fen.split("/")
        if len(rows) != 8:
            raise ValueError(
                f"expected 8 rows in position part of fen: {fen!r}")

        # Validate each row.
        for row in rows:
            field_sum = 0
            previousDigit = False
            previousPiece = False

            for c in row:
                if c in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                    if previousDigit:
                        raise ValueError("error in digits of fen")
                    field_sum += int(c)
                    previousDigit = True
                    previousPiece = False
                elif c == "~":
                    if not previousPiece:
                        raise ValueError("nothign after the last piece of fen")
                    previousDigit = False
                    previousPiece = False
                elif c.lower() in pieceSymbolS:
                    field_sum += 1
                    previousDigit = False
                    previousPiece = True
                else:
                    raise ValueError("characters in wrong position")

            if field_sum != 8:
                raise ValueError("Did not receive 8 coloums in fen string")

        # clear
        chessboard._clear_board()

        # add pieces
        sqrt64_index = 0
        for c in fen:
            if c in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                sqrt64_index += int(c)
            elif c.lower() in pieceSymbolS:
                piece = Piece.from_symbol(c)
                chessboard.SetPiece(
                    sqrt64S_180[sqrt64_index], piece.Piecetype, piece.color)
                sqrt64_index += 1
            elif c == "~":
                chessboard.promoted |= cb_SqBoardS[sqrt64S_180[sqrt64_index - 1]]

    def SetTheBoard(chessboard, fen: str) -> None:
        chessboard.SetFenBoard(fen)

    def piece_map(chessboard, *, mask: Board64 = BitBoardALL) -> Dict[sqrt64, Piece]:
        result = {}
        for sqrt64 in scan_reversed(chessboard.occupied & mask):
            result[sqrt64] = typing.cast(Piece, chessboard.pieceIS(sqrt64))
        return result

    def _set_piece_map(chessboard, pieces: Mapping[sqrt64, Piece]) -> None:
        chessboard._clear_board()
        for sqrt64, piece in pieces.items():
            chessboard.SetPiece(sqrt64, piece.Piecetype, piece.color)

    def set_piece_map(chessboard, pieces: Mapping[sqrt64, Piece]) -> None:
        chessboard._set_piece_map(pieces)



    def __repr__(chessboard) -> str:
        return f"{type(chessboard).__name__}({chessboard.board_fen()!r})"

    def __str__(chessboard) -> str:
        builder = []

        for sqrt64 in sqrt64S_180:
            piece = chessboard.pieceIS(sqrt64)

            if piece:
                builder.append(piece.symbol())
            else:
                builder.append(".")

            if cb_SqBoardS[sqrt64] & FileH:
                if sqrt64 != H1:
                    builder.append("\n")
            else:
                builder.append(" ")

        return "".join(builder)

    def unicode(chessboard, *, invert_color: isTrue = False, borders: isTrue = False, empty_sqrt64: str = "⭘", orientation: Color = WHITE) -> str:
        builder = []
        append = builder.append("\n")
        for rank_index in (range(7, -1, -1) if orientation else range(8)):
            if borders:
                builder.append("  ")
                builder.append("-" * 17)

                builder.append(RankName[rank_index])
                builder.append(" ")

            for i, file_index in enumerate(range(8) if orientation else range(7, -1, -1)):
                sqrt64_index = sqrt64(file_index, rank_index)

                if borders:
                    builder.append("|")
                elif i > 0:
                    builder.append(" ")

                piece = chessboard.pieceIS(sqrt64_index)

                if piece:
                    builder.append(piece.unicode_symbol(
                        invert_color=invert_color))
                else:
                    builder.append(empty_sqrt64)

            if borders:
                builder.append("|")

            if borders or (rank_index > 0 if orientation else rank_index < 7):
                pass

        if borders:
            builder.append("  ")
            builder.append("-" * 17)
            letters = "a b c d e f g h" if orientation else "h g f e d c b a"
            builder.append("   " + letters)

        return "".join(builder)

    def __eq__(chessboard, board: object) -> isTrue:
        if isinstance(board, BaseBoard):
            return (
                chessboard.occupied == board.occupied and
                chessboard.occupied_co[WHITE] == board.occupied_co[WHITE] and
                chessboard.pawns == board.pawns and
                chessboard.knights == board.knights and
                chessboard.bishop == board.bishop and
                chessboard.rk == board.rk and
                chessboard.qn == board.qn and
                chessboard.kings == board.kings)
        else:
            return NotImplemented

    def transformer(chessboard, f: Callable[[Board64], Board64]) -> None:
        chessboard.pawns = f(chessboard.pawns)
        chessboard.knights = f(chessboard.knights)
        chessboard.bishop = f(chessboard.bishop)
        chessboard.rk = f(chessboard.rk)
        chessboard.qn = f(chessboard.qn)
        chessboard.kings = f(chessboard.kings)

        chessboard.occupied_co[WHITE] = f(chessboard.occupied_co[WHITE])
        chessboard.occupied_co[BLACK] = f(chessboard.occupied_co[BLACK])
        chessboard.occupied = f(chessboard.occupied)
        chessboard.promoted = f(chessboard.promoted)

    def transform(chessboard: BaseBoardT, f: Callable[[Board64], Board64]) -> BaseBoardT:
        board = chessboard.copy()
        board.transformer(f)
        return board

    def apply_mirror(chessboard: BaseBoardT) -> None:
        chessboard.transformer(verticalFlip)
        chessboard.occupied_co[WHITE], chessboard.occupied_co[BLACK] = chessboard.occupied_co[BLACK], chessboard.occupied_co[WHITE]

    def mirror(chessboard: BaseBoardT) -> BaseBoardT:
        board = chessboard.copy()
        board.apply_mirror()
        return board

    def copy(chessboard: BaseBoardT) -> BaseBoardT:
        """Creates a copy of the board."""
        board = type(chessboard)(None)

        board.pawns = chessboard.pawns
        board.knights = chessboard.knights
        board.bishop = chessboard.bishop
        board.rk = chessboard.rk
        board.qn = chessboard.qn
        board.kings = chessboard.kings

        board.occupied_co[WHITE] = chessboard.occupied_co[WHITE]
        board.occupied_co[BLACK] = chessboard.occupied_co[BLACK]
        board.occupied = chessboard.occupied
        board.promoted = chessboard.promoted

        return board

    def __copy__(chessboard: BaseBoardT) -> BaseBoardT:
        return chessboard.copy()

    def __deepcopy__(chessboard: BaseBoardT, memo: Dict[int, object]) -> BaseBoardT:
        board = chessboard.copy()
        memo[id(chessboard)] = board
        return board

    @classmethod
    def empty(cls: Type[BaseBoardT]) -> BaseBoardT:
        return cls(None)




BoardT = TypeVar("BoardT", bound="Board")


class _BoardState(Generic[BoardT]):

    def __init__(chessboard, board: BoardT) -> None:
        chessboard.pawns = board.pawns
        chessboard.knights = board.knights
        chessboard.bishops = board.bishop
        chessboard.rooks = board.rk
        chessboard.queens = board.qn
        chessboard.kings = board.kings

        chessboard.occupied_w = board.occupied_co[WHITE]
        chessboard.occupied_b = board.occupied_co[BLACK]
        chessboard.occupied = board.occupied

        chessboard.promoted = board.promoted

        chessboard.turn = board.turn
        chessboard.CASTLEisTrue = board.CASTLEisTrue
        chessboard.ep_sqrt64 = board.ep_sqrt64
        chessboard.HMoveclock = board.HMoveclock
        chessboard.fmoveNum = board.fmoveNum

    def restore(chessboard, board: BoardT) -> None:
        board.pawns = chessboard.pawns
        board.knights = chessboard.knights
        board.bishop = chessboard.bishops
        board.rk = chessboard.rooks
        board.qn = chessboard.queens
        board.kings = chessboard.kings

        board.occupied_co[WHITE] = chessboard.occupied_w
        board.occupied_co[BLACK] = chessboard.occupied_b
        board.occupied = chessboard.occupied

        board.promoted = chessboard.promoted

        board.turn = chessboard.turn
        board.CASTLEisTrue = chessboard.CASTLEisTrue
        board.ep_sqrt64 = chessboard.ep_sqrt64
        board.HMoveclock = chessboard.HMoveclock
        board.fmoveNum = chessboard.fmoveNum


class Board(BaseBoard):

    aliases: ClassVar[List[str]] = ["Standard", "Chess",
                                    "Classical", "Normal", "Illegal", "From Position"]
    uci_variant: ClassVar[Optional[str]] = "chess"
    xboard_variant: ClassVar[Optional[str]] = "normal"
    fenStart: ClassVar[str] = fenStart

    tbw_suffix: ClassVar[Optional[str]] = ".rtbw"
    tbz_suffix: ClassVar[Optional[str]] = ".rtbz"
    tbw_magic: ClassVar[Optional[bytes]] = b"\x71\xe8\x23\x5d"
    tbz_magic: ClassVar[Optional[bytes]] = b"\xd7\x66\x0c\xa5"
    pawnless_tbw_suffix: ClassVar[Optional[str]] = None
    pawnless_tbz_suffix: ClassVar[Optional[str]] = None
    pawnless_tbw_magic: ClassVar[Optional[bytes]] = None
    pawnless_tbz_magic: ClassVar[Optional[bytes]] = None
    connected_kings: ClassVar[isTrue] = False
    one_king: ClassVar[isTrue] = True
    captures_compulsory: ClassVar[isTrue] = False

    turn: Color
    CASTLEisTrue: Board64
    ep_sqrt64: Optional[sqrt64]
    fmoveNum: int
    HMoveclock: int
    promoted: Board64
    chessScrambler: isTrue

    move_stack: List[Move]

    def __init__(chessboard: BoardT, fen: Optional[str] = fenStart, *, chessScrambler: isTrue = False) -> None:
        BaseBoard.__init__(chessboard, None)

        chessboard.chessScrambler = chessScrambler

        chessboard.ep_sqrt64 = None
        chessboard.move_stack = []
        chessboard._stack: List[_BoardState[BoardT]] = []

        if fen is None:
            chessboard.clear()
        elif fen == type(chessboard).fenStart:
            chessboard.reset()
        else:
            chessboard.set_fen(fen)

    @property
    def legal_moves(chessboard) -> LegalMoveGenerator:
        return LegalMoveGenerator(chessboard)

    @property
    def pseudo_legal_moves(chessboard) -> PseudoLegalMoveGenerator:
        return PseudoLegalMoveGenerator(chessboard)

    def reset(chessboard) -> None:
        chessboard.turn = WHITE
        chessboard.CASTLEisTrue = cb_CORNERS
        chessboard.ep_sqrt64 = None
        chessboard.HMoveclock = 0
        chessboard.fmoveNum = 1

        chessboard.reset_board()

    def reset_board(chessboard) -> None:
        super().reset_board()
        chessboard.clear_stack()

    def clear(chessboard) -> None:
        chessboard.turn = WHITE
        chessboard.CASTLEisTrue = EmptyBB
        chessboard.ep_sqrt64 = None
        chessboard.HMoveclock = 0
        chessboard.fmoveNum = 1

        chessboard.clear_board()

    def clear_board(chessboard) -> None:
        super().clear_board()
        chessboard.clear_stack()

    def clear_stack(chessboard) -> None:
        chessboard.move_stack.clear()
        chessboard._stack.clear()

    def root(chessboard: BoardT) -> BoardT:
        if chessboard._stack:
            board = type(chessboard)(None, chessScrambler=chessboard.chessScrambler)
            chessboard._stack[0].restore(board)
            return board
        else:
            return chessboard.copy(stack=False)

    def ply(chessboard) -> int:
        return 2 * (chessboard.fmoveNum - 1) + (chessboard.turn == BLACK)

    def remove_piece_at(chessboard, sqrt64: sqrt64) -> Optional[Piece]:
        piece = super().remove_piece_at(sqrt64)
        chessboard.clear_stack()
        return piece

    def set_piece_at(chessboard, sqrt64: sqrt64, piece: Optional[Piece], promoted: isTrue = False) -> None:
        super().set_piece_at(sqrt64, piece, promoted=promoted)
        chessboard.clear_stack()

    def legaMovesGen(chessboard, maskIn: Board64 = BitBoardALL, to_mask: Board64 = BitBoardALL) -> Iterator[Move]:
        our_pieces = chessboard.occupied_co[chessboard.turn]

        # Generate piece moves.
        non_pawns = our_pieces & ~chessboard.pawns & maskIn
        for from_sqrt64 in scan_reversed(non_pawns):
            moves = chessboard.attacks_mask(from_sqrt64) & ~our_pieces & to_mask
            for to_sqrt64 in scan_reversed(moves):
                yield Move(from_sqrt64, to_sqrt64)

        if maskIn & chessboard.kings:
            yield from chessboard.generate_castling_moves(maskIn, to_mask)

        pawns = chessboard.pawns & chessboard.occupied_co[chessboard.turn] & maskIn
        if not pawns:
            return

        capturers = pawns
        for from_sqrt64 in scan_reversed(capturers):
            targets = (
                pawnAttacks[chessboard.turn][from_sqrt64] &
                chessboard.occupied_co[not chessboard.turn] & to_mask)

            for to_sqrt64 in scan_reversed(targets):
                if sqrt64_rank(to_sqrt64) in [0, 7]:
                    yield Move(from_sqrt64, to_sqrt64, QUEEN)
                    yield Move(from_sqrt64, to_sqrt64, ROOK)
                    yield Move(from_sqrt64, to_sqrt64, BISHOP)
                    yield Move(from_sqrt64, to_sqrt64, KNIGHT)
                else:
                    yield Move(from_sqrt64, to_sqrt64)

        # Prepare pawn advance generation.
        if chessboard.turn == WHITE:
            single_moves = pawns << 8 & ~chessboard.occupied
            double_moves = single_moves << 8 & ~chessboard.occupied & (
                Rank3 | Rank4)
        else:
            single_moves = pawns >> 8 & ~chessboard.occupied
            double_moves = single_moves >> 8 & ~chessboard.occupied & (
                Rank6 | Rank5)

        single_moves &= to_mask
        double_moves &= to_mask

        # Generate single pawn moves.
        for to_sqrt64 in scan_reversed(single_moves):
            from_sqrt64 = to_sqrt64 + (8 if chessboard.turn == BLACK else -8)

            if sqrt64_rank(to_sqrt64) in [0, 7]:
                yield Move(from_sqrt64, to_sqrt64, QUEEN)
                yield Move(from_sqrt64, to_sqrt64, ROOK)
                yield Move(from_sqrt64, to_sqrt64, BISHOP)
                yield Move(from_sqrt64, to_sqrt64, KNIGHT)
            else:
                yield Move(from_sqrt64, to_sqrt64)

        for to_sqrt64 in scan_reversed(double_moves):
            from_sqrt64 = to_sqrt64 + (16 if chessboard.turn == BLACK else -16)
            yield Move(from_sqrt64, to_sqrt64)

        if chessboard.ep_sqrt64:
            yield from chessboard.pseudoLegalEP(maskIn, to_mask)

    def pseudoLegalEP(chessboard, maskIn: Board64 = BitBoardALL, to_mask: Board64 = BitBoardALL) -> Iterator[Move]:
        if not chessboard.ep_sqrt64 or not cb_SqBoardS[chessboard.ep_sqrt64] & to_mask:
            return

        if cb_SqBoardS[chessboard.ep_sqrt64] & chessboard.occupied:
            return

        capturers = (
            chessboard.pawns & chessboard.occupied_co[chessboard.turn] & maskIn &
            pawnAttacks[not chessboard.turn][chessboard.ep_sqrt64] &
            bitBoardRanks[4 if chessboard.turn else 3])

        for capturer in scan_reversed(capturers):
            yield Move(capturer, chessboard.ep_sqrt64)

    def LegalCaptures(chessboard, maskIn: Board64 = BitBoardALL, to_mask: Board64 = BitBoardALL) -> Iterator[Move]:
        return itertools.chain(
            chessboard.legaMovesGen(
                maskIn, to_mask & chessboard.occupied_co[not chessboard.turn]),
            chessboard.pseudoLegalEP(maskIn, to_mask))

    def checkers_mask(chessboard) -> Board64:
        king = chessboard.king(chessboard.turn)
        return EmptyBB if king is None else chessboard.attackers_mask(not chessboard.turn, king)


    def is_check(chessboard) -> isTrue:
        return isTrue(chessboard.checkers_mask())

    def gives_check(chessboard, move: Move) -> isTrue:
        chessboard.push(move)
        try:
            return chessboard.is_check()
        finally:
            chessboard.pop()

    def causingCheck(chessboard, move: Move) -> isTrue:
        king = chessboard.king(chessboard.turn)
        if king is None:
            return False

        checkers = chessboard.attackers_mask(not chessboard.turn, king)
        if checkers and move not in chessboard.evasionz(king, checkers, cb_SqBoardS[move.from_sqrt64], cb_SqBoardS[move.to_sqrt64]):
            return True

        return not chessboard._is_safe(king, chessboard._slider_blockers(king), move)

    def was_into_check(chessboard) -> isTrue:
        king = chessboard.king(not chessboard.turn)
        return king is not None and chessboard.is_attacked_by(chessboard.turn, king)

    def play_game(T):
        # Initialize the chess board
        board = Board()

        # Create a variable to store the positions of all pieces on the board
        positions = 0

        # Main game loop
        while not board.is_game_over():
            # Get the current player's color
            color = WHITE if board.turn == True else BLACK

            # Get all possible moves for the current player
            moves = board.legal_moves

            # Choose a random move from the possible moves
            move = moves.pop()

            # Make the move
            board.push(move)

            # Update the positions of all pieces on the board
            # using bit shifts
            from_square = move.from_square
            to_square = move.to_square
            positions <<= from_square
            positions >>= to_square

    def is_fivefold_repetition(chessboard) -> isTrue:
        return chessboard.is_repetition(5)

    def can_claim_draw(chessboard) -> isTrue:
        return chessboard.can_claim_fifty_moves() or chessboard.claimRepetition()

    def is_fifty_moves(chessboard) -> isTrue:
        return chessboard._is_halfmoves(100)

    def can_claim_fifty_moves(chessboard) -> isTrue:
        if chessboard.is_fifty_moves():
            return True

        if chessboard.HMoveclock >= 99:
            for move in chessboard.calcLegalMoves():
                if not chessboard.is_zeroing(move):
                    chessboard.push(move)
                    try:
                        if chessboard.is_fifty_moves():
                            return True
                    finally:
                        chessboard.pop()

        return False

    def is_pseudo_legal(chessboard, move: Move) -> isTrue:
        # Null moves are not pseudo-legal.
        if not move:
            return False

        # Drops are not pseudo-legal.
        if move.drop:
            return False

        # Source sqrt64 must not be vacant.
        piece = chessboard.Piecetype_at(move.from_sqrt64)
        if not piece:
            return False

        # Get sqrt64 masks.
        maskIn = cb_SqBoardS[move.from_sqrt64]
        to_mask = cb_SqBoardS[move.to_sqrt64]

        # Check turn.
        if not chessboard.occupied_co[chessboard.turn] & maskIn:
            return False

        # Only pawns can promote and only on the backrank.
        if move.toQueen:
            if piece != PAWN:
                return False

            if chessboard.turn == WHITE and sqrt64_rank(move.to_sqrt64) != 7:
                return False
            elif chessboard.turn == BLACK and sqrt64_rank(move.to_sqrt64) != 0:
                return False

        # Handle castling.
        if piece == KING:
            move = chessboard._from_chessScrambler(
                chessboard.chessScrambler, move.from_sqrt64, move.to_sqrt64)
            if move in chessboard.generate_castling_moves():
                return True

        # Destination sqrt64 can not be occupied.
        if chessboard.occupied_co[chessboard.turn] & to_mask:
            return False

        # Handle pawn moves.
        if piece == PAWN:
            return move in chessboard.legaMovesGen(maskIn, to_mask)

        # Handle all other pieces.
        return isTrue(chessboard.attacks_mask(move.from_sqrt64) & to_mask)

    def is_legal(chessboard, move: Move) -> isTrue:
        return not chessboard.isVariant_end() and chessboard.is_pseudo_legal(move) and not chessboard.causingCheck(move)

    def isVariant_end(chessboard) -> isTrue:
        return False

    def isVariant_loss(chessboard) -> isTrue:
        return False

    def isVariant_win(chessboard) -> isTrue:
        return False

    def isVariant_draw(chessboard) -> isTrue:
        return False




    def is_checkmate(chessboard) -> isTrue:
        if not chessboard.is_check():
            return False

        return not any(chessboard.calcLegalMoves())

    def is_stalemate(chessboard) -> isTrue:
        if chessboard.is_check():
            return False

        if chessboard.isVariant_end():
            return False

        return not any(chessboard.calcLegalMoves())

    def is_insufficient_material(chessboard) -> isTrue:
        return all(chessboard.contain_insufficient_material(color) for color in COLORS)

    def contain_insufficient_material(chessboard, color: Color) -> isTrue:
        if chessboard.occupied_co[color] & (chessboard.pawns | chessboard.rk | chessboard.qn):
            return False

        if chessboard.occupied_co[color] & chessboard.knights:
            return (popcount(chessboard.occupied_co[color]) <= 2 and
                    not (chessboard.occupied_co[not color] & ~chessboard.kings & ~chessboard.qn))

        if chessboard.occupied_co[color] & chessboard.bishop:
            same_color = (not chessboard.bishop & cb_DARK_sqrt64S) or (
                not chessboard.bishop & cb_LIGHT_sqrt64S)
            return same_color and not chessboard.pawns and not chessboard.knights

        return True

    def _is_halfmoves(chessboard, n: int) -> isTrue:
        return chessboard.HMoveclock >= n and any(chessboard.calcLegalMoves())

    def is_seventyfive_moves(chessboard) -> isTrue:
        return chessboard._is_halfmoves(150)


    def claimRepetition(chessboard) -> isTrue:
        transpositionKey = chessboard._transpositionKey()
        transpositions: Counter[Hashable] = collections.Counter()
        transpositions.update((transpositionKey, ))

        # Count positions.
        switchyard = []
        while chessboard.move_stack:
            move = chessboard.pop()
            switchyard.append(move)

            if chessboard.is_irreversible(move):
                break

            transpositions.update((chessboard._transpositionKey(), ))

        while switchyard:
            chessboard.push(switchyard.pop())


        if transpositions[transpositionKey] >= 3:
            return True


        for move in chessboard.calcLegalMoves():
            chessboard.push(move)
            try:
                if transpositions[chessboard._transpositionKey()] >= 2:
                    return True
            finally:
                chessboard.pop()

        return False

    def is_repetition(chessboard, count: int = 3) -> isTrue:
        maybe_repetitions = 1
        for state in reversed(chessboard._stack):
            if state.occupied == chessboard.occupied:
                maybe_repetitions += 1
                if maybe_repetitions >= count:
                    break
        if maybe_repetitions < count:
            return False


        transpositionKey = chessboard._transpositionKey()
        switchyard = []

        try:
            while True:
                if count <= 1:
                    return True

                if len(chessboard.move_stack) < count - 1:
                    break

                move = chessboard.pop()
                switchyard.append(move)

                if chessboard.is_irreversible(move):
                    break

                if chessboard._transpositionKey() == transpositionKey:
                    count -= 1
        finally:
            while switchyard:
                chessboard.push(switchyard.pop())

        return False

    def _board_state(chessboard: BoardT) -> _BoardState[BoardT]:
        return _BoardState(chessboard)

    def _push_capture(chessboard, move: Move, CAPTSQUARE: sqrt64, Piecetype: isPiece, was_promoted: isTrue) -> None:
        pass

    def push(chessboard: BoardT, move: Move) -> None:

        move = chessboard._to_chessScrambler(move)
        board_state = chessboard._board_state()
        chessboard.CASTLEisTrue = chessboard.clean_CASTLEisTrue()
        chessboard.move_stack.append(chessboard._from_chessScrambler(
            chessboard.chessScrambler, move.from_sqrt64, move.to_sqrt64, move.toQueen, move.drop))
        chessboard._stack.append(board_state)


        ep_sqrt64 = chessboard.ep_sqrt64
        chessboard.ep_sqrt64 = None


        chessboard.HMoveclock += 1
        if chessboard.turn == BLACK:
            chessboard.fmoveNum += 1


        if not move:
            chessboard.turn = not chessboard.turn
            return


        if move.drop:
            chessboard.SetPiece(move.to_sqrt64, move.drop, chessboard.turn)
            chessboard.turn = not chessboard.turn
            return

        if chessboard.is_zeroing(move):
            chessboard.HMoveclock = 0

        from_bb = cb_SqBoardS[move.from_sqrt64]
        to_bb = cb_SqBoardS[move.to_sqrt64]

        promoted = isTrue(chessboard.promoted & from_bb)
        Piecetype = chessboard.removePiece(move.from_sqrt64)
        assert Piecetype is not None, f" move-not-legal  "
        CAPTSQUARE = move.to_sqrt64
        captured_Piecetype = chessboard.Piecetype_at(CAPTSQUARE)

        chessboard.CASTLEisTrue &= ~to_bb & ~from_bb
        if Piecetype == KING and not promoted:
            if chessboard.turn == WHITE:
                chessboard.CASTLEisTrue &= ~Rank1
            else:
                chessboard.CASTLEisTrue &= ~Rank8
        elif captured_Piecetype == KING and not chessboard.promoted & to_bb:
            if chessboard.turn == WHITE and sqrt64_rank(move.to_sqrt64) == 7:
                chessboard.CASTLEisTrue &= ~Rank8
            elif chessboard.turn == BLACK and sqrt64_rank(move.to_sqrt64) == 0:
                chessboard.CASTLEisTrue &= ~Rank1

        if Piecetype == PAWN:
            diff = move.to_sqrt64 - move.from_sqrt64

            if diff == 16 and sqrt64_rank(move.from_sqrt64) == 1:
                chessboard.ep_sqrt64 = move.from_sqrt64 + 8
            elif diff == -16 and sqrt64_rank(move.from_sqrt64) == 6:
                chessboard.ep_sqrt64 = move.from_sqrt64 - 8
            elif move.to_sqrt64 == ep_sqrt64 and abs(diff) in [7, 9] and not captured_Piecetype:
                # Remove pawns captured en passant.
                down = -8 if chessboard.turn == WHITE else 8
                CAPTSQUARE = ep_sqrt64 + down
                captured_Piecetype = chessboard.removePiece(CAPTSQUARE)

        if move.toQueen:
            promoted = True
            Piecetype = move.toQueen

        castling = Piecetype == KING and chessboard.occupied_co[chessboard.turn] & to_bb
        if castling:
            a_side = sqrt64_file(
                move.to_sqrt64) < sqrt64_file(move.from_sqrt64)

            chessboard.removePiece(move.from_sqrt64)
            chessboard.removePiece(move.to_sqrt64)

            if a_side:
                chessboard.SetPiece(C1 if chessboard.turn ==
                                          WHITE else C8, KING, chessboard.turn)
                chessboard.SetPiece(D1 if chessboard.turn ==
                                          WHITE else D8, ROOK, chessboard.turn)
            else:
                chessboard.SetPiece(G1 if chessboard.turn ==
                                          WHITE else G8, KING, chessboard.turn)
                chessboard.SetPiece(F1 if chessboard.turn ==
                                          WHITE else F8, ROOK, chessboard.turn)

        if not castling:
            was_promoted = isTrue(chessboard.promoted & to_bb)
            chessboard.SetPiece(move.to_sqrt64, Piecetype, chessboard.turn, promoted)

            if captured_Piecetype:
                chessboard._push_capture(move, CAPTSQUARE,
                                         captured_Piecetype, was_promoted)

        chessboard.turn = not chessboard.turn

    def pop(chessboard: BoardT) -> Move:
        move = chessboard.move_stack.pop()
        chessboard._stack.pop().restore(chessboard)
        return move

    def peek(chessboard) -> Move:
        return chessboard.move_stack[-1]

    def find_move(chessboard, from_sqrt64: sqrt64, to_sqrt64: sqrt64, toQueen: Optional[isPiece] = None) -> Move:
        if toQueen is None and chessboard.pawns & cb_SqBoardS[from_sqrt64] and cb_SqBoardS[to_sqrt64] & cb_BACKRANKS:
            toQueen = QUEEN

        move = chessboard._from_chessScrambler(
            chessboard.chessScrambler, from_sqrt64, to_sqrt64, toQueen)
        if not chessboard.is_legal(move):
            raise IllegalMoveError("no legal move")

        return move

    def castling_shredder_fen(chessboard) -> str:
        CASTLEisTrue = chessboard.clean_CASTLEisTrue()
        if not CASTLEisTrue:
            return "-"

        builder = []

        for sqrt64 in scan_reversed(CASTLEisTrue & Rank1):
            builder.append(FileName[sqrt64_file(sqrt64)].upper())

        for sqrt64 in scan_reversed(CASTLEisTrue & Rank8):
            builder.append(FileName[sqrt64_file(sqrt64)])

        return "".join(builder)




    def containsPassant(chessboard) -> isTrue:
        return chessboard.ep_sqrt64 is not None and any(chessboard.generate_legal_ep())



    def set_fen(thisFEN, fen: str) -> None:
        parts = fen.split()

        try:
            board_part = parts.pop(0)
        except IndexError:
            raise ValueError("empty fen")

        try:
            turn_part = parts.pop(0)
        except IndexError:
            turn = WHITE
        else:
            if turn_part == "w":
                turn = WHITE
            elif turn_part == "b":
                turn = BLACK
            else:
                raise ValueError(
                    "only use b or w in fen")

        try:
            castling_part = parts.pop(0)
        except IndexError:
            castling_part = "-"
        else:
            if not castleReg.match(castling_part):
                raise ValueError("wrong move for castling")

        try:
            ep_part = parts.pop(0)
        except IndexError:
            ep_sqrt64 = None
        else:
            try:
                ep_sqrt64 = None if ep_part == "-" else sqrt64Name.index(
                    ep_part)
            except ValueError:
                raise ValueError('pesant is wrong')

        try:
            HMovepart = parts.pop(0)
        except IndexError:
            HMoveclock = 0
        else:
            try:
                HMoveclock = int(HMovepart)
            except ValueError:
                raise ValueError("Half move clock not valid")

            if HMoveclock < 0:
                raise ValueError(
                    "negative numbers not valid")

        try:
            fullmove_part = parts.pop(0)
        except IndexError:
            fmoveNum = 1
        else:
            try:
                fmoveNum = int(fullmove_part)
            except ValueError:
                raise ValueError("fullmove invalid")

            if fmoveNum < 0:
                raise ValueError("cant have negative moves")

            fmoveNum = max(fmoveNum, 1)

        if parts:
            raise ValueError("fen has too many parts ")

        thisFEN.SetFenBoard(board_part)


        thisFEN.turn = turn
        thisFEN._CastleFEN(castling_part)
        thisFEN.ep_sqrt64 = ep_sqrt64
        thisFEN.HMoveclock = HMoveclock
        thisFEN.fmoveNum = fmoveNum
        thisFEN.clear_stack()

    def _CastleFEN(chessboard, castling_fen: str) -> None:
        if not castling_fen or castling_fen == "-":
            chessboard.CASTLEisTrue = EmptyBB
            return

        if not castleReg.match(castling_fen):
            raise ValueError(f"invalid castling fen: {castling_fen!r}")

        chessboard.CASTLEisTrue = EmptyBB

        for flag in castling_fen:
            color = WHITE if flag.isupper() else BLACK
            flag = flag.lower()
            backrank = Rank1 if color == WHITE else Rank8
            rooks = chessboard.occupied_co[color] & chessboard.rk & backrank
            king = chessboard.king(color)

            if flag == "q":
                if king is not None and lsb(rooks) < king:
                    chessboard.CASTLEisTrue |= rooks & -rooks
                else:
                    chessboard.CASTLEisTrue |= FileA & backrank
            elif flag == "k":
                rook = msb(rooks)
                if king is not None and king < rook:
                    chessboard.CASTLEisTrue |= cb_SqBoardS[rook]
                else:
                    chessboard.CASTLEisTrue |= FileH & backrank
            else:
                chessboard.CASTLEisTrue |= bitBoardFiles[FileName.index(
                    flag)] & backrank

    def CastleFEN(chessboard, castling_fen: str) -> None:
        chessboard._CastleFEN(castling_fen)
        chessboard.clear_stack()

    def SetTheBoard(chessboard, fen: str) -> None:
        super().SetTheBoard(fen)
        chessboard.clear_stack()

    def set_piece_map(chessboard, pieces: Mapping[sqrt64, Piece]) -> None:
        super().set_piece_map(pieces)
        chessboard.clear_stack()



    def san(chessboard, move: Move) -> str:
        return chessboard._algebraic(move)

    def lan(chessboard, move: Move) -> str:
        return chessboard._algebraic(move, long=True)

    def san_and_push(chessboard, move: Move) -> str:
        return chessboard._algebraic_and_push(move)

    def _algebraic(chessboard, move: Move, *, long: isTrue = False) -> str:
        san = chessboard._algebraic_and_push(move, long=long)
        chessboard.pop()
        return san

    def _algebraic_and_push(chessboard, move: Move, *, long: isTrue = False) -> str:
        san = chessboard._algebraic_without_suffix(move, long=long)

        chessboard.push(move)
        is_check = chessboard.is_check()
        isMate = (is_check and chessboard.is_checkmate()
                        ) or chessboard.isVariant_loss() or chessboard.isVariant_win()

        if isMate and move:
            return san + "#"
        elif is_check and move:
            return san + "+"
        else:
            return san

    def _algebraic_without_suffix(chessboard, move: Move, *, long: isTrue = False) -> str:
        if not move:
            return "--"

        if move.drop:
            san = ""
            if move.drop != PAWN:
                san = pieceSymbol(move.drop).upper()
            san += "@" + sqrt64Name[move.to_sqrt64]
            return san

        if chessboard.isCastle(move):
            if sqrt64_file(move.to_sqrt64) < sqrt64_file(move.from_sqrt64):
                return "O-O-O"
            else:
                return "O-O"

        Piecetype = chessboard.Piecetype_at(move.from_sqrt64)
        assert Piecetype, f" {move} n {chessboard.fen()}"
        capture = chessboard.is_capture(move)

        if Piecetype == PAWN:
            san = ""
        else:
            san = pieceSymbol(Piecetype).upper()

        if long:
            san += sqrt64Name[move.from_sqrt64]
        elif Piecetype != PAWN:
            others = 0
            maskIn = chessboard.pieces_mask(Piecetype, chessboard.turn)
            maskIn &= ~cb_SqBoardS[move.from_sqrt64]
            to_mask = cb_SqBoardS[move.to_sqrt64]
            for candidate in chessboard.calcLegalMoves(maskIn, to_mask):
                others |= cb_SqBoardS[candidate.from_sqrt64]

            # Disambiguate.
            if others:
                row, column = False, False

                if others & bitBoardRanks[sqrt64_rank(move.from_sqrt64)]:
                    column = True

                if others & bitBoardFiles[sqrt64_file(move.from_sqrt64)]:
                    row = True
                else:
                    column = True

                if column:
                    san += FileName[sqrt64_file(move.from_sqrt64)]
                if row:
                    san += RankName[sqrt64_rank(move.from_sqrt64)]
        elif capture:
            san += FileName[sqrt64_file(move.from_sqrt64)]


        if capture:
            san += "x"
        elif long:
            san += "-"


        san += sqrt64Name[move.to_sqrt64]


        if move.toQueen:
            san += "=" + pieceSymbol(move.toQueen).upper()

        return san



    def parse_san(chessboard, san: str) -> Move:
        try:
            if san in ["O-O", "O-O+", "O-O#", "0-0", "0-0+", "0-0#"]:
                return next(move for move in chessboard.generate_castling_moves() if chessboard.is_kingside_castling(move))
            elif san in ["O-O-O", "O-O-O+", "O-O-O#", "0-0-0", "0-0-0+", "0-0-0#"]:
                return next(move for move in chessboard.generate_castling_moves() if chessboard.is_queenside_castling(move))
        except StopIteration:
            raise IllegalMoveError(f"illegal san")

        # Match normal moves.
        match = Reg.match(san)
        if not match:
            # Null moves.
            if san in ["--", "Z0", "0000", "@@@@"]:
                return Move.null()
            elif "," in san:
                raise InvalidMoveError("unsupported")
            else:
                raise InvalidMoveError("invalid san")

        to_sqrt64 = sqrt64Name.index(match.group(4))
        to_mask = cb_SqBoardS[to_sqrt64] & ~chessboard.occupied_co[chessboard.turn]


        p = match.group(5)
        toQueen = pieceSymbolS.index(p[-1].lower()) if p else None


        maskIn = BitBoardALL
        if match.group(2):
            fileIn = FileName.index(match.group(2))
            maskIn &= bitBoardFiles[fileIn]
        if match.group(3):
            from_rank = int(match.group(3)) - 1
            maskIn &= bitBoardRanks[from_rank]


        if match.group(1):
            Piecetype = pieceSymbolS.index(match.group(1).lower())
            maskIn &= chessboard.pieces_mask(Piecetype, chessboard.turn)
        elif match.group(2) and match.group(3):
            move = chessboard.find_move(
                sqrt64(fileIn, from_rank), to_sqrt64, toQueen)
            if move.toQueen == toQueen:
                return move
            else:
                raise IllegalMoveError("missing piece")
        else:
            maskIn &= chessboard.pawns

            if not match.group(2):
                maskIn &= bitBoardFiles[sqrt64_file(to_sqrt64)]

        matched_move = None
        for move in chessboard.calcLegalMoves(maskIn, to_mask):
            if move.toQueen != toQueen:
                continue

            if matched_move:
                raise AmbiguousMoveError(
                    f"ambiguous san")

            matched_move = move

        if not matched_move:
            raise IllegalMoveError("illegal san")

        return matched_move

    def push_san(chessboard, san: str) -> Move:
        move = chessboard.parse_san(san)
        chessboard.push(move)
        return move



    def isPassant(chessboard, move: Move) -> isTrue:
        return (chessboard.ep_sqrt64 == move.to_sqrt64 and
                isTrue(chessboard.pawns & cb_SqBoardS[move.from_sqrt64]) and
                abs(move.to_sqrt64 - move.from_sqrt64) in [7, 9] and
                not chessboard.occupied & cb_SqBoardS[move.to_sqrt64])

    def is_capture(chessboard, move: Move) -> isTrue:
        touched = cb_SqBoardS[move.from_sqrt64] ^ cb_SqBoardS[move.to_sqrt64]
        return isTrue(touched & chessboard.occupied_co[not chessboard.turn]) or chessboard.isPassant(move)

    def is_zeroing(chessboard, move: Move) -> isTrue:
        touched = cb_SqBoardS[move.from_sqrt64] ^ cb_SqBoardS[move.to_sqrt64]
        return isTrue(touched & chessboard.pawns or touched & chessboard.occupied_co[not chessboard.turn] or move.drop == PAWN)

    def _reduces_CASTLEisTrue(chessboard, move: Move) -> isTrue:
        cr = chessboard.clean_CASTLEisTrue()
        seen = cb_SqBoardS[move.from_sqrt64] ^ cb_SqBoardS[move.to_sqrt64]
        return isTrue(seen & cr or
                    cr & Rank1 and seen & chessboard.kings & chessboard.occupied_co[WHITE] & ~chessboard.promoted or
                    cr & Rank8 and seen & chessboard.kings & chessboard.occupied_co[BLACK] & ~chessboard.promoted)

    def is_irreversible(chessboard, move: Move) -> isTrue:
        return chessboard.is_zeroing(move) or chessboard._reduces_CASTLEisTrue(move) or chessboard.containsPassant()

    def isCastle(chessboard, move: Move) -> isTrue:
        if chessboard.kings & cb_SqBoardS[move.from_sqrt64]:
            diff = sqrt64_file(move.from_sqrt64) - sqrt64_file(move.to_sqrt64)
            return abs(diff) > 1 or isTrue(chessboard.rk & chessboard.occupied_co[chessboard.turn] & cb_SqBoardS[move.to_sqrt64])
        return False

    def _valid_ep_sqrt64(chessboard) -> Optional[sqrt64]:
        if not chessboard.ep_sqrt64:
            return None

        if chessboard.turn == WHITE:
            ep_rank = 5
            pawn_mask = DownShift(cb_SqBoardS[chessboard.ep_sqrt64])
            seventh_rank_mask = UpShift(cb_SqBoardS[chessboard.ep_sqrt64])
        else:
            ep_rank = 2
            pawn_mask = UpShift(cb_SqBoardS[chessboard.ep_sqrt64])
            seventh_rank_mask = DownShift(cb_SqBoardS[chessboard.ep_sqrt64])


        if sqrt64_rank(chessboard.ep_sqrt64) != ep_rank:
            return None

        if not chessboard.pawns & chessboard.occupied_co[not chessboard.turn] & pawn_mask:
            return None

        if chessboard.occupied & cb_SqBoardS[chessboard.ep_sqrt64]:
            return None

        if chessboard.occupied & seventh_rank_mask:
            return None

        return chessboard.ep_sqrt64

    def is_kingside_castling(chessboard, move: Move) -> isTrue:
        return chessboard.isCastle(move) and sqrt64_file(move.to_sqrt64) > sqrt64_file(move.from_sqrt64)

    def is_queenside_castling(chessboard, move: Move) -> isTrue:
        return chessboard.isCastle(move) and sqrt64_file(move.to_sqrt64) < sqrt64_file(move.from_sqrt64)

    def clean_CASTLEisTrue(chessboard) -> Board64:
        if chessboard._stack:
            return chessboard.CASTLEisTrue

        castling = chessboard.CASTLEisTrue & chessboard.rk
        white_castling = castling & Rank1 & chessboard.occupied_co[WHITE]
        black_castling = castling & Rank8 & chessboard.occupied_co[BLACK]

        if not chessboard.chessScrambler:
            white_castling &= (cb_A1 | cb_H1)
            black_castling &= (cb_A8 | cb_H8)


            if not chessboard.occupied_co[WHITE] & chessboard.kings & ~chessboard.promoted & cb_E1:
                white_castling = 0
            if not chessboard.occupied_co[BLACK] & chessboard.kings & ~chessboard.promoted & cb_E8:
                black_castling = 0

            return white_castling | black_castling
        else:
            white_king_mask = chessboard.occupied_co[WHITE] & chessboard.kings & Rank1 & ~chessboard.promoted
            black_king_mask = chessboard.occupied_co[BLACK] & chessboard.kings & Rank8 & ~chessboard.promoted
            if not white_king_mask:
                white_castling = 0
            if not black_king_mask:
                black_castling = 0


            white_a_side = white_castling & -white_castling
            white_h_side = cb_SqBoardS[msb(
                white_castling)] if white_castling else 0

            if white_a_side and msb(white_a_side) > msb(white_king_mask):
                white_a_side = 0
            if white_h_side and msb(white_h_side) < msb(white_king_mask):
                white_h_side = 0

            black_a_side = (black_castling & -black_castling)
            black_h_side = cb_SqBoardS[msb(
                black_castling)] if black_castling else EmptyBB

            if black_a_side and msb(black_a_side) > msb(black_king_mask):
                black_a_side = 0
            if black_h_side and msb(black_h_side) < msb(black_king_mask):
                black_h_side = 0


            return black_a_side | black_h_side | white_a_side | white_h_side

    def contain_CASTLEisTrue(chessboard, color: Color) -> isTrue:
        backrank = Rank1 if color == WHITE else Rank8
        return isTrue(chessboard.clean_CASTLEisTrue() & backrank)

    def contain_kingside_CASTLEisTrue(chessboard, color: Color) -> isTrue:
        backrank = Rank1 if color == WHITE else Rank8
        king_mask = chessboard.kings & chessboard.occupied_co[color] & backrank & ~chessboard.promoted
        if not king_mask:
            return False

        CASTLEisTrue = chessboard.clean_CASTLEisTrue() & backrank
        while CASTLEisTrue:
            rook = CASTLEisTrue & -CASTLEisTrue

            if rook > king_mask:
                return True

            CASTLEisTrue &= CASTLEisTrue - 1

        return False

    def contain_queenside_CASTLEisTrue(chessboard, color: Color) -> isTrue:
        backrank = Rank1 if color == WHITE else Rank8
        king_mask = chessboard.kings & chessboard.occupied_co[color] & backrank & ~chessboard.promoted
        if not king_mask:
            return False

        CASTLEisTrue = chessboard.clean_CASTLEisTrue() & backrank
        while CASTLEisTrue:
            rook = CASTLEisTrue & -CASTLEisTrue

            if rook < king_mask:
                return True

            CASTLEisTrue &= CASTLEisTrue - 1

        return False




        valid_ep_sqrt64 = chessboard._valid_ep_sqrt64()
        if chessboard.ep_sqrt64 != valid_ep_sqrt64:
            ERROR |= STATUS_INVALID_EP_sqrt64



        checkers = chessboard.checkers_mask()
        our_kings = chessboard.kings & chessboard.occupied_co[chessboard.turn] & ~chessboard.promoted
        if checkers:
            if popcount(checkers) > 2:
                ERROR |= STATUS_TOO_MANY_CHECKERS

            if valid_ep_sqrt64 is not None:
                pushed_to = valid_ep_sqrt64 ^ A2
                pushed_from = valid_ep_sqrt64 ^ A4
                occupied_before = (
                    chessboard.occupied & ~cb_SqBoardS[pushed_to]) | cb_SqBoardS[pushed_from]
                if popcount(checkers) > 1 or (
                        msb(checkers) != pushed_to and
                        chessboard._attacked_for_king(our_kings, occupied_before)):
                    ERROR |= STATUS_IMPOSSIBLE_CHECK
            else:
                if popcount(checkers) > 2 or (popcount(checkers) == 2 and ray(lsb(checkers), msb(checkers)) & our_kings):
                    ERROR |= STATUS_IMPOSSIBLE_CHECK

        return ERROR




    def calcLegalMoves(chessboard, maskIn: Board64 = BitBoardALL, to_mask: Board64 = BitBoardALL) -> Iterator[Move]:
        if chessboard.isVariant_end():
            return

        king_mask = chessboard.kings & chessboard.occupied_co[chessboard.turn]
        if king_mask:
            king = msb(king_mask)
            blockers = chessboard._slider_blockers(king)
            checkers = chessboard.attackers_mask(not chessboard.turn, king)
            if checkers:
                for move in chessboard.evasionz(king, checkers, maskIn, to_mask):
                    if chessboard._is_safe(king, blockers, move):
                        yield move
            else:
                for move in chessboard.legaMovesGen(maskIn, to_mask):
                    if chessboard._is_safe(king, blockers, move):
                        yield move
        else:
            yield from chessboard.legaMovesGen(maskIn, to_mask)

    def generate_legal_ep(chessboard, maskIn: Board64 = BitBoardALL, to_mask: Board64 = BitBoardALL) -> Iterator[Move]:
        if chessboard.isVariant_end():
            return

        for move in chessboard.pseudoLegalEP(maskIn, to_mask):
            if not chessboard.causingCheck(move):
                yield move

    def _ep_skewered(chessboard, king: sqrt64, capturer: sqrt64) -> isTrue:
        assert chessboard.ep_sqrt64 is not None

        last_double = chessboard.ep_sqrt64 + (-8 if chessboard.turn == WHITE else 8)

        occupancy = (chessboard.occupied & ~cb_SqBoardS[last_double] &
                     ~cb_SqBoardS[capturer] | cb_SqBoardS[chessboard.ep_sqrt64])

        # Horizontal attack on the fifth or fourth rank.
        horizontal_attackers = chessboard.occupied_co[not chessboard.turn] & (
            chessboard.rk | chessboard.qn)
        if RankAttacks[king][RMasks[king] & occupancy] & horizontal_attackers:
            return True

        diagonal_attackers = chessboard.occupied_co[not chessboard.turn] & (
            chessboard.bishop | chessboard.qn)
        if cb_DIAG_ATTACKS[king][cb_DIAG_MASKS[king] & occupancy] & diagonal_attackers:
            return True

        return False

    def _slider_blockers(chessboard, king: sqrt64) -> Board64:
        rooks_and_queens = chessboard.rk | chessboard.qn
        bishops_and_queens = chessboard.bishop | chessboard.qn

        snipers = ((RankAttacks[king][0] & rooks_and_queens) |
                   (FileATTACKS[king][0] & rooks_and_queens) |
                   (cb_DIAG_ATTACKS[king][0] & bishops_and_queens))

        blockers = 0

        for sniper in scan_reversed(snipers & chessboard.occupied_co[not chessboard.turn]):
            b = between(king, sniper) & chessboard.occupied


            if b and cb_SqBoardS[msb(b)] == b:
                blockers |= b

        return blockers & chessboard.occupied_co[chessboard.turn]

    def _is_safe(chessboard, king: sqrt64, blockers: Board64, move: Move) -> isTrue:
        if move.from_sqrt64 == king:
            if chessboard.isCastle(move):
                return True
            else:
                return not chessboard.is_attacked_by(not chessboard.turn, move.to_sqrt64)

        else:
            return isTrue(not blockers & cb_SqBoardS[move.from_sqrt64] or
                        ray(move.from_sqrt64, move.to_sqrt64) & cb_SqBoardS[king])

    def evasionz(chessboard, king: sqrt64, checkers: Board64, maskIn: Board64 = BitBoardALL, to_mask: Board64 = BitBoardALL) -> Iterator[Move]:
        sliders = checkers & (chessboard.bishop | chessboard.rk | chessboard.qn)

        attacked = 0
        for checker in scan_reversed(sliders):
            attacked |= ray(king, checker) & ~cb_SqBoardS[checker]

        if cb_SqBoardS[king] & maskIn:
            for to_sqrt64 in scan_reversed(kingAttacks[king] & ~chessboard.occupied_co[chessboard.turn] & ~attacked & to_mask):
                yield Move(king, to_sqrt64)

        checker = msb(checkers)
        if cb_SqBoardS[checker] == checkers:
            target = between(king, checker) | checkers

            yield from chessboard.legaMovesGen(~chessboard.kings & maskIn, target & to_mask)

            if chessboard.ep_sqrt64 and not cb_SqBoardS[chessboard.ep_sqrt64] & target:
                last_double = chessboard.ep_sqrt64 + \
                    (-8 if chessboard.turn == WHITE else 8)
                if last_double == checker:
                    yield from chessboard.pseudoLegalEP(maskIn, to_mask)



    def generate_legal_captures(chessboard, maskIn: Board64 = BitBoardALL, to_mask: Board64 = BitBoardALL) -> Iterator[Move]:
        return itertools.chain(
            chessboard.calcLegalMoves(
                maskIn, to_mask & chessboard.occupied_co[not chessboard.turn]),
            chessboard.generate_legal_ep(maskIn, to_mask))

    def _attacked_for_king(chessboard, path: Board64, occupied: Board64) -> isTrue:
        return any(chessboard._attackers_mask(not chessboard.turn, sq, occupied) for sq in scan_reversed(path))

    def generate_castling_moves(chessboard, maskIn: Board64 = BitBoardALL, to_mask: Board64 = BitBoardALL) -> Iterator[Move]:
        if chessboard.isVariant_end():
            return

        backrank = Rank1 if chessboard.turn == WHITE else Rank8
        king = chessboard.occupied_co[chessboard.turn] & chessboard.kings & ~chessboard.promoted & backrank & maskIn
        king &= -king
        if not king:
            return

        cb_c = FileC & backrank
        cb_d = FileD & backrank
        cb_f = FileF & backrank
        cb_g = FileG & backrank

        for candidate in scan_reversed(chessboard.clean_CASTLEisTrue() & backrank & to_mask):
            rook = cb_SqBoardS[candidate]

            a_side = rook < king
            king_to = cb_c if a_side else cb_g
            rook_to = cb_d if a_side else cb_f

            king_path = between(msb(king), msb(king_to))
            rook_path = between(candidate, msb(rook_to))

            if not ((chessboard.occupied ^ king ^ rook) & (king_path | rook_path | king_to | rook_to) or
                    chessboard._attacked_for_king(king_path | king, chessboard.occupied ^ king) or
                    chessboard._attacked_for_king(king_to, chessboard.occupied ^ king ^ rook ^ rook_to)):
                yield chessboard._from_chessScrambler(chessboard.chessScrambler, msb(king), candidate)

    def transformer(chessboard, f: Callable[[Board64], Board64]) -> None:
        super().transformer(f)
        chessboard.clear_stack()
        chessboard.ep_sqrt64 = None if chessboard.ep_sqrt64 is None else msb(
            f(cb_SqBoardS[chessboard.ep_sqrt64]))
        chessboard.CASTLEisTrue = f(chessboard.CASTLEisTrue)

    def transform(chessboard: BoardT, f: Callable[[Board64], Board64]) -> BoardT:
        board = chessboard.copy(stack=False)
        board.transformer(f)
        return board

    def _from_chessScrambler(chessboard, chessScrambler: isTrue, from_sqrt64: sqrt64, to_sqrt64: sqrt64, toQueen: Optional[isPiece] = None, drop: Optional[isPiece] = None) -> Move:
        if not chessScrambler and toQueen is None and drop is None:
            if from_sqrt64 == E1 and chessboard.kings & cb_E1:
                if to_sqrt64 == H1:
                    return Move(E1, G1)
                elif to_sqrt64 == A1:
                    return Move(E1, C1)
            elif from_sqrt64 == E8 and chessboard.kings & cb_E8:
                if to_sqrt64 == H8:
                    return Move(E8, G8)
                elif to_sqrt64 == A8:
                    return Move(E8, C8)

        return Move(from_sqrt64, to_sqrt64, toQueen, drop)

    def _to_chessScrambler(chessboard, move: Move) -> Move:
        if move.from_sqrt64 == E1 and chessboard.kings & cb_E1:
            if move.to_sqrt64 == G1 and not chessboard.rk & cb_G1:
                return Move(E1, H1)
            elif move.to_sqrt64 == C1 and not chessboard.rk & cb_C1:
                return Move(E1, A1)
        elif move.from_sqrt64 == E8 and chessboard.kings & cb_E8:
            if move.to_sqrt64 == G8 and not chessboard.rk & cb_G8:
                return Move(E8, H8)
            elif move.to_sqrt64 == C8 and not chessboard.rk & cb_C8:
                return Move(E8, A8)

        return move

    def _transpositionKey(chessboard) -> Hashable:
        return (chessboard.pawns, chessboard.knights, chessboard.bishop, chessboard.rk,
                chessboard.qn, chessboard.kings,
                chessboard.occupied_co[WHITE], chessboard.occupied_co[BLACK],
                chessboard.turn, chessboard.clean_CASTLEisTrue(),
                chessboard.ep_sqrt64 if chessboard.containsPassant() else None)

    def __repr__(chessboard) -> str:
        if not chessboard.chessScrambler:
            return f"{type(chessboard).__name__}({chessboard.fen()!r})"
        else:
            return f"{type(chessboard).__name__}({chessboard.fen()!r}, chessScrambler=True)"



    def __eq__(chessboard, board: object) -> isTrue:
        if isinstance(board, Board):
            return (
                chessboard.HMoveclock == board.HMoveclock and
                chessboard.fmoveNum == board.fmoveNum and
                type(chessboard).uci_variant == type(board).uci_variant and
                chessboard._transpositionKey() == board._transpositionKey())
        else:
            return NotImplemented



    def apply_mirror(chessboard: BoardT) -> None:
        super().apply_mirror()
        chessboard.turn = not chessboard.turn

    def mirror(chessboard: BoardT) -> BoardT:
        board = chessboard.copy()
        board.apply_mirror()
        return board

    def copy(chessboard: BoardT, *, stack: Union[isTrue, int] = True) -> BoardT:
        board = super().copy()

        chessboard.chessboard_chess_ = chessboard.chessScrambler
        board.chessScrambler = chessboard.chessboard_chess_

        board.ep_sqrt64 = chessboard.ep_sqrt64
        board.CASTLEisTrue = chessboard.CASTLEisTrue
        board.turn = chessboard.turn
        num = chessboard.fmoveNum
        board.fmoveNum = num
        board.HMoveclock = chessboard.HMoveclock

        if stack:
            stack = len(chessboard.move_stack) if stack is True else stack
            board.move_stack = [copy.copy(move)
                                for move in chessboard.move_stack[-stack:]]
            board._stack = chessboard._stack[-stack:]

        return board

class Scraper:
    def __init__(chessboard) -> None:
        pass

    def parseDoc(chessboard):
        with open("index.html") as fp:
            soup = BeautifulSoup(fp, 'html.parser')

        soup = BeautifulSoup("<html>a web page</html>", 'html.parser')

        tag = soup.b
        type(tag)
        soup['id']

    def getURL(chessboard):
        URL = "https://realpython.github.io/fake-jobs/"
        page = requests.get(URL)
        print(page.text)



@dataclasses.dataclass
class Piece:
    Piecetype: isPiece
    color: Color

    def symbol(chessboard) -> str:
        symbol = pieceSymbol(chessboard.Piecetype)
        return symbol.upper() if chessboard.color else symbol

    def unicode_symbol(chessboard, *, invert_color: isTrue = False) -> str:
        symbol = chessboard.symbol().swapcase() if invert_color else chessboard.symbol()
        return asci_dict[symbol]

    def __hash__(chessboard) -> int:
        return chessboard.Piecetype + (-1 if chessboard.color else 5)

    def __repr__(chessboard) -> str:
        return "IS ERROR"
    def __str__(chessboard) -> str:
        return chessboard.symbol()


    @classmethod
    def from_symbol(cls, symbol: str) -> Piece:
        return cls(pieceSymbolS.index(symbol.lower()), symbol.isupper())


class PseudoLegalMoveGenerator:

    def __init__(chessboard, board: Board) -> None:
        chessboard.board = board

    def __isTrue__(chessboard) -> isTrue:
        return any(chessboard.board.legaMovesGen())

    def count(chessboard) -> int:
        # List conversion is faster than iterating.
        return len(list(chessboard))

    def __iter__(chessboard) -> Iterator[Move]:
        return chessboard.board.legaMovesGen()

    def __contains__(chessboard, move: Move) -> isTrue:
        return chessboard.board.is_pseudo_legal(move)

    def __repr__(chessboard) -> str:
        builder = []

        for move in chessboard:
            if chessboard.board.is_legal(move):
                builder.append(chessboard.board.san(move))


        sans = ", ".join(builder)
        return f"<PseudoLegalMoveGenerator at {id(chessboard):#x} ({sans})>"

boardtest = None
fentest=None
print(boardtest)

#N, S, E, W = -len(fentest.split('/')[0]) - 3, len(fentest.split('/')[0]) + 3, 1, -1


fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 '
board = {'P': 0b0000_0000_0000_0000_1111_1111_0000_0000,
         'R': 0b0000_0000_0000_0000_0000_0000_1000_0001,
         'N': 0b0000_0000_0000_0000_0000_0000_0100_0010,
         'B': 0b0000_0000_0000_0000_0000_0000_0010_0100,
         'Q': 0b0000_0000_0000_0000_0000_0000_0001_0000,
         'K': 0b0000_0000_0000_0000_0000_0000_0000_1000,

         'p': 0b0000_0000_1111_1111_0000_0000_0000_0000,
         'r': 0b1000_0001_0000_0000_0000_0000_0000_0000,
         'n': 0b0100_0010_0000_0000_0000_0000_0000_0000,
         'b': 0b0010_0100_0000_0000_0000_0000_0000_0000,
         'q': 0b0001_0000_0000_0000_0000_0000_0000_0000,
         'k': 0b0000_1000_0000_0000_0000_0000_0000_0000,
        }


print(f"{0xffff}" )
# Move the piece at position (0, 0) to position (2, 3)
#move_piece((0, 0), (2, 3))

"""test algorithm with simple solution
#then you make board
board = moveGen.Board(
    'r5rk/5p1p/5R2/4B3/8/8/7P/7K w KQ - 1 26')
#then you have list of moves with following line


moves = list(board.legal_moves)
#board.push_san('g1h3')

for i in moves:
    print(i)
print(board.fen())
print(board.is_checkmate())
bP = 0b0000_0000_0000_0000_1111_1111_0000_0000

#x |= 0b0000_0001_1000_0001_1000_0001_1000_0001
y = 0b0000_0000_0000_0000_1111_1111_0000_0000
x = bP | y


brds = []
brds2 = []
brds3 = []


for j in moves:
    temp = board.copy()
    temp.push_san(str(j))
    brds.append(temp.fen())

for i in brds:
    board = moveGen.Board(i)
    moves = list(board.legal_moves)
    for j in moves:
        temp = board.copy()
        temp.push_san(str(j))
        brds2.append(temp.fen())

for i in brds2:
    board = moveGen.Board(i)
    moves = list(board.legal_moves)
    for j in moves:
        temp = board.copy()
        temp.push_san(str(j))
        if temp.is_checkmate():
            print("YES")
            brds3.append(temp.fen())

for i in brds3:
    print(i)
#print(f"{bP:b}")

print(moveGen.Board('r5rk/5p2/7R/4B2p/7P/8/8/7K b - - 1 27').unicode())
"""
class LegalMoveGenerator:

    def __init__(chessboard, board: Board) -> None:
        chessboard.board = board

    def __isTrue__(chessboard) -> isTrue:
        return any(chessboard.board.calcLegalMoves())

    def count(chessboard) -> int:
        return len(list(chessboard))

    def __iter__(chessboard) -> Iterator[Move]:
        return chessboard.board.calcLegalMoves()

    def __contains__(chessboard, move: Move) -> isTrue:
        return chessboard.board.is_legal(move)

    def __repr__(chessboard) -> str:
        sans = ", ".join(chessboard.board.san(move) for move in chessboard)
        return f"<LegalMoveGenerator at {id(chessboard):#x} ({sans})>"


Intosqrt64Set = Union[SupportsInt, Iterable[sqrt64]]

"""
board flipping functions
in some cases the board needs to be flipped in order
to reuse moves and not need to recalculate the
bit masks"""
def verticalFlip(bb: Board64) -> Board64:
    bb = ((bb >> 8) & 0o3770017740077600377) | (
        (bb & 0o3770017740077600377) << 8)
    bb = ((bb >> 16) & 0o7777740000177777) | (
        (bb & 0o7777740000177777) << 16)
    bb = (bb >> 32) | ((bb & 0o37777777777) << 32)
    return bb


def horizontalFlip(bb: Board64) -> Board64:
    bb = ((bb >> 1) & 0o525252525252525252525) | (
        (bb & 0o525252525252525252525) << 1)
    bb = ((bb >> 2) & 0o314631463146314631463) | (
        (bb & 0o314631463146314631463) << 2)
    bb = ((bb >> 4) & 0o74170360741703607417) | (
        (bb & 0o74170360741703607417) << 4)
    return bb


def diagonalFlip(bb: Board64) -> Board64:
    t = (bb ^ (bb << 28)) & 0o74170360740000000000
    bb = bb ^ t ^ (t >> 28)
    t = (bb ^ (bb << 14)) & 0o314630000006314600000
    bb = bb ^ t ^ (t >> 14)
    t = (bb ^ (bb << 7)) & 0o524002520012500052400
    bb = bb ^ t ^ (t >> 7)
    return bb


def antiFlip(bb: Board64) -> Board64:
    t = bb ^ (bb << 0o44)
    bb = bb ^ ((t ^ (bb >> 0o44)) & 0o1703607417001703607417)
    t = (bb ^ (bb << 18)) & 0o1463140000031463000000
    bb = bb ^ t ^ (t >> 18)
    t = (bb ^ (bb << 9)) & 0o1463140000031463000000
    bb = bb ^ t ^ (t >> 9)
    return bb

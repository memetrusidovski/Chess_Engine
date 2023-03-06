# Mate-in-3 Chess Engine using Bitboards

This is a Python implementation of a chess engine that uses bitboards to efficiently represent the chess board and pieces. The engine is designed to search for mate-in-3 puzzles, meaning that it will look for a sequence of moves that results in a checkmate within three moves.

Prerequisites
Before running the chess engine, you will need to have the following installed on your machine:

-Python 3.x

# Usage
To use the chess engine, simply run the mate.py script. This will start the engine and prompt you to enter the FEN notation for the puzzle you want to solve.

# How it works
The chess engine uses bitboards to represent the state of the chess board and pieces. Bitboards are a way of representing a chess board using a set of 64-bit integers, with each bit in the integer representing a square on the board. This allows for fast and efficient manipulation of the board state.

The engine uses a depth-first search algorithm to search for mate-in-3 puzzles. At each depth, the engine generates all possible legal moves for the current player and evaluates the resulting board state. If a mate-in-3 solution is found, the engine outputs the sequence of moves needed to achieve checkmate.

The evaluation function used by the engine takes into account the material balance, piece mobility, and king safety for both players. This helps guide the search towards positions that are more likely to result in a mate-in-3 solution.

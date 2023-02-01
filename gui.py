from tkinter import *
import mate, moveGen, chessBoard
solved = {}

# START_PATTERN = "8/8/6p1/7k/3r2NP/B5PK/2br1R2/8 w 0 1"

def solveMate():
    # if len(entry.get()) == 0:
        
    #     empty = Label(
    #         text=("Nothing was submitted" % solutions),
    #         foreground="black",
    #         background="grey",
    #         height = 5, width = 50
    #     )
    
    # Checks if what is being entered into the prompt has not already been entered
    if solved.get(entry.get()) is None:
        # Calls the function which solves the the checkmate
        solutions = mate.callGUI(entry.get())

        # Passes the entry so that we can take the FEN string and use it as input for the UI board
        chessBoard.START_PATTERN = entry.get().replace('- ','')

        # Displays how many solutions there are to this puzzle that result in either mate in 1, 2 or 3
        Label(
            text=("There are %d solutions" % solutions),
            foreground="black",
            background="grey",
            height = 5, width = 50
        ).pack()

        solved[entry.get()] = 1
        
    return None


tk = Tk()

# The greeting label, tells the user what to input
greeting = Label(
    text="Give me a chess FEN string that is a mate in 1, 2 or 3 and I'll solve it for you",
    foreground="black",
    background="grey",
    height = 5, width = 100
)
# Sets the size of the window 
tk.geometry("%dx%d" % (tk.winfo_screenwidth(), tk.winfo_screenheight()))

entry = Entry(tk, width = 20)
button = Button(tk,text="Solve", command = solveMate)
greeting.pack()
entry.pack()
button.pack()
tk.mainloop()

# import chessBoard
import tkinter as tk

class GUI:
 
    foc = None
    images = {}
    lightColour = "#DDB88C"
    darkColour = "#A66D4F"
    rows = 8
    cols = 8
    size = 64
    hColour = "khaki"
    
    def __init__(self, parent, chessboard):
        self.chessboard = chessboard
        self.parent = parent

        canvWidth, canvHeight = self.cols * self.size,self.rows * self.size

        self.canvas = tk.Canvas(parent, height = canvHeight, width = canvWidth)

        self.canvas.pack(padx = 7, pady = 7)
        self.drawBoard()

    # A function that will draw and colour the chess board
    def drawBoard(self):
        colour = self.darkColour

        for r in range(self.rows):
       
            if colour == self.darkColour:
                colour = self.lightColour
            else:
                colour = self.darkColour

            for c in range(self.cols):
                x1, y1 = (c * self.size) , ((7 - r) * self.size)
                x2, y2 = x1 + self.size , y1 + self.size

                if (self.foc is not None and (r, c) in self.foc):
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill = self.hColour, tags="area")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill= colour, tags="area")
               
                if colour == self.darkColour:
                    colour = self.lightColour 
                else:
                    colour = self.darkColour

        self.canvas.tag_raise("occupied")
        self.canvas.tag_lower("area")

    def draw_pieces(self):
        self.canvas.delete("occupied")

        for crd, pce in self.chessboard.items():
            coor_x, coor_y = self.chessboard.num_notation(crd)
        
            if pce is not None:
                img = "pieceImages/%s%s.png" % (pce.shortname.lower(), pce.colour)
                piecename = "%s%s%s" % (pce.shortname, coor_x, coor_y)
        
                if img not in self.images:
                    self.images[img] = tk.PhotoImage(file=img)
        
                self.canvas.create_image(0, 0, image=self.images[img], tags=(piecename, "occupied"), anchor="c")
        
                x_ = (coor_y * self.size) + int(self.size / 2)
                y_ = ((7 - coor_x) * self.size) + int(self.size / 2)
                self.canvas.coords(piecename, x_, y_)

def main(chessboard):
    tkin = tk.Tk()
    tkin.title("CP 468: Term Project")
    gui = GUI(tkin, chessboard)
    gui.drawBoard()
    gui.draw_pieces()
    tkin.mainloop()


if __name__ == "__main__":
    game = chessBoard.Board()
    main(game)
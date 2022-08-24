import pandas as pd

class Curve():
    def __init__(self, ax, fig, col, lw=4, ls="-"):
        self.col = col

        if str(type(col))=="<class 'src.data_transform.GraphColumn.GraphColumn'>":
            self.plot = ax.plot(self.col.name,data=self.col.data, lw=lw, ls=ls)    
            self.legend = "{0} ({1};{2})".format(col.cell.type,col.cell.coord["X"],col.cell.coord["Y"])

        elif type(col) == pd.DataFrame:
            self.plot = ax.plot(col.columns[0], data=col, lw=lw, ls="dotted")
            self.legend = "Theoretical curve {0}".format(col.columns[0])
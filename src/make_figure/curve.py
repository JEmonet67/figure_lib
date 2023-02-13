import pandas as pd

class Curve():
    def __init__(self, ax, fig, col, lw=6, ls="-",plot_color=None):
        self.col = col

        if str(type(col)).split(".")[-1][0:-2]=="GraphColumn":
            if plot_color == None:
                self.plot = ax.plot(self.col.name,data=self.col.data, lw=lw, ls=ls)    
            else:
                self.plot = ax.plot(self.col.name,data=self.col.data, lw=lw, ls=ls,color=plot_color)    
            self.legend = "{0} ({1};{2})".format(col.cell.type,col.cell.coord["X"],col.cell.coord["Y"])

        elif type(col) == pd.DataFrame:
            if plot_color == None:
                self.plot = ax.plot(col.columns[0], data=col, lw=lw, ls="dotted")
            else:
                self.plot = ax.plot(col.columns[0], data=col, lw=lw, ls="dotted",color=plot_color)
            self.legend = "Theoretical curve {0}".format(col.columns[0])
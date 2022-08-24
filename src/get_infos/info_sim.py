

class InfoSim():
    def __init__(self, dfg, stim, grid):
        self.dfg=dfg
        self.stim = stim
        self.grid = grid


    
    def __repr__(self):
        text = "\n\t--- SIM ---\n### Time :\nBeginning : {0}\nEnd : {1}\n".format(round(self.dfg.data.index[0],4), round(self.dfg.data.index[self.dfg.data.shape[0]-1],4))
        text += self.grid.__repr__()
        text += self.stim.__repr__()
        
        return text





    # def save_param_file(self):

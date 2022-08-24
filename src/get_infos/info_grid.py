

class InfoGrid():
    def __init__(self,n_cells_x, grid_cell_height,n_cells_y=None, grid_cell_width=None):
        self.n_cells_x = n_cells_x
        self.grid_cell_height = grid_cell_height
        
        if type(n_cells_y) == None:
            self.n_cells_y = self.n_cells_x
        else:
            self.n_cells_y = n_cells_y
        
        if type(grid_cell_width) == None:
            self.grid_cell_width = self.grid_cell_height
        else:
            self.grid_cell_width = grid_cell_width
        
        self.cells_spacing_x = self.grid_cell_width/self.n_cells_x
        self.cells_spacing_y = self.grid_cell_height/self.n_cells_y



    def __repr__(self):
        text = "\n\t--- GRID ---\n### Cells :\n"
        text += "Nb cells => {0} ({1}x{2})\n".format(self.n_cells_x*self.n_cells_y, self.n_cells_x, self.n_cells_y)
        text += "Length => {0} mm || Width => {1} mm\n".format(self.grid_cell_width, self.grid_cell_height)
        text += "Cell Spacing => {0} mm (x) || {1} mm (y)\n".format(self.cells_spacing_x,self.cells_spacing_y)

        return text
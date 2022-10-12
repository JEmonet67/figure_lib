import pandas as pd
import math
from re import search, compile

class InfoCell():
    def __init__(self,name_col,n_cells_x, n_cells_y):
        self.num = self.get_num(name_col)
        self.type, self.output = self.get_name_output(name_col)
        self.coord = self.id_to_coordinates(n_cells_x, n_cells_y)
        # self.coord_mm = self.pixelsCoord_to_mmCoord(n_cells_x, n_cells_y)


    def __repr__(self):
        return "### {0} n°{1} :\n- Output = {2}\n- Layer = {3}\n- Position = ({4},{5})".format(self.type,self.num,self.output,self.coord["Z"],self.coord["X"],self.coord["Y"])

    def __eq__(self,cell):
        return str(self.num) + self.type + str(self.coord) == str(cell.num) + cell.type + str(cell.coord)

    def get_num(self,name_col):
        reg_num = compile(r'[\d]+')

        return int(search(reg_num,name_col)[0])

    def get_name_output(self,name_col):
        type_output = name_col.split(" ("+str(self.num)+") ")
        output = type_output[0]
        type_name = type_output[1]

        return type_name,output

    def id_to_coordinates(self,n_cells_x, n_cells_y):
        '''
        -------------
        Description :  
                Function to convert cell id into coordinates.
        -------------
        Arguments :
                numero -- int, Unique numero of the cell.
                n_cells -- int, Number of cells in the x axis of the grid cell.
        -------------
        Returns :
                Return a dictionary with X, Y and Z values.
        '''
        dict_coordinates = {}

        Z = (self.num)/n_cells_x/n_cells_y
        X = (Z - math.floor(Z)) * n_cells_x
        Y = (X - math.floor(round(X,2))) * n_cells_y

        dict_coordinates["Z"] = math.floor(Z)
        dict_coordinates["X"] = math.floor(round(X,2))
        dict_coordinates["Y"] = abs(math.floor(round(Y,2)))

        return dict_coordinates


    # def pixelsCoord_to_mmCoord(self,n_cells_x, n_cells_y):
    #     '''
    #     -------------
    #     Description :  
    #             Function to convert cell id into coordinates.
    #     -------------
    #     Arguments :
    #             numero -- int, Unique numero of the cell.
    #             n_cells -- int, Number of cells in the x axis of the grid cell.
    #     -------------
    #     Returns :
    #             Return a dictionary with X, Y and Z values.
    #     '''
    #     dict_coordinates_mm = {}
    #     dx = n_cells_x
    #     dy = n_cells_x

    #     dict_coordinates_mm["X"] = self.coord["X"]*dx
    #     dict_coordinates_mm["Y"] = self.coord["Y"]*dy
    #     dict_coordinates_mm["Z"] = self.coord["Z"]

    #     return dict_coordinates_mm
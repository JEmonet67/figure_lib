import math
import numpy as np



# TODO
#def get_center_coordinates():



# TODO
#def macular_id_to_coord(macular_id):



# TODO Be able to do horizontal and vertical lines to plot
def get_horizontal_interval_macular_cell(n_cells, layer, first_cell, last_cell, step_cell=1):
    """
    ### FUNCTION TO OBTAIN ID NUMBER OF A HORIZONTAL MACULAR CELL INTERVAL ###

        -- Input --
    n_cells : Tuple of cell number in X and Y axis.
    layer : Number of the macular layer contening cells to put in the interval.
    first_cell : X position of the first cell to save.
    last_cell : X position of the last cell to save.
    step_cell : Step of cell to save.

        -- Output --
    Str with number separate by commat without spacing : (132,244,424).

    """

    interval = ",".join \
        ([str(i) for i in range(n_cells[0 ] *n_cells[1 ] *layer +int(np.ceil(n_cells[1 ] /2 ) +n_cells[1 ] *first_cell) ,
                                               n_cells[0 ] *n_cells[1 ] *(layer +1 ) -n_cells[1 ] *(n_cells[0 ] -last_cell)
                               ,n_cells[1 ] *step_cell)])

    return interval


# TODO MODIFY
def id_to_coordinates(num, n_cells):  # Fonction pour transformer un id de cellule macular en coordonnées macular
    dict_coordinates = {}

    Z = (num) / n_cells[0] / n_cells[1]
    X = (Z - math.floor(Z)) * n_cells[0]
    Y = (X - math.floor(round(X, 2))) * n_cells[1]

    dict_coordinates["Z"] = math.floor(Z)
    dict_coordinates["X"] = math.floor(round(X, 2))
    dict_coordinates["Y"] = abs(math.floor(round(Y, 2)))

    return dict_coordinates

# TODO MODIFY
def convert_coord_macular_to_arrays_numpy(X_macular, Y_macular, n_cell):
    dict_coordinates = {}

    dict_coordinates["X"] = (n_cell[1] - 1) - Y_macular
    dict_coordinates["Y"] = X_macular

    return dict_coordinates

# TODO MODIFY
def convert_coord_arrays_numpy_to_macular(X_macular, Y_macular, n_cell):
    dict_coordinates = {}

    dict_coordinates["X"] = Y_macular
    dict_coordinates["Y"] = (n_cell[0] - 1) - X_macular

    return dict_coordinates
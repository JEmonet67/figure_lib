import re
import pandas as pd
import numpy as np
import coordinate_manager as cm
import src.data_type.dict_array as da


class MacularCsvConverter:
    """Convert data contains in a Macular csv file into a dictionary of arrays.

        This converter is made to handle the data format saved in csv files by Macular and transform it in a DictArray
        object. Each column of the csv file correspond to the output of a cell with a given cell type and id number :
        output (num) cell_type.
        A regular expression allow us to separate each of this information.

        One converter can be used to convert as many file as wanted in one time with multiple_converts or one by one
        with the convert method.

        """

    def __init__(self):
        """Create a macular data to csv converter. Use a regular expression to isolate the output, the ID number (num)
        and the cell type of each column of a Macular csv file."""
        self.reg_output_num_cell_type = re.compile(r"(.*?) \(([0-9]{1,5})\) (.*)")

    def multiple_converts(self, list_path_csv, grid_sizes):
        """Convert a list of csv file into DictArrays.

        Parameters
        ----------
        list_path_csv : list of str
            List containing path for each csv file to convert.

        grid_sizes : list of tuples
            Size of the grid size in a tuple of size two : (X_cells, Y_cells). It allows to know np.array sizes.

        Returns
        ----------
        list_dict_arr : list of DictArray
            List containing all the dictionary of array associated to each csv file given in input.

        """
        list_dict_arr = []
        for path_csv, grid_size in zip(list_path_csv, grid_sizes):
            list_dict_arr += [self.convert(path_csv, grid_size)]

        return list_dict_arr

    def convert(self, path_csv, grid_size):
        """Convert one csv file into DictArray.

        Parameters
        ----------
        path_csv : str
            Path of the csv file to convert.

        grid_size : tuple
            Size of the grid size in a tuple of size two : (X_cells, Y_cells). It allows to know np.array sizes.

        Returns
        ----------
        dict_arr : DictArray
            Dictionary of array associated to the csv file given in input.

        """
        csv_chunks = self.read_csv(path_csv)
        list_num, list_output_cell_type = self.get_output_num_cell_type_lists(self.get_name_columns(path_csv))
        dict_arr_data = self.empty_dict_list_maker(set(list_output_cell_type))
        list_arr_index = []

        for i_df, df in enumerate(csv_chunks):
            dict_arr_data, list_arr_index = self.chunk_loop(df, i_df, list_num, list_output_cell_type, dict_arr_data,
                                                            list_arr_index, grid_size)

        dict_arr_data, dict_arr_index = self.concatenate_chunk(dict_arr_data, {}, list_arr_index)
        dict_arr = da.DictArray(dict_arr_data, dict_arr_index)
        dict_arr.rotate(dict_arr_data, "counterclockwise")

        return dict_arr

    @staticmethod
    def concatenate_chunk(dict_arr_data, dict_arr_index, list_arr_index):

        arr_index = np.concatenate(list_arr_index)

        for key in dict_arr_data:
            dict_arr_index[key] = arr_index
            dict_arr_data[key] = np.concatenate(dict_arr_data[key], axix=-1)

        return dict_arr_data, dict_arr_index

    def set_up_chunk(self, df, dict_arr_data, grid_size):
        df = df.set_index("Time")
        df = df.sort_index(axis=1, key=lambda x: [
            (self.reg_output_num_cell_type.findall(elt)[0][2],
             self.reg_output_num_cell_type.findall(elt)[0][0],
             int(self.reg_output_num_cell_type.findall(elt)[0][1]))
            for elt in x.tolist()
        ])

        dict_arr_data = self.init_dict_arr_data(dict_arr_data, grid_size[0], grid_size[1], df.shape[0])

        return df, dict_arr_data

    @staticmethod
    def init_dict_arr_data(dict_arr_data, x_size, y_size, z_size):
        """Initialize the data dict array with an empty array of a given x, y and z size."""
        for key in dict_arr_data:
            dict_arr_data[key] += [np.zeros(x_size, y_size, z_size)]

        return dict_arr_data

    def chunk_loop(self, df, i_df, list_num, list_output_cell_type, dict_arr_data, list_arr_index, grid_size):
        df, dict_arr_data = self.set_up_chunk(df, dict_arr_data, grid_size)

        # Transfer index and data into list / dictionary
        list_arr_index += [df.index.to_numpy()]
        for output_cell_type in list_output_cell_type:
            self.transfer_macular_data_to_array(df, dict_arr_data(output_cell_type[i_df]), list_num, grid_size)

        return dict_arr_data, list_arr_index

    @staticmethod
    def transfer_macular_data_to_array(df, data_array, list_num, grid_size):

        for i, num in enumerate(list_num):
            dict_coord_macular = cm.id_to_coordinates(num, (grid_size[0], grid_size[1]))
            data_array[dict_coord_macular["X"], dict_coord_macular["Y"]] = df.iloc[i].to_numpy()

        return data_array

    def get_output_num_cell_type_lists(self, names_col):
        """Make lists of ID number, output and cell type based on a list of all the columns name in a Macular csv file.

        Each macular data column name are separated into two lists, one for ID number alone and a second for output/cell
        type together. Indexes of both list correspond of the index of the column in the csv file.

        Parameters
        ----------
        names_col : list of str
            List of all the column present in the csv Macular file.

        Returns
        ----------
        list_num : list of int
            List all ID number in the given list of a file header.

        list_output_cell_type : list of str
            List all output_cell_type in the given list of a file header.

        """
        list_num = []
        list_output_cell_type = []

        for col in names_col:
            # Output, number and type name extraction from column names
            (output, num, cell_type) = self.reg_output_num_cell_type.findall(col)[0]
            list_num += [int(num)]
            list_output_cell_type += [f"{output}_{cell_type}"]

        return list_num, list_output_cell_type

    @staticmethod
    def empty_dict_list_maker(set_output_cell_type):
        """Create a dictionary of empty list where each key the output and the cell type present in a given set.
        This set have to be created base on the columns of a Macular csv file.

        Parameters
        ----------
        set_output_cell_type : set of str
            Set of each "output_celltype" pair presents in the columns name of the Macular csv file.

        Returns
        ----------
        dict of empty list
            Dictionary containing each output_celltype" pair as a key associated to an empty list.

        """
        return {output_cell_type: [] for output_cell_type in set_output_cell_type}

    @staticmethod
    def read_csv(path_csv):
        """Read a csv file by parallelized chunk of size 2000."""
        return pd.read_csv(path_csv, chunksize=2000)

    @staticmethod
    def get_name_columns(path_csv):
        """Return the header of the given csv file in order to get his column name list."""
        with open(path_csv) as f:
            return f.readline().replace("\"", "").split(",")[1:]

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

        A counterclockwise rotation is done to correct a difference between the coordinate system of Macular and numpy.

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
        dicts_num_output_cell_type_index = self.get_output_num_cell_type_dict(self.get_name_columns(path_csv))
        list_num, list_output_cell_type = self.get_output_num_cell_type_lists(self.get_name_columns(path_csv))

        dict_arr_data = self.empty_dict_list_maker(set(dicts_num_output_cell_type_index))
        list_arr_index = []

        for i_df, df in enumerate(csv_chunks):
            dict_arr_data, list_arr_index = self.chunk_loop(df, i_df, dicts_num_output_cell_type_index, dict_arr_data,
                                                            list_arr_index, grid_size)

        dict_arr_data, dict_arr_index = self.concatenate_chunk(dict_arr_data, {}, list_arr_index)
        dict_arr = da.DictArray(dict_arr_data, dict_arr_index)
        dict_arr.rotate_data(direction="counterclockwise")

        return dict_arr

    @staticmethod
    def concatenate_chunk(dict_arr_data, dict_arr_index, list_arr_index):
        """Concatenate all data and index arrays list into one data and one index array.

        Data and index arrays are store in two different dictionary of arrays with the same "output_celltype" keys.

        Parameters
        ----------
        dict_arr_data : dict of list of numpy.arrays.
            Dictionary with "output_celltype" keys and containing a list used to store all data chunk arrays.

        dict_arr_index : empty dict
            Empty dictionary that will receive the unidimensional concatenated index array.

        list_arr_index : list of numpy.arrays
            List to store time index of each chunck dataframe.

        Returns
        ----------
        dict_arr_data : dict of numpy.arrays.
            Dictionary with "output_celltype" keys and containing one data array.

        dict_arr_index : dict of arrays
            Dictionary with "output_celltype" keys and containing unidimensional index arrays.

        """

        arr_index = np.concatenate(list_arr_index)

        for key in dict_arr_data:
            dict_arr_index[key] = arr_index
            dict_arr_data[key] = np.concatenate(dict_arr_data[key], axix=-1)

        return dict_arr_data, dict_arr_index

    def set_up_chunk(self, df, dict_arr_data, grid_size):
        """Prepare the dataframe chunk and new arrays destined to receive their data.

        The dataframe chunk need to set his index to the time column. The dictionary of array must get
        new empty arrays of the simulation cell grid size for each pairs of output-cell type. This chunk are distributed
        in a list store in the dictionary dict_arr_data.
²
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe corresponding to a chunk of the csv Macular file data without index.

        dict_arr_data : dict of list empty or fill with numpy.arrays.
            Dictionary with "output_celltype" keys and containing a list used to store all data chunk arrays. This list
            can be empty or already fill with arrays.

        grid_size : tuple
            Size of the grid size in a tuple of size two : (X_cells, Y_cells). It allows to know np.array sizes.

        Returns
        ----------
        df : pandas.DataFrame
            Dataframe corresponding to a chunk of the csv Macular file data with a time index.

        dict_arr_data : dict of list of numpy.arrays.
            Dictionary with "output_celltype" keys and containing a list used to store all data chunk arrays.

        """
        df = df.set_index("Time")

        dict_arr_data = self.new_data_dict_chunk_arr(dict_arr_data, grid_size[0], grid_size[1], df.shape[0])

        return df, dict_arr_data

    @staticmethod
    def new_data_dict_chunk_arr(dict_arr_data, x_size, y_size, z_size):
        """Add a new empty chunk array of a given x, y and z size in the data dictionary chunk list."""
        for key in dict_arr_data:
            dict_arr_data[key] += [np.zeros(x_size, y_size, z_size)]

        return dict_arr_data

    def chunk_loop(self, df, i_df, dicts_num_output_cell_type_index, dict_arr_data, list_arr_index, grid_size):
        """Extract data and index in each data chunk before putting it in the two corresponding storage list of chunk
        arrays.

        One loop is done for each dataframe chunk created in the parallelized reading of the current csv file. The first
        part of the loop correspond to his set-up.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe corresponding to a chunk of the csv Macular file data without index.

        i_df : int
            Index of the data chunk extract from a Macular csv file. It's used in order to know in which array put data
            in the list of the data dict arrays.

        dicts_num_output_cell_type_index : dict of dict of int
            Dictionary of "output_celltype" containing dictionary of ID number where columns index are set.

        dict_arr_data : dict of list of numpy.arrays.
            Dictionary with "output_celltype" keys and containing a list used to store all data chunk arrays. This list
            can be empty at the beginning.

        list_arr_index : list of numpy.arrays
            List to store time index of each chunck dataframe. At the beginning, this list can be empty.

        grid_size : tuple
            Size of the grid size in a tuple of size two : (X_cells, Y_cells). It allows to know np.array sizes.

        Returns
        ----------
        dict_arr_data : dict of list of numpy.arrays.
            Dictionary with "output_celltype" keys and containing a list used to store all data chunk arrays.

        list_arr_index : list of numpy.arrays
            List to store time index of each chunck dataframe.

        """
        # Prepare df and dict_arr_data to be used
        df, dict_arr_data = self.set_up_chunk(df, dict_arr_data, grid_size)

        # Transfer index and data into list / dictionary
        list_arr_index += [df.index.to_numpy()]
        dict_arr_data = self.copy_macular_data_chunk_to_list_array(df, i_df, dicts_num_output_cell_type_index,
                                                                   dict_arr_data)

        return dict_arr_data, list_arr_index

    def copy_macular_data_chunk_to_list_array(self, df, i_df, dicts_num_output_cell_type_index, dict_arr_data):
        """Copy all the data of a chunk dataframe into a list of array in the dictionary of array.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe corresponding to a chunk of the csv Macular file data without index.

        i_df : int
            Index of the data chunk extract from a Macular csv file. It's used in order to know in which array put data
            in the list of the data dict arrays.

        dicts_num_output_cell_type_index : dict of dict of int
            Dictionary of "output_celltype" containing dictionary of ID number where columns index are set.

        dict_arr_data : dict of list of numpy.arrays.
            Dictionary with "output_celltype" keys and containing a list used to store all data chunk arrays. This list
            can be empty at the beginning.

        Returns
        ----------
        dict_arr_data : dict of list of numpy.arrays.
            Dictionary with "output_celltype" keys and containing a list used to store all data chunk arrays.


        """
        for output_cell_type in dicts_num_output_cell_type_index:
            dict_arr_data[output_cell_type][i_df] = self.copy_macular_data_to_array(df,
                                                    dict_arr_data[output_cell_type][i_df],
                                                    dicts_num_output_cell_type_index[output_cell_type])

        return dict_arr_data


    @staticmethod
    def copy_macular_data_to_array(df, data_array, dict_num_index):
        """Copy all the data chunk of a given "output_celltype" pair into the corresponding array.

        Data are collected in the csv Macular file dataframe based on their index associated to their ID number and
        store in a dictionary. The ID number have to be converted into X and Y macular coordinates to know where to put
        each chunk dataframe columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe corresponding to a chunk of the csv Macular file data with time index.

        data_array : numpy.Array
            Empty array of which x, y and z size are the same dimension as the Macular simulation (cell number in X,
            cell number in Y and time).

        dict_num_index : list of int
            Dictionary of ID number keys associated to their corresponding index columns in the Macular csv file.

        Returns
        ----------
        data_array : np.array
            Array of size x,y,z and filled with the data of the corresponding output_num_celltype column of the csv
            file.

        """
        for num, index in dict_num_index:
            dict_coord_macular = cm.id_to_coordinates(num, (data_array.shape[0], data_array.shape[1]))
            data_array[dict_coord_macular["X"], dict_coord_macular["Y"]] = df.iloc[index].to_numpy()

        return data_array

    def get_output_num_cell_type_lists(self, names_col): #TODO Suppress it
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
            List of all ID number in the file header of a Macular csv file.

        list_output_cell_type : list of str
            List of all "output_celltype" present in the file header of a Macular csv file.

        """
        list_num = []
        list_output_cell_type = []

        for col in names_col:
            # Output, number and type name extraction from column names
            (output, num, cell_type) = self.reg_output_num_cell_type.findall(col)[0]
            list_num += [int(num)]
            list_output_cell_type += [f"{output}_{cell_type}"]

        return list_num, list_output_cell_type


    def get_output_num_cell_type_dict(self, names_col):
        """Make dictionaries of ID number and output-cell type associated to the index of the corresponding columns
        in a Macular csv file.

        Each pairs of "output_celltype" are keys of a first dictionary which contains a second dictionary with ID number
        keys. Each macular data column index are store in these combined dictionaries.

        Parameters
        ----------
        names_col : list of str
            List of all the column present in the csv Macular file.

        Returns
        ----------
        dicts_num_output_cell_type_index : dict of dict of int
            Dictionary of "output_celltype" containing dictionary of ID number where columns index are set.

        """
        dicts_num_output_cell_type_index = {}

        for i, col in enumerate(names_col):
            (output, num, cell_type) = self.reg_output_num_cell_type.findall(col)[0]
            num = int(num)
            output_cell_type = f"{output}_{cell_type}"

            try:
                dicts_num_output_cell_type_index[output_cell_type][num] = i
            except KeyError:
                dicts_num_output_cell_type_index[output_cell_type] = {num: i}

        return dicts_num_output_cell_type_index


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
            Dictionary containing each "output_celltype" pair as a key associated to an empty list.

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

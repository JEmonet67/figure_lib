import numpy as np


class DictArray:
    """A pair of index and data dictionaries fill with arrays.

        This database structure allow to store multiple dataset. Each dataset is divide into a multidimensional data
        array and a unidimensional index array. Both are put in two different dictionaries. Each dataset will be
        associated to one name used for dictionaries keys.

        Index and data dictionaries must have the same keys and the same size of their last axis.

        Parameters
        ----------
        dict_arr_data : dict of numpy.arrays
            Dictionary of arrays containing data arrays.

        dict_arr_index : dict of 1D numpy.arrays
            Dictionary of arrays containing index arrays corresponding to data arrays.

        Attributes
        ----------
        index : dict of numpy.arrays

        data : dict of 1D numpy.arrays
        """

    def __init__(self, dict_arr_data, dict_arr_index):
        """Create a DictArray from both data and index dictionaries arrays."""
        self._data = None
        self._index = None
        self.set_data_index(dict_arr_data, dict_arr_index)

    @property
    def data(self):
        """Method to return the data dictionary array."""
        return self._data

    @data.setter
    def data(self, dict_array):
        """Method to modify the data dictionary array.

        Parameters
        ----------
        dict_array : dict of numpy.array
            Dictionary of arrays containing data arrays.

        Returns
        ----------
        None

        Raises
        ----------
        TypeError
            Input isn't a dictionary of array.

        KeyError
            Input dictionary array haven't the same keys as the current index array.

        IndexError
            Arrays in the input dictionary haven't the same last axis as the current index array.
        """
        self.check_input_type(dict_array)
        self.check_keys_compatibility(dict_array, self._index)
        self.check_input_size(dict_array)
        self._data = dict_array

    @property
    def index(self):
        """Method to return the index dictionary array."""
        return self._index

    @index.setter
    def index(self, dict_array):
        """Method to modify the index dictionary array.

        Parameters
        ----------
        dict_array : dict of numpy.array
            Dictionary of arrays containing index arrays.

        Returns
        ----------
        None

        Raises
        ----------
        TypeError
            Input isn't a dictionary of array.

        KeyError
            Input dictionary array haven't the same keys as the current data array.

        IndexError
            Arrays in the input dictionary haven't the same last axis as the current index array.
            Arrays in the input dictionnary are not unidimensional.
        """
        self.check_input_type(dict_array)
        self.check_keys_compatibility(dict_array, self._data)
        self.check_input_size(dict_array)
        self.check_input_index_unidimensionality(dict_array)
        self._index = dict_array

    def set_data_index(self, dict_arr_data, dict_arr_index):
        """Method to modify the index dictionary array.

        Parameters
        ----------
        dict_arr_data : dict of numpy.array
            Dictionary of arrays containing data arrays.

        dict_arr_index : dict of numpy.array
            Dictionary of arrays containing index arrays.

        Returns
        ----------
        None

        Raises
        ----------
        TypeError
            Input isn't a dictionary of array.

        KeyError
            Data or index dictionary array haven't the same keys as the current index or data array.

        IndexError
            Data or index arrays in the input dictionary haven't the same last axis as the current index array.
            Index arrays in the input dictionnary are not unidimensional.
        """
        self.check_input_type(dict_arr_index)
        self.check_input_index_unidimensionality(dict_arr_index)
        self._index = dict_arr_index
        self.data = dict_arr_data

    @staticmethod
    def check_input_type(dict_array):
        """Check if a given input is a dictionary of arrays"""
        if not type(dict_array) == dict:
            raise TypeError("Dict array should be a dictionary.")

        for key_array in dict_array:
            if type(dict_array[key_array]) != np.ndarray:
                raise TypeError("Dict array dictionary should contain array.")

    @staticmethod
    def check_keys_compatibility(dict_array1, dict_array2):
        """Check if two dictionaries of arrays have the same set of keys."""
        if set(dict_array1) != set(dict_array2):
            raise KeyError("Index and data arrays should have the same keys.")

    def check_input_size(self, dict_array):
        """Check if a given dictionary of arrays have the same last axis as the current index dictionary arrays."""
        for key_array in dict_array:
            if dict_array[key_array].shape[-1] != self.index[key_array].shape[0]:
                raise IndexError("Index array dimension incompatible with data dictionary array dimension.")

    @staticmethod
    def check_input_index_unidimensionality(dict_index):
        """Check if a given dictionary of arrays is unidimensional."""
        for key_array in dict_index:
            if len(dict_index[key_array].shape) != 1:
                raise IndexError("Index array should be unidimensional.")

    def __getitem__(self, key):
        """Return value of the item : DictArray[key]."""
        return self.data[key], self.index[key]

    def __setitem__(self, key, arrays):
        """Add or change value of a given item : DictArray[key]

        Parameters
        ----------
        key : str
            Key corresponding to the item to add or modify.

        arrays : tuple of np.arrays
            Tuple of size two containing the data array in first position and the index array in last position.

        Returns
        ----------
        None

        Raises
        ----------
        IndexError
            Data and index arrays have not the same last axis size.
        """
        if arrays[0].shape[-1] == arrays[1].shape[-1]:
            self.data[key] = arrays[0]
            self.index[key] = arrays[1]
        else:
            raise IndexError("Wrong axis size for index and data to put in DictArray.")

    def remove(self, key):
        """Remove corresponding item : DictArray[key]."""
        del self.index[key]
        del self.data[key]

    def __repr__(self):
        """Display index and data dictionary arrays of a DictArray."""
        return f"index : {self.index}\ndata : {self.data}"

    def rotate_data(self, direction):
        """Rotate data dictionary arrays in the clockwise or counterclockwise direction."""
        if direction == "clockwise":
            for key in self.data:
                self.data[key] = np.rot90(self.data[key], axes=(1, 0))
        elif direction == "counterclockwise":
            for key in self.data:
                self.data[key] = np.rot90(self.data[key], axes=(0, 1))
        else:
            raise "Error in direction for rotating DictArray"

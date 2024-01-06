import src.data_type.dict_array as da
import numpy as np
import pytest

"""Tests to verify the good functioning of the DictArray class."""


def test_dict_array_creation():
    """ Test that the dict array can be create without any issues."""
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx = {"a": np.ones(20), "b": np.zeros(20)}
    dict_arr = da.DictArray(dict_data, dict_idx)

    assert dict_arr.data == dict_data
    assert dict_arr.index == dict_idx


def test_dict_array_should_receive_dict_array():
    """Test to see if create a dict array with wrong input raise errors."""
    dict_wrong = {"a": 2, "b": 3}
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx = {"a": np.ones(20), "b": np.zeros(20)}

    with pytest.raises(TypeError):
        da.DictArray(dict_wrong, dict_idx)
    with pytest.raises(TypeError):
        da.DictArray(dict_data, dict_wrong)
    with pytest.raises(TypeError):
        da.DictArray(dict_data, 1)
    with pytest.raises(TypeError):
        da.DictArray(1, dict_idx)


def test_input_should_have_same_size_as_index():
    """Test to see if index or data input of the wrong size raise errors."""
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx = {"a": np.ones(20), "b": np.zeros(20)}
    dict_arr = da.DictArray(dict_data, dict_idx)

    dict_wrong_idx = {"a": np.ones(30), "b": np.zeros(30)}
    dict_wrong_data = {"a": np.ones((10, 10, 30)), "b": np.zeros((10, 10, 30))}
    dict_correct_data = {"a": np.ones((10, 20)), "b": np.zeros((10, 20))}

    with pytest.raises(IndexError):
        dict_arr.data = dict_wrong_data
    with pytest.raises(IndexError):
        dict_arr.index = dict_wrong_idx
    dict_arr.data = dict_correct_data
    assert dict_arr.data == dict_correct_data

    dict_arr.data = dict_correct_data
    assert dict_arr.data == dict_correct_data


def test_index_should_be_unidimensional():
    """Test that non-unidimensional raise errors."""
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx = {"a": np.ones((20, 20)), "b": np.zeros((20, 20))}
    with pytest.raises(IndexError):
        da.DictArray(dict_data, dict_idx)


def test_able_to_modify_index_and_data_at_the_same_time():
    """Test to verify if we can modify data and index at the same time with set_data_index method."""
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx = {"a": np.ones(20), "b": np.zeros(20)}
    dict_arr = da.DictArray(dict_data, dict_idx)

    dict_new_data = {"a": np.ones((10, 20)), "b": np.zeros((10, 20))}
    dict_new_array = {"a": np.ones(20), "b": np.zeros(20)}

    dict_arr.set_data_index(dict_new_data, dict_new_array)
    assert dict_arr.data == dict_new_data
    assert dict_arr.index == dict_new_array


def test_data_should_have_same_keys_as_index():
    """Test that data without the same keys as index raises an error."""
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx_more = {"a": np.ones(20), "b": np.zeros(20), "c": np.zeros(20)}
    dict_idx_less = {"a": np.ones(20)}
    dict_idx_diff = {"c": np.ones(20), "d": np.zeros(20)}

    with pytest.raises(KeyError):
        da.DictArray(dict_data, dict_idx_more)
    with pytest.raises(KeyError):
        da.DictArray(dict_data, dict_idx_less)
    with pytest.raises(KeyError):
        da.DictArray(dict_data, dict_idx_diff)


def test_index_should_have_same_keys_as_data():
    """Test that index without the same keys as data raises an error."""
    dict_idx = {"a": np.ones(20), "b": np.zeros(20)}
    dict_data_correct = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_data_more = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20)), "c": np.zeros((10, 10, 20))}
    dict_data_less = {"a": np.ones((10, 10, 20))}
    dict_data_diff = {"c": np.ones((10, 10, 20)), "d": np.zeros((10, 10, 20))}

    dict_array = da.DictArray(dict_data_correct, dict_idx)

    with pytest.raises(KeyError):
        dict_array.data = dict_data_more
    with pytest.raises(KeyError):
        dict_array.data = dict_data_less
    with pytest.raises(KeyError):
        dict_array.data = dict_data_diff


def test_get_item():
    """Test if get_item method is functioning."""
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx = {"a": np.ones(20), "b": np.zeros(20)}
    dict_arr = da.DictArray(dict_data, dict_idx)
    assert dict_arr["a"] == (dict_data["a"], dict_idx["a"])


def test_modify_item():
    """Test to modify item with the __setitem__ method."""
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx = {"a": np.ones(20), "b": np.zeros(20)}
    dict_arr = da.DictArray(dict_data, dict_idx)

    dict_arr["a"] = (np.ones((10, 10, 10)), np.ones(10))
    assert np.array_equal(dict_arr.data["a"], np.ones((10, 10, 10)))
    assert np.array_equal(dict_arr.index["a"], np.ones(10))


def test_add_item():
    """Test to add a new item with the __setitem__ method."""
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx = {"a": np.ones(20), "b": np.zeros(20)}
    dict_arr = da.DictArray(dict_data, dict_idx)

    dict_arr["c"] = (np.ones((10, 10, 5)), np.ones(5))
    assert np.array_equal(dict_arr.data["c"], np.ones((10, 10, 5)))
    assert np.array_equal(dict_arr.index["c"], np.ones(5))
    assert set(dict_arr.index) == {"a", "b", "c"}
    assert set(dict_arr.data) == {"a", "b", "c"}


def test_remove_item():
    """Test to remove a item from a DictArray."""
    dict_data = {"a": np.ones((10, 10, 20)), "b": np.zeros((10, 10, 20))}
    dict_idx = {"a": np.ones(20), "b": np.zeros(20)}
    dict_arr = da.DictArray(dict_data, dict_idx)

    dict_arr.remove("a")
    assert set(dict_arr.index) == {"b"}
    assert set(dict_arr.data) == {"b"}
    assert np.array_equal(dict_arr.index["b"], np.zeros(20))
    assert np.array_equal(dict_arr.data["b"], np.zeros((10, 10, 20)))

def test_rotate_data():
    """Test to check proper functioning of rotate_data method."""
    arr = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    arr_ccw = np.array(((3, 6, 9), (2, 5, 8), (1, 4, 7)))
    arr_cw = np.array(((7, 4, 1), (8, 5, 2), (9, 6, 3)))

    dict_data = {"a": arr, "b": arr}
    dict_idx = {"a" : np.ones(3), "b": np.ones(3)}
    dict_arr = da.DictArray(dict_data, dict_idx)

    dict_arr.rotate_data("clockwise")
    assert np.array_equal(dict_arr.data["a"], arr_cw)
    assert np.array_equal(dict_arr.data["b"], arr_cw)
    dict_arr.rotate_data("counterclockwise")
    dict_arr.rotate_data("counterclockwise")
    assert np.array_equal(dict_arr.data["a"], arr_ccw)
    assert np.array_equal(dict_arr.data["b"], arr_ccw)

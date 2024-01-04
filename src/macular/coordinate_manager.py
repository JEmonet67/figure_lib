import math


def distance(point1, point2):
    """Compute euclidian distance between two points.

    Parameters
    ----------
    point1 : tuple
        Coordinates of the first point.

    point2 : tuple
        Coordinates of the second point.

    Returns
    ----------
    float
        Distance value between the two points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def mm_cortex_to_deg(value_mm):
    """
    Convert a value from millimeters of cortex to degrees in monkeys.
    """
    return value_mm / 3


def mm_retina_to_deg(value_mm):
    """
    Convert a value from millimeters of retina to degrees in monkeys.
    """
    return value_mm / 0.3


def deg_to_mm_cortex(value_deg):
    """
    Convert a value from degrees to millimeters of cortex in monkeys.
    """
    return value_deg * 3


def deg_to_mm_retina(value_deg):
    """
    Convert a value from degrees to millimeters of retina in monkeys.
    """
    return value_deg * 0.3


def coordinates_to_macular_id(x, y, z, n_cells):
    """Convert coordinates into macular ID.

    Parameters
    ----------
    x, y, z : float
        Coordinates x, y and z of the macular cell.

    n_cells : tuple
        Number of cells (x, y) in the macular grid.

    Returns
    ----------
    int
        ID of the Macular cell.
    """
    return n_cells[0]*n_cells[1]*z + (n_cells[1] * x + y)


def macular_id_to_coordinates(num, n_cells):
    """Convert macular ID into a coordinates dictionary.

    Parameters
    ----------
    num : int
        Macular ID of the cell.

    n_cells : tuple
        Number of cells (x, y) in the macular grid.

    Returns
    ----------
    dict
        Dictionary containing a key X, Y and Z associated to their value.
    """
    z = num / n_cells[0] / n_cells[1]
    x = (z - math.floor(z)) * n_cells[0]
    y = (x - math.floor(round(x, 2))) * n_cells[1]

    return {"X": math.floor(round(x, 2)), "Y": abs(math.floor(round(y, 2))), "Z": math.floor(z)}


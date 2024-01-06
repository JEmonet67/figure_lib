import numpy as np
import src.macular.coordinate_manager as cm


def test_distance_compute():
    """Test of a simple distance computation."""
    assert cm.distance((4, 5), (10, 2)) == np.sqrt(45)


def test_mm_cortex_to_deg():
    """Test of the conversion from mm cortex to degrees."""
    assert cm.mm_cortex_to_deg(9.45) == 3.15


def test_mm_retina_to_deg():
    """Test of the conversion from mm retina to degrees."""
    assert cm.mm_retina_to_deg(0.945) == 3.15


def test_deg_to_mm_cortex():
    """Test of the conversion from degrees to mm cortex."""
    assert cm.deg_to_mm_cortex(3.15) == 9.45


def test_deg_to_mm_retina():
    """Test of the conversion from degrees to mm retina."""
    assert cm.deg_to_mm_retina(3.15) == 0.945


def test_id_to_coordinates():
    """Tests of the macular id to coordinates conversion."""
    assert cm.macular_id_to_coordinates(307, (41, 15)) == {"X": 20, "Y": 7, "Z": 0}
    assert cm.macular_id_to_coordinates(2767, (41, 15)) == {"X": 20, "Y": 7, "Z": 4}
    assert cm.macular_id_to_coordinates(0, (41, 15)) == {"X": 0, "Y": 0, "Z": 0}
    assert cm.macular_id_to_coordinates(614, (41, 15)) == {"X": 40, "Y": 14, "Z": 0}


def test_compute_cell_id():
    """Tests of the coordinates to macular id conversion."""
    assert cm.coordinates_to_macular_id(20, 7, 0, (41, 15)) == 307
    assert cm.coordinates_to_macular_id(20, 7, 4, (41, 15)) == 2767
    assert cm.coordinates_to_macular_id(0, 0, 0, (41, 15)) == 0
    assert cm.coordinates_to_macular_id(40, 14, 0, (41, 15)) == 614

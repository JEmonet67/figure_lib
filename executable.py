import sys
sys.path.append("/home/jemonet/Documents/These/Code/figure_lib/src")
sys.path.append("/home/jemonet/Documents/These/Code")
sys.path.append("/user/jemonet/home/Documents/These/stimuli")
import preprocessing as pre
import graphs as gr

if __name__ == '__main__':
    # DEFAULT PARAMS TO CHANGE BETWEEN DIRECTORY OR FILE
    path_rep = "/user/jemonet/home/Documents/These/Data Macular/GaussianCorrection_PvaluesAnticipation/semiBioGraph_extDriveParams/41x5c/horizontal_white_bar/bar0,67x0,9°_6dps/"
    caract = "6"
    name_caract = "barSpeed"
    unit_caract = "dps"
    path_file = f"{name_caract}/RC_PvA_sBGeDP0004_{name_caract}{caract}{unit_caract}_9f.csv"  # TO CHANGE

    path = path_rep + path_file
    dict_re = pre.compile_regexp()
    image_extension = f"_{name_caract}{caract}{unit_caract}"

    params_sim = {
        "dx": 0.225,  # degrees
        "delta_t": 0.0167,  # seconds  TO CHANGE
        "dt": 0.0004,  # seconds
        "speed": 6,  # °/s
        "size_bar": 0.67,  # main axis size bar, °   TO CHANGE
        "n_transient_frame": 9,  # TO CHANGE
        "n_cells_X": 41,  # TO CHANGE
        "n_cells_Y": 5,  # TO CHANGE
        "axis": 0  # 0 for horizontal and 1 for vertical.
    }

    info_fig = {
        "title": f"Simulation {params_sim['n_cells_X']}x{params_sim['n_cells_Y']} with horizontal white bar of 0,9x0,66° moving at {caract}°/s",
        # "title":f"Simulation {params_sim['n_cells_X']}x{params_sim['n_cells_Y']} with horizontal white bar of 0,9x0,66° moving at 6°/s\n{name_caract} {caract}{unit_caract}",
        "subtitles": "",
        "image_name": "",  # TO CHANGE
        "xlabel": "Time (ms)",
        "ylabel": [""],  # TO CHANGE
        "sharex": True,
        "sharey": False,
        "legend": ["coord_degree"],  # TO CHANGE
        "postprod": {}
    }

    params_fig = {
        "wspace": 0.2,
        "hspace": 0.4,
        "height_ratios": [1, 1],
        "width_ratios": [1, 1]
    }

    font_size = {
        "main_title": 35,
        "subtitle": 25,
        "xlabel": 35,
        "ylabel": 35,
        "g_xticklabel": 35,
        "g_yticklabel": 35,
        "legend": 20
    }

    params_plot = {
        "grid_color": "lightgray",
        "grid_width": 4,
        "ticklength": 7,
        "tickwidth": 3,
        "labelpad": 10,
        "Xlim": [()],  # TO CHANGE
        "Ylim": [()],  # TO CHANGE
        "plots_color": [[(0, 0, 0)]],  # TO CHANGE
        "center": False
    }

    # PARAMS TO CHANGE BETWEEN EACH GRAPH
    info_fig["legend"] = ["coord_degree", "coord_degree"]
    params_plot["Xlim"] = [(), ()]
    params_plot["Ylim"] = [(), ()]
    font_size["legend"] = 15
    params_plot["center"] = False
    info_fig["postprod"] = {}

    info_cells = {
        "name_output": ["v_e", "v_i"],  # BipolarResponse, V, FiringRate, muVn, v_e, v_i, VSDI
        "name_cell": ["CorticalExcitatory", "CorticalInhibitory"],
        # BipolarGainControl, GanglionGainControl, Amacrine, CorticalExcitatory, CorticalInhibitory
        "num": [-1, -1],
        "layer": [3, 4]
    }

    info_fig["image_name"] = "ve_vi" + image_extension
    info_fig["ylabel"] = ["Excitatory firing rate (Hz)", "Inhibitory firing rate (Hz)"]
    params_plot["plots_color"] = [[(0, 0.7, 0)], [(1, 0.3, 0.3)]]

    gr.plot_one_graph(path, params_sim, info_cells, info_fig, params_fig, font_size, params_plot)

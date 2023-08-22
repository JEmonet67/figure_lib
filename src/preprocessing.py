import re
import os
import pandas as pd
import numpy as np
import coordinate_manager as cm
import data_transformation as dt

# TODO : Fonction pour sauvegarder le dictionnaire d'arrays une fois les prétraitements terminés, faire une fonction
# au départ de Modèles_graphes qui prend le nom du dictionnaire et le cherche pour l'importer ou le créé s'il
# n'existe pas

# TODO
def import_multiple_csv_to_array(path_csv, params_sim, outputs, celltypes, dict_re, xlim):
    """
    ### FUNCTION TO IMPORT MULTIPLE CSV FILE IN ONE PATH WITH THE SAME SIMULATION PARAMETERS ###

        -- Input --


        -- Output --
    LIST OF EACH FILE ARRAYS DICTIONNARIES.

    """


# TODO : Modifier pour prendre tous les output_cellules différentes dans le csv et tout extraire en arrays sans avoir à indiquer "info_cells"
# faire aussi en sorte que si muVn exc et muVn inh sont présents, le VSDI est calculé automatiquement.
# Créer un dictionnaire preproduction qui prends tous les noms de prétraitement à faire (centrage, dérivées, vsdi...).
# y mettre par exemple center=True.
def import_csv_to_array(path_csv, params_sim, dict_re, xlim):
    dfs = pd.read_csv(path_csv, chunksize=2000)  # Lecture csv en dataframe de 2000 lignes de csv

    i_df = 0  # Compteur de dataframe 2000 lignes de csv

    for df in dfs:  # Parcours des dataframe de 2000 lignes de csv
        df = df.set_index("Time")  # Renommage index
        print("Sorting...", end="")
        df = df.sort_index(axis=1, key=lambda x: [
            # Tri de l'index temporel en fonction du type cellulaire puis de l'output et du numéro
            (dict_re["output_num_celltype"].findall(elt)[0][2],
             dict_re["output_num_celltype"].findall(elt)[0][0],
             int(dict_re["output_num_celltype"].findall(elt)[0][1]))
            for elt in x.tolist()])
        print("Done !")

        n_t = df.shape[0]  # Sauvegarde de la taille temporelle du dataframe en cours

        if i_df==0: # Création dictionnaire d'array à partir de tous les outputs_typeCell uniques du dataframe
            names_col = df.columns.tolist()  # Sauvegarde des noms de colonnes du dataframe
            list_output_celltype = []
            list_num = []

            for col in names_col: # Parcours des noms de colonnes du dataframe
                (output, num, celltype) = dict_re["output_num_celltype"].findall(col)[0]  # Extraction output, numéro et type cellulaire depuis le nom de la colonne actuelle
                list_num += [int(num)]  # Transformation du numéro en int et ajout dans la liste de num.
                list_output_celltype += [f"{output}_{celltype}"] # Ajout str output_celltype dans sa liste.

            outputs_celltypes = list(set(list_output_celltype)) # Utilisation du set pour enlever les doublons de noms de colonnes
            print("outputs_celltypes", outputs_celltypes)
            dict_arr_outputs = {outputs_celltype: [] for outputs_celltype in
                                outputs_celltypes}  # Déclaration dictionnaire [output_celltype:list[arrays]]


        print("Initialize arrays...", end="")
        for output_celltype in outputs_celltypes:  # Parcours des pairs outputs/type cellule à transformer en arrays
            dict_arr_outputs[f"{output_celltype}"] += [np.zeros([params_sim["n_cells_X"], params_sim["n_cells_Y"],
                                                                 n_t])]  # Initialisation array de 0 dans chaque liste de pair output/type cellule
        print("Done !")

        print("Iterate index...")
        # TODO : Changer pour utiliser un parcours de dictionnaire {num : output_celltype} créé au tout début pour faire la liste des
        # outputs à faire. Cela permet d'éviter de parcourir deux fois pour rien les colonnes du df et le regexp.
        for i,num in enumerate(list_num):
            dict_coord_macular = cm.id_to_coordinates(num, (params_sim["n_cells_X"], params_sim[
                "n_cells_Y"]))  # Transformation id colonne en cours en coordonnées macular

            # Enregistrement de la colonne temporelle en cours à sa coordonnée correspondante
            dict_arr_outputs[list_output_celltype[i]][i_df][
                dict_coord_macular["X"], dict_coord_macular["Y"]] = df.iloc[:, i].to_numpy()
        print("Done !\n")
        i_df += 1
    print("DFs DONE")

    for key in dict_arr_outputs:  # Parcours des outputs type cellule
        dict_arr_outputs[key] = np.concatenate(dict_arr_outputs[key],
                                               axis=-1)  # Concaténation des listes d'array des dataframe de 2000 lignes
        dict_arr_outputs[key] = np.rot90(dict_arr_outputs[key]) # Conversion coordonnées numpy to macular
        dict_arr_outputs[key] = dict_arr_outputs[key][:, :, int(np.ceil(
            params_sim["n_transient_frame"] * params_sim["delta_t"] / params_sim["dt"])):]

    if "muVn_CorticalExcitatory" in list_output_celltype and "muVn_CorticalInhibitory" in list_output_celltype:
        print("Make VSDI")
        dict_arr_outputs["VSDI"] = dt.muVn_to_VSDI(dict_arr_outputs["muVn_CorticalExcitatory"],
                                                   dict_arr_outputs["muVn_CorticalInhibitory"])

    print("END")

    return dict_arr_outputs


def compile_regexp():
    """
    ### FUNCTION TO COMPILE REGEXP IN ORDER TO DON'T REPLICATE IT ###

        -- Input --


        -- Output --
    Compiled regexp for find all information in path and file.

    """
    reg_cond = re.compile(
        r"([\w ]*)([\d]{1,3},?\d{0,3}\w*)")  # RE to find all experimental conditions names based on repertory path
    reg_path = re.compile(
        r"/(\d{1,3})x(\d{1,3})c/(.*?)/(stim_(\d*,?\d*)x(\d*,?\d*)deg_(\d*)(.*))?")  # RE to find nb of cells,typeStim, stim size, stim speed and unit
    reg_file = re.compile(
        r"\w{1,5}_\w{1,10}_\w{1,10}\d{4}_([A-Za-z]*)(\d*,?\d{0,3})([A-Za-z]*)_(\w{1,5})f.csv")  # RE to find caract name, value and unit from file name following nomenclature
    reg_output_num_celltype = re.compile(
        r"(.*?) \(([0-9]{1,5})\) (.*)")  # Expression régulière output/numéro/type cellule dans le nom de la colonne d'un csv macular

    dict_re = {"path": reg_path, "cond":reg_cond, "file": reg_file, "output_num_celltype":reg_output_num_celltype}

    return dict_re

# TODO : Faire en sorte d'ajouter toutes les tailles de stim d'un stimulus renseigné dans le path si elles sont différentes
def path_analyzer(path, dict_re, test):
    """
    ### FUNCTION TO MAKE TITLE AND RECUPERATE FRAME TRANSIENT NUMBER, CELLS NUMBER BY ANALYZING PATH ###

        -- Input --
    path : Path of the the CSV file used for the graph. Have to follow the nomenclature of data :
    branchGitMacular}/{setGraph}_{setParams}/{X}x{Y}c/{colorBack}_background_{colorForm}_{formStim}/
    stim{xForm}x{yForm}°_{dps}°perSec/[specificConditionSim]/IDsim_{nameParam}{valueParam}{unitParam}_{n_frame_transient}f.csv

    dict_re : Dictionnary contening every regexp needed for graph ploting.

    test : Variable to pass from to test mode if test is True. By default test is False.

        -- Output --
    Return the graph title, the number of cells in X and Y axis, the number of frame of the transient and the name of the last simulation condition
    used to construct image name.

    """

    if test == 1:
        curr_path = f"{os.getcwd()}/tests/data_tests/GaussianCorrection_PvaluesAnticipation/optiParams/20x4c/white_bar/stim_0,9x0,66deg_6°perSec"
    elif test == 2:
        curr_path = f"{os.getcwd()}/tests/data_tests/GaussianCorrection_PvaluesAnticipation/semiBioGraph_extDriveParams/20x5c/white_bar/stim_0,9x0,66deg_6°perSec"
    else:
        curr_path = os.getcwd()

    # Define path and the path of simulation condition used
    file = path.split("/")[-1]
    path_condSim = path.replace(file, "").replace(curr_path, "")[1:-1]
    list_cond = path_condSim.replace("_", " ").split("/")[:-1]

    # Command specific to one path
    X, Y, typeStim, stim, x_stim, y_stim, speed, speed_unit = dict_re["path"].findall(curr_path)[0]
    typeStim = typeStim.replace("_", " ")
    X = int(X);
    Y = int(Y)
    str_cond = ", ".join([" ".join(dict_re["cond"].findall(list_cond[i])[0]) for i in range(len(list_cond))])

    # Command specific to one file
    name_caract, value_caract, unit_caract, n_transient_frame = re.findall(file)[0]
    n_transient_frame = int(n_transient_frame)
    str_lastcond = f"{name_caract} {value_caract}{unit_caract}"
    str_lastcond_name_file = f"{name_caract}{value_caract}{unit_caract}"
    if str_cond == "":
        str_cond += str_lastcond
    else:
        str_cond += ", " + str_lastcond

    # Make formated title
    if stim == '':  # Data with background stimuli
        title = f"Simulation {X}x{Y} with {typeStim}\n{str_cond}"

    else:
        if speed_unit == "ms":  # Data with flashed stimuli
            title = f"Simulation {X}x{Y} with {typeStim} of {x_stim}x{y_stim}° flashed {speed}ms\n{str_cond}"
        elif speed_unit == "°perSec":  # Data with motion stimuli
            title = f"Simulation {X}x{Y} with {typeStim} of {x_stim}x{y_stim}° moving at {speed}°/s\n{str_cond}"

    print(title)

    return title, X, Y, n_transient_frame, str_lastcond_name_file


def set_default_graph_params():
    """
    ### FUNCTION TO SET DEFAULT DICTIONNARIES CONTAINING PARAMETERS FOR GRAPH ###

        -- Input --


        -- Output --
    Return 4 dictionnaries.

    """
    info_fig = {"title" :"" ,"subtitles" :"", "image_name" :"",
                "xlabel" :"Time (s)" ,"ylabel" :"", "sharex" :True ,"sharey" :False}

    params_fig = dict(wspace=0.2 ,hspace=0.4 ,height_ratios=[1 ,1] ,width_ratios=[1 ,1])

    font_size = {"main_title" :35, "subtitle" :25, "xlabel" :35, "ylabel" :35, "g_xticklabel" :35, "g_yticklabel" :35,
                 "legend" :20}

    params_plot = {"grid_color" :"lightgray" ,"grid_width" :4 ,"ticklength" :7 ,"tickwidth" :3, "labelpad" :10, "Xlim" :(), "Ylim" :(),
                   "plots_color" :[(0 ,0 ,0)]}

    return info_fig, params_fig, font_size, params_plot

# TODO  : Ajouter "name_cell" dans l'ajout du dictionnaire info_cells
# TODO : Il faudra peut être créer des dictionnaires spécifiques à chaque type de
# graphe (heatmap, multigraphe...) avec leurs paramètres à eux.
def setup_dict_graph(base_p,dict_re):
    """
    ### FUNCTION TO MAKE AND MODIFY ALL THE DICTIONNARIES NEEDED FOR GRAPH GENERATION ###

        -- Input --
    base_p : Dictionnary contening base parameters depending from the file used for the graph.

        -- Output --
    Return 6 dictionnaries

    """
    # Analyze path
    title, X, Y, n_transient_frame, image_name_end = path_analyzer(base_p["path"], dict_re, test = base_p["test"])

    # Creation and modifications of the dictionnaries used for ploting
    info_fig, params_fig, font_size, params_plot = set_default_graph_params() # Default parameters for the graph
    info_fig["title"] = title; info_fig["ylabel"] = base_p["ylabel"]; info_fig["legend"] = base_p["legend"]; info_fig["postprod"] = base_p["postprod"]
    params_plot["plots_color"] = base_p["colors"]; params_plot["Xlim"] = base_p["Xlim"]; params_plot["Ylim"] = base_p["Ylim"]
    params_plot["center"] = base_p["center"]
    for dim in base_p["diminutive_output"]:
        info_fig["image_name"] += dim
        info_fig["image_name"] += "_"
    info_fig["image_name"] += image_name_end

    params_sim = {"dx" :base_p["dx"], "delta_t": base_p["delta_t"], "n_transient_frame": n_transient_frame,
                  "n_cells_X": X, "n_cells_Y": Y}
    info_cells = {"name_output": base_p["name_output"], "name_cell":base_p["name_cell"], "num": base_p["num"], "layer": base_p["layer"]}

    return params_sim, info_cells, info_fig, params_fig, font_size, params_plot
import pandas as pd
import numpy as np

# TODO : ADD FUNCTION TO CONVERT CSV IN ARRAYS DICTIONNARY


# TODO : get_center_coordinates() and macular_id_to_coord(num) and test this code
def select_position_to_plot(output, num, params_sim):
    if num == "": # Case 0 for taking all
        arrays_to_plot = []
        for i in output.shape[0]:
            for j in output.shape[1]:
                arrays_to_plot += [output[i,j]]

    if num == -1:  # Case 1 for taking default center position of the array
        coord = cm.get_center_coordinates()  # get coordinates of the array central position
        arrays_to_plot = output[coord[0], coord[1]]  # add central position to outputs

    elif type(num) == int and num >= 0:  # Case 2 to have one position thanks to the macular id
        coord = cm.macular_id_to_coord(num)
        arrays_to_plot = output[coord[0], coord[1]]  # add given position to outputs

    elif type(num) == tuple and len(num) == 2:  # Case 3 to take one position thanks to the coordinates
        arrays_to_plot = output[num[0], num[1]]  # add given position to outputs

    elif type(num) == dict:  # Case 4 for taking horizontal or vertical line of an array
        arrays_to_plot = []
        if num["axis"] == 0:  # Get horizontal line of positions
            if "step" in num:  # With step
                for i in range(num["start"],num["end"],num["step"]):
                    arrays_to_plot += [output[i, np.ceil(params_sim["n_cells_Y"]/2),:]]
            else:  # Without step
                for i in range(num["start"], num["end"]):
                    arrays_to_plot += [output[i, np.ceil(params_sim["n_cells_Y"]/2),:]]

        elif num["axis"] == 1:  # Get vertical line of positions
            if "step" in num:  # With step
                for i in range(num["start"],num["end"],num["step"]):
                    arrays_to_plot += [output[np.ceil(params_sim["n_cells_X"]/2), i,:]]
            else:  # Without step
                for i in range(num["start"], num["end"]):
                    arrays_to_plot += [output[np.ceil(params_sim["n_cells_X"]/2), i,:]]

    elif type(num) == list:  # Case 5 for taking a list of given macular id.
        arrays_to_plot = []
        for macular_id in num:
            coord = cm.macular_id_to_coord(macular_id)
            arrays_to_plot += [output[coord[0], coord[1]]]

    elif type(num) == list and type(num[0]) == tuple:  # Case 5 for taking a list of given cell coordinates.
        arrays_to_plot = []
        for coord in num:
            arrays_to_plot += [output[coord[0], coord[1]]]

    return arrays_to_plot

# TODO : ADAPT THIS FUNCTION
def muVn_to_VSDI(muVn_exc, muVn_inh):
    """
    FUNCTION TO COMPUTE VSDI ARRAY FROM EXCITATORY AND INHIBITORY MEAN VOLTAGE ARRAYS
    """
    muVn0_exc = muVn_exc[: ,:, 0]
    muVn0_exc = np.repeat(muVn0_exc[:, :, np.newaxis], muVn_exc.shape[-1], axis=2)
    muVn0_inh = muVn_inh[: ,:, 0]
    muVn0_inh = np.repeat(muVn0_inh[:, :, np.newaxis], muVn_inh.shape[-1], axis=2)
    vsdi_exc = (-(muVn_exc - muVn0_exc) / muVn_exc[: ,:])
    vsdi_inh = (-(muVn_inh - muVn0_inh) / muVn_inh[: ,:])

    vsdi = vsdi_exc *0.8 + vsdi_inh *0.2

    return vsdi


# Méthode pour calculer la dérivée d'un gdf.
def compute_derivate(df):
    df_dXdt = pd.DataFrame(np.NaN, index=df.data.index, columns=df.data.columns)
    for i_cell in range(len(df.data.columns)):
        for i_time in range(len(df.data.index)):
            if i_time == 0:
                t_moins1 = df.data.iloc[[i_time],[i_cell]]
                t_plus1 = df.data.iloc[[i_time+1],[i_cell]]
                df_dXdt.iloc[[i_time],[i_cell]] = (t_plus1.iloc[0,0] - t_moins1.iloc[0,0]) / (t_plus1.index[0] - t_moins1.index[0])

            elif i_time == len(df.data.index)-1:
                t_moins1 = df.data.iloc[[i_time-1],[i_cell]]
                t_plus1 = df.data.iloc[[i_time],[i_cell]]
                df_dXdt.iloc[[i_time],[i_cell]] = (t_plus1.iloc[0,0] - t_moins1.iloc[0,0]) / (t_plus1.index[0] - t_moins1.index[0])

            else:
                t_moins1 = df.data.iloc[[i_time-1],[i_cell]]
                t_plus1 = df.data.iloc[[i_time+1],[i_cell]]
                df_dXdt.iloc[[i_time],[i_cell]] = (t_plus1.iloc[0,0] - t_moins1.iloc[0,0]) / (t_plus1.index[0] - t_moins1.index[0])

    return df_dXdt

# Méthode pour calculer la dérivée d'un dataframe.
def compute_derivate_df(df):
    df_dXdt = pd.DataFrame(np.NaN, index=df.index, columns=df.columns)
    for i_cell in range(len(df.columns)):
        for i_time in range(len(df.index)):
            if i_time == 0:
                t_moins1 = df.iloc[[i_time],[i_cell]]
                t_plus1 = df.iloc[[i_time+1],[i_cell]]
                if t_plus1.index[0] - t_moins1.index[0] != 0:
                    df_dXdt.iloc[[i_time],[i_cell]] = (t_plus1.iloc[0,0] - t_moins1.iloc[0,0]) / (t_plus1.index[0] - t_moins1.index[0])
                else:
                    df_dXdt.iloc[[i_time],[i_cell]] = 0

            elif i_time == len(df.index)-1:
                t_moins1 = df.iloc[[i_time-1],[i_cell]]
                t_plus1 = df.iloc[[i_time],[i_cell]]
                if t_plus1.index[0] - t_moins1.index[0] != 0:
                    df_dXdt.iloc[[i_time],[i_cell]] = (t_plus1.iloc[0,0] - t_moins1.iloc[0,0]) / (t_plus1.index[0] - t_moins1.index[0])
                else:
                    df_dXdt.iloc[[i_time],[i_cell]] = 0
            else:
                t_moins1 = df.iloc[[i_time-1],[i_cell]]
                t_plus1 = df.iloc[[i_time+1],[i_cell]]
                if t_plus1.index[0] - t_moins1.index[0] != 0:
                    df_dXdt.iloc[[i_time],[i_cell]] = (t_plus1.iloc[0,0] - t_moins1.iloc[0,0]) / (t_plus1.index[0] - t_moins1.index[0])
                else:
                    df_dXdt.iloc[[i_time],[i_cell]] = 0

    return df_dXdt

def delay_to_duration(list_df_latence, list_df_sttp, params_sim):
    """
    FUNCTION TO CONVERT LIST OF LATENCY DATAFRAMES INTO DURATION DATAFRAMES
    """

    # Variables declaration for new list
    new_list_df_latence = []
    new_list_df_sttp = []


    # Loop on graph index in latency and sttp lists
    for i_graph in range(len(list_df_latence)):
        # Copy each dataframes in a new list
        new_list_df_latence += [list_df_latence[i_graph].copy()]
        new_list_df_sttp += [list_df_sttp[i_graph].copy()]

        duration_latency = []
        duration_sttp = []

        # Loop on index for row in latency and sttp dataframes
        for j in range(new_list_df_latence[i_graph].shape[0]):
            duration_sttp += [new_list_df_sttp[i_graph].iloc[j,0]/(params_sim["speed"]/1000) + new_list_df_sttp[i_graph].index[j]]
            duration_latency += [new_list_df_latence[i_graph].iloc[j,0]/(params_sim["speed"]/1000) + new_list_df_latence[i_graph].index[j]]

        new_list_df_latence[i_graph]["duration_latency"] = duration_latency
        new_list_df_sttp[i_graph]["duration_latency"] = duration_sttp

        new_list_df_latence[i_graph] = new_list_df_latence[i_graph].set_index("duration_latency")
        new_list_df_sttp[i_graph] = new_list_df_sttp[i_graph].set_index("duration_latency")



    return new_list_df_latence, new_list_df_sttp

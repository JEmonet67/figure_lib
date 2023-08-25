import pandas as pd
import numpy as np

# TODO : ADD FUNCTION TO CONVERT CSV IN ARRAYS DICTIONNARY


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




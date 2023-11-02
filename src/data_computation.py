import pandas as pd
import pwlf
import numpy as np
import data_transformation as dt
from os.path import isfile, join
import os
import pickle

import data_transform.GraphDF as gdf
import coordinate_manager as cm

def compute_peak_latency_time_fit(list_df_latency_delays, list_df_peak_delays, list_caract, params_sim):
    # Conversion delay to time latency and peak
    list_df_latency_time, list_df_peak_time = dt.delay_to_time(list_df_latency_delays, list_df_peak_delays,
                                                               params_sim, list_caract)

    # Define output dataframe and dicts
    df_fit = pd.DataFrame(index=[list_caract[1][i][0] for i in range(len(list_caract[1]))])
    list_pred = []

    # Loop on data
    for i in range(len(list_df_latency_time)):
        # Compute latency and peak fit
        dict_fit_latency_time = latency_fit(list_df_latency_time[i])
        dict_fit_peak_time = peak_fit(list_df_peak_time[i])

        # Compute stationary end latency delay
        end_latency_delays = list_df_latency_delays[i][list_df_latency_delays[i][0] >
                                                       dict_fit_latency_time["inflexion point"]]
        s_end_latency_delay = end_latency_delays.index.to_series().mean()

        # Compute stationary peak delay
        s_peak_delay = list_df_peak_delays[i].index.to_series().mean()

        # Fill fit dataframe.
        df_fit.loc[[f"{list_caract[1][i][0]}"], ["Inflexion point"]] = dict_fit_latency_time["inflexion point"]
        df_fit.loc[[f"{list_caract[1][i][0]}"], ["Start latency speed"]] = dict_fit_latency_time["start speed"]
        df_fit.loc[[f"{list_caract[1][i][0]}"], ["End latency speed"]] = dict_fit_latency_time["end speed"]
        df_fit.loc[[f"{list_caract[1][i][0]}"], ["Peak speed"]] = dict_fit_peak_time["peak speed"]
        df_fit.loc[[f"{list_caract[1][i][0]}"], ["Stationary end latency delay"]] = s_end_latency_delay
        df_fit.loc[[f"{list_caract[1][i][0]}"], ["Stationary peak delay"]] = s_peak_delay

        # Fill dictionaries ouptuts
        list_pred += [{
            "x pred latency": dict_fit_latency_time["x pred"],
            "y pred latency": dict_fit_latency_time["y pred"],
            "x pred peak": dict_fit_peak_time["x pred"],
            "y pred peak": dict_fit_peak_time["y pred"],
        }]

    return df_fit, list_pred


def latency_fit(df_latency_time):
    # Output dictionary declaration
    dict_fit = {}

    # Define x and y for each fit
    x_latency_time = df_latency_time.index / 1000  # s
    y_cort_ext = df_latency_time[0].values  # °

    # Piecewise linear fit of latency time
    pwlf_latency_time = pwlf.PiecewiseLinFit(x_latency_time, y_cort_ext)
    breaks = pwlf_latency_time.fit(2)
    slopes = pwlf_latency_time.calc_slopes()

    # Prediction for latency time
    x_pred_latency_time = np.linspace(x_latency_time.min(), x_latency_time.max(), 100)
    y_pred_latency_time = pwlf_latency_time.predict(x_pred_latency_time)

    # Compute inflexion point
    i_inflex = np.where(x_pred_latency_time > breaks[1])[0][0]

    # Set outputs variables
    dict_fit["inflexion point"] = y_pred_latency_time[i_inflex]
    dict_fit["start speed"] = slopes[0]
    dict_fit["end speed"] = slopes[1]
    dict_fit["x pred"] = x_pred_latency_time*1000  # Second to ms
    dict_fit["y pred"] = y_pred_latency_time

    return dict_fit


def peak_fit(df_peak_time):
    # Output dictionary declaration
    dict_fit = {}

    # Define x and y for each fit
    x_peak_time = df_peak_time.index / 1000  # s
    y_cort_ext = df_peak_time[0].values  # °

    # Piecewise linear fit of latency time
    pwlf_peak_time = pwlf.PiecewiseLinFit(x_peak_time, y_cort_ext)
    pwlf_peak_time.fit(1)
    slopes = pwlf_peak_time.calc_slopes()

    # Prediction for latency time
    x_pred_peak_time = np.linspace(x_peak_time.min(), x_peak_time.max(), 100)
    y_pred_peak_time = pwlf_peak_time.predict(x_pred_peak_time)

    # Set outputs variables
    dict_fit["peak speed"] = slopes[0]
    dict_fit["x pred"] = x_pred_peak_time*1000  # Second to ms
    dict_fit["y pred"] = y_pred_peak_time

    return dict_fit

# TODO : Faire une RE qui donne les conditions sous forme nameParamValueUnits puis utiliser un autre re pour séparer les
# trois variables et les placer dans une liste afin de permettre d'avoir autant de conditions de notre choix et d'en
# avoir parfois qu'une seule au lieu d'être forcé d'en à avoir forcément 2 comme actuellement.
def compute_peak_latency_delays(path, params_sim, params_plot, dict_re):
    files = [f for f in os.listdir(path) if isfile(join(path,f)) and dict_re["file"].findall(f) != []]
    print("Files sorting...", end="")
    files.sort(key = lambda x : float(dict_re["file"].findall(x)[0][1].replace(",",".")))
    print("Done !")
    print(files)
    list_df_latency = []
    list_df_ttp = []
    list_list_value_caract = []

    first_cell = 5
    last_cell = params_sim["n_cells_X"]-6

    print("Files browsing.")
    i = 0
    for file in files:
        info_caract = dict_re["file"].findall(file)[0]
        list_name_caract = list(info_caract[0:-1:3])
        list_value_caract = list(info_caract[1:-1:3])
        list_unit_caract = list(info_caract[2:-1:3])

        for i_name, name_value in enumerate(list_name_caract):
            if name_value == "barSpeed":
                speed = float(list_value_caract[i_name])
                break
            elif name_value != "barSpeed":
                speed = params_sim["speed"]

        n_transient_frame = info_caract[-1]

        list_list_value_caract.append(list_value_caract)
        n_transient_frame = int(n_transient_frame)
        print("Create GraphDF...", end="")
        df = gdf.GraphDF(f"{path}/{file}",params_sim["delta_t"],60,params_sim["n_cells_X"],params_sim["n_cells_Y"])
        print("Done!")

        print("Crop GraphDF...", end="")
        df = df.crop(params_sim["delta_t"]*n_transient_frame)
        print("Done!")

        # Calcul VSDI
        print("Isolate cortical column outputs...", end="")
        df_muV = df.isolate_dataframe_byoutputs("muVn")
        num_exc = cm.get_interval_macular_cell((params_sim["n_cells_X"], params_sim["n_cells_Y"]), 3, first_cell, last_cell)
        num_inh = cm.get_interval_macular_cell((params_sim["n_cells_X"], params_sim["n_cells_Y"]), 4, first_cell, last_cell)
        print("num...", end="")
        exc = df_muV.isolate_dataframe_columns_bynum(num_exc)
        inh = df_muV.isolate_dataframe_columns_bynum(num_inh)
        print("Done!")
        print("Compute VSDI...",end="")
        exc.data = (-(exc.data - exc.data.iloc[0]) / exc.data.iloc[0])
        inh.data = (-(inh.data - inh.data.iloc[0]) / inh.data.iloc[0])

        col_exc_rename = {exc.data.columns[i]:f"CorticalColumn ({i}) vsdi" for i in range(exc.data.columns.shape[0])}
        col_inh_rename = {inh.data.columns[i]:f"CorticalColumn ({i}) vsdi" for i in range(exc.data.columns.shape[0])}
        exc.data = exc.data.rename(columns=col_exc_rename)
        inh.data = inh.data.rename(columns=col_inh_rename)

        vsdi = exc.copy()
        vsdi.data=exc.data*0.8+inh.data*0.2
        print("Done!")

        # Calcul temps centre barre milieu champ récepteur (t0)
        print("Compute t0 bar center on middle RF...", end="")
        list_barcenter = [np.round(((np.round(x_col * params_sim["dx"], 2) + params_sim["size_bar"] / 2) / speed - params_sim["delta_t"]), 2)
                          for x_col in range(first_cell, last_cell + 1)]
        print("Done !")


        # Calcul liste des temps des pics de chaque cellule (t1)
        print("Compute VSDI peaks...", end="")
        list_max_vsdi = [col.tmax for col in vsdi.list_col]
        print("Done!")

        # Calcul liste de temps de début d'activation de chaque colonnes corticales (t2)
        print("Activation time...", end="")
        # TODO Use compute_derivate as a generator to loop only untill find a index validating the inflexion condition
        dVSDIdt = dt.compute_derivate(vsdi)
        vsdi_f = vsdi.data[(dVSDIdt>0.01) & (vsdi.data > 0.001)] # vsdi.data > 0.001 dVSDIdt>0.01
        list_inflex_vsdi = [round(vsdi_f.iloc[:,i].dropna().index[0],3) for i in range(len(vsdi_f.columns))]
        print("Done!")

        # Calcul de la STTP
        print("Compute TTP and latency delays...", end="")
        list_ttp = [(list_max_vsdi[i]-list_barcenter[i])*1000 for i in range(len(list_max_vsdi))]
        #list_ttp = list_ttp[6:params_sim["n_cells_X"]-4]
        list_latency = [(list_inflex_vsdi[i]-list_barcenter[i])*1000 for i in range(len(list_inflex_vsdi))]
        #list_latency = list_latency[6:params_sim["n_cells_X"]-4]

        print("Done!")

        print("Make TTP and latency delays dataframes")
        df_ttp = pd.DataFrame([round(i*params_sim["dx"],2) for i in range(first_cell, last_cell + 1)], index=list_ttp)
        df_latency = pd.DataFrame([round(i*params_sim["dx"],2) for i in range(first_cell, last_cell + 1)], index=list_latency)
        list_df_latency.append(df_latency)
        list_df_ttp.append(df_ttp)

        i+=1

    list_caract = [list_name_caract, list_list_value_caract, list_unit_caract]
    dict_latency_ttp_caract = {"caract":list_caract,
                                "latency":list_df_latency,
                                "sttp":list_df_ttp}
    with open(path+f"dict_TTP_latency_{list_caract[0][0]}_newVSDI_rfCenter_new", "wb") as file:  # Pickling
        pickle.dump(dict_latency_ttp_caract, file)

    return list_df_latency, list_df_ttp, list_caract
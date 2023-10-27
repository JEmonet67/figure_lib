import pandas as pd
import pwlf
import numpy as np
import data_transformation as dt


def peak_latency_compute(list_df_latency_delays, list_df_peak_delays, list_caract, params_sim):
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

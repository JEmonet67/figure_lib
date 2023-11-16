import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import postproduction as post
import data_transformation as dt
import make_figure.graphFigure as gfg
import data_transform.GraphDF as gdf
import coordinate_manager as cm
import stim_help_functions as hf



# TODO : Nettoyer plot_one_graph pour n'avoir qu'une fonction qui prend un dictionnaire
# d'arrays 1D en entrée et crée un graphe à partir de ça.
# Retirer toute la partie "pre-process" qui se trouve dans la fonction pour l'enlever
# si pas nécessaire ou transférer dans une autre fonction dans le cas contraire comme
# celle pour sélectionner l'array 1D d'une cellule à partir de son numéro d'ID.

# TODO
def fig_saving():
    """
    ### FUNCTION TO SAVE ONE FIG IN A PNG FILE ###

        -- Input --


        -- Output --
    PNG File.

    """

# TODO
def plot_one_output(ax, output, index):
    """
    ### FUNCTION TO PLOT ONE CELL OUTPUT IN FUNCTION OF A GIVEN INDEX ###

        -- Input --


        -- Output --
    Make one curve in the plot.

    """
def make_graph(ax, dict_arrays, dict_index, params_sim, info_cells, info_fig, params_fig, font_size, params_plot, save=True):
    """
    ### FUNCTION TO PLOT ALL WANTED CELL OUTPUT IN FUNCTION OF A GIVEN INDEX ###

        -- Input --


        -- Output --
    Construct one graph by ploting cell output set in info_cells and with a given index.

    """
    dict_outputs = {} # outputs dict to plot in the graph
    for output_typeCell in dict_arrays: # Iterate on output_typeCell in dict_arrays
        num = info_cells[output_typeCell]["num"] # Pick id of the current output_typeCell
        output = dict_arrays[output_typeCell]
        index = dict_index[output_typeCell]

        dict_outputs[output_typeCell] = dt.select_position_to_plot(output, num, params_sim) # transform output array by selecting position to plot
        # we can have list of arrays or arrays in dict_outputs[output_typeCell] values

        # TODO Make plot with the filtred list arrays
        # TODO Make legend
        # TODO Set graph post production treatment


def plot_one_graph(path, df, params_sim, info_cells, info_fig, params_fig, font_size, params_plot):
    """
    ### FUNCTION TO PLOT CELL OUTPUT IN FUNCTION OF TIME ###

        -- Input --
    title : Name of the title to set in the graph.
    num : Macular ID number of the cell to display. By default, the function will chose the middle cell.

        -- Output --
    Make a figure with of one graph.

    """
    list_outputs = []
    print("Compute macular cell num")
    # Macular cell numero computation and legend if needed
    for i in range(len(info_cells["num"])):
        # VSDI graphs
        if info_cells["name_output"][i] == "VSDI":
            # Multiple curve graphs
            if type(info_cells["num"][i][0] )==list:
                if len(info_cells["num"][i][0]) == 2:
                    info_cells["num"][i][0] = cm.get_interval_macular_cell \
                        ((params_sim["n_cells_X"] ,params_sim["n_cells_Y"]) ,info_cells["layer"][i][0]
                        ,info_cells["num"][i][0][0] ,info_cells["num"][i][0][1], axis=params_sim["axis"])
                elif len(info_cells["num"][i][0]) == 3:
                    info_cells["num"][i][0] = cm.get_interval_macular_cell \
                        ((params_sim["n_cells_X"] ,params_sim["n_cells_Y"]) ,info_cells["layer"][i][0]
                        ,info_cells["num"][i][0][0] ,info_cells["num"][i][0][1] ,info_cells["num"][i][0][2], axis=params_sim["axis"])

            if type(info_cells["num"][i][1] )==list:
                if len(info_cells["num"][i][1]) == 2:
                    info_cells["num"][i][1] = cm.get_interval_macular_cell \
                        ((params_sim["n_cells_X"] ,params_sim["n_cells_Y"]) ,info_cells["layer"][i][1]
                        ,info_cells["num"][i][1][0] ,info_cells["num"][i][1][1], axis=params_sim["axis"])
                elif len(info_cells["num"][i][1]) == 3:
                    info_cells["num"][i][1] = cm.get_interval_macular_cell \
                        ((params_sim["n_cells_X"] ,params_sim["n_cells_Y"]) ,info_cells["layer"][i][1]
                        ,info_cells["num"][i][1][0] ,info_cells["num"][i][1][1] ,info_cells["num"][i][1][2], axis=params_sim["axis"])
            # One curve graphs
            if info_cells["num"][i][0]==-1:
                info_cells["num"][i][0] = params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i][0] + \
                                        params_sim["n_cells_Y"] * (int(np.ceil(params_sim["n_cells_X"] / 2)) - 1) + \
                                        int(np.floor(params_sim["n_cells_Y"] / 2))
            if info_cells["num"][i][1]==-1:
                info_cells["num"][i][1] = params_sim["n_cells_X"]*params_sim["n_cells_Y"]*info_cells["layer"][i][1] + \
                                       params_sim["n_cells_Y"]*(int(np.ceil(params_sim["n_cells_X"]/2))-1)+\
                                       int(np.floor(params_sim["n_cells_Y"]/2))


            print("Exc and Inh filtering.")
            exc = df.isolate_dataframe_columns_bynum(f'{info_cells["num"][i][0]}')
            exc = exc.isolate_dataframe_byoutputs("muVn")
            inh = df.isolate_dataframe_columns_bynum(f'{info_cells["num"][i][1]}')
            inh = inh.isolate_dataframe_byoutputs("muVn")

            print("Compute VSDI")
            #exc.data = (-(exc.data - exc.data.iloc[0].mean()) / exc.data.iloc[0].mean())
            #inh.data = (-(inh.data - inh.data.iloc[0].mean()) / inh.data.iloc[0].mean())
            exc.data = (-(exc.data - exc.data.iloc[0]) / exc.data.iloc[0])
            inh.data = (-(inh.data - inh.data.iloc[0]) / inh.data.iloc[0])

            col_exc_rename = {exc.data.columns[i] :f"CorticalColumn ({i}) vsdi" for i in range(exc.data.columns.shape[0])}
            col_inh_rename = {inh.data.columns[i] :f"CorticalColumn ({i}) vsdi" for i in range(inh.data.columns.shape[0])}
            exc.data = exc.data.rename(columns=col_exc_rename)
            inh.data = inh.data.rename(columns=col_inh_rename)

            vsdi = exc.copy()
            vsdi.data = exc.data *0.8 +inh.data *0.2
            if params_plot["center"]:
                print("Centering VSDI")
                if params_sim["axis"]: # Vertical axis
                    list_pos_col = [np.round(np.round((int(num) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i][0]) /
                                params_sim["n_cells_Y"]%1*41,0)* params_sim["dx"],2)
                               for num in info_cells["num"][i][0].split(",")]
                    list_pos_col.reverse()
                    vsdi = vsdi.rf_centering_df(list_pos_col, params_sim["speed"], params_sim["size_bar"], params_sim["delta_t"])
                    vsdi.reverse()

                else: # Horizontal axis
                    list_pos_col = [np.round(np.floor(
                        (int(num) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i][0]) /
                        params_sim["n_cells_Y"]) * params_sim["dx"], 2)
                               for num in info_cells["num"][i][0].split(",")]
                    vsdi = vsdi.rf_centering_df(list_pos_col, params_sim["speed"], params_sim["size_bar"], params_sim["delta_t"])
            list_outputs += [vsdi]

        # Classical macular outputs graphs
        else:

            # Multiple curve graphs
            if type(info_cells["num"][i])==list:
                if len(info_cells["num"][i]) == 2:
                    info_cells["num"][i] = cm.get_interval_macular_cell \
                        ((params_sim["n_cells_X"] ,params_sim["n_cells_Y"]) ,info_cells["layer"][i]
                        ,info_cells["num"][i][0] ,info_cells["num"][i][1], axis=params_sim["axis"])
                elif len(info_cells["num"][i]) == 3:
                    info_cells["num"][i] = cm.get_interval_macular_cell \
                        ((params_sim["n_cells_X"] ,params_sim["n_cells_Y"]) ,info_cells["layer"][i]
                        ,info_cells["num"][i][0] ,info_cells["num"][i][1] ,info_cells["num"][i][2], axis=params_sim["axis"])

            # One curve graphs
            if info_cells["num"][i]==-1:
                info_cells["num"][i] = params_sim["n_cells_X"]*params_sim["n_cells_Y"]*info_cells["layer"][i] + \
                                       params_sim["n_cells_Y"]*(int(np.ceil(params_sim["n_cells_X"]/2))-1)+\
                                       int(np.floor(params_sim["n_cells_Y"]/2))

            output = df.isolate_dataframe_columns_bynum(f'{info_cells["num"][i]}')
            output = output.isolate_dataframe_byoutputs(info_cells["name_output"][i])

            if params_plot["center"]:
                print("Centering Data")
                if params_sim["axis"]: # Vertical axis
                    list_pos_col = [np.round(np.round((int(num) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i]) /
                                params_sim["n_cells_Y"]%1*41,0)* params_sim["dx"],2)
                               for num in info_cells["num"][i].split(",")]
                    list_pos_col.reverse()
                    output = output.rf_centering_df(list_pos_col, params_sim["speed"], params_sim["size_bar"],
                                                    params_sim["delta_t"])
                    output.reverse()
                else:
                    list_pos_col = [round(np.floor(
                        (int(num) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i]) /
                        params_sim["n_cells_Y"]) * params_sim["dx"], 2)
                        for num in info_cells["num"][i].split(",")]
                    output = output.rf_centering_df(list_pos_col, params_sim["speed"], params_sim["size_bar"],
                                                    params_sim["delta_t"])
                #output =output.tmax_centering_df()

            list_outputs += [output]


        print("Set legend")
        # Legend name generation for coordinates in degree selected
        if info_fig["legend"][i]=="coord_degree":
            if info_cells["name_output"][i] == "VSDI":
                if type(info_cells["num"][i]) == list:
                    if params_sim["axis"]: # Vertical axis
                        info_fig["legend"][i] = [f'''{np.round(np.round((int(num) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i][0]) / params_sim["n_cells_Y"]%1*41,0)* params_sim["dx"],2)}°'''
                                   for num in info_cells["num"][i][0].split(",")]
                        #info_fig["legend"][i].reverse()
                    else: # Horizontal axis
                        info_fig["legend"][i] = \
                            [f'{round(np.floor((int(num) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i][0]) / params_sim["n_cells_Y"]) * params_sim["dx"], 2)}°'
                            for num in str(info_cells["num"][i][0]).split(",")]

                elif type(info_cells["num"][i]) == int:
                    info_fig["legend"][i] = f'{round(np.floor((int(info_cells["num"][i]) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i]) / params_sim["n_cells_Y"]) * params_sim["dx"], 2)}°'

            else:
                if type(info_cells["num"][i]) == str:
                    if params_sim["axis"]: # Vertical axis
                        info_fig["legend"][i] = [f'''{np.round(np.round(
                            (int(num) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i]) /
                            params_sim["n_cells_Y"] % 1 * 41, 0) * params_sim["dx"], 2)}°'''
                                   for num in info_cells["num"][i].split(",")]

                        #info_fig["legend"][i].reverse()
                    else: # Horizontal axis
                        info_fig["legend"][i] = [
                            f'{round(np.floor((int(num) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i]) / params_sim["n_cells_Y"]) * params_sim["dx"], 2)}°'
                            for num in str(info_cells["num"][i]).split(",")]


                elif type(info_cells["num"][i]) == int:
                    info_fig["legend"][i] = [f'{round(np.floor((int(info_cells["num"][i]) - params_sim["n_cells_X"] * params_sim["n_cells_Y"] * info_cells["layer"][i]) / params_sim["n_cells_Y"]) * params_sim["dx"], 2)}°']

        # elif info_fig["legend"][i] = []: # Add specific legend

    f = gfg.graphFigure(list_outputs, len(list_outputs), 1, 20, 20, dict_info_fig=info_fig,
                                dict_font_size=font_size, dict_params_plot=params_plot, dict_params_fig=params_fig)
    # f,ax = plt.subplots(1,1, figsize = (20,20))
    # ax.plot(list_outputs[0].data)

    #plt.tight_layout(pad=3)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.10, top=0.89)

    # Set graph post production treatment
    print("Plot")
    # For multiple graph figures
    if type(f.ax) == np.ndarray:
        for i in range(len(f.ax)):
            # Set X and Y egde values
            if params_plot["Xlim"][i] != ():
                f.ax[i].set_xlim(params_plot["Xlim"][i][0], params_plot["Xlim"][i][1])
            if params_plot["Ylim"][i] != ():
                f.ax[i].set_ylim(params_plot["Ylim"][i][0], params_plot["Ylim"][i][1])

            # Set legend
            f.ax[i].legend(info_fig["legend"][i], fontsize=font_size['legend'])

            # Browse selected post-prod treatment
            for name, value in info_fig["postprod"].items():
                # Set highlight x interval
                if name == "highlight_interv":
                    for interval in value[i]:
                        post.highlight_x_interval(f.ax[i], interval[0], interval[1], interval[2], interval[3], interval[4])


    # For one graph figures
    else:
        # Set X and Y egde values
        if params_plot["Xlim"][i] != ():
            f.ax.set_xlim(params_plot["Xlim"][0][0], params_plot["Xlim"][0][1])
        if params_plot["Ylim"][i] != ():
            f.ax.set_ylim(params_plot["Ylim"][0][0], params_plot["Ylim"][0][1])

        # Set legend
        if type(info_fig["legend"][i]) == list:
            f.ax.legend(info_fig["legend"][0], fontsize=font_size['legend'])

        # Browse selected post-prod treatment
        for name, value in info_fig["postprod"].items():

            # Set highlight x interval
            if name == "highlight_interv":
                for interval in value[0]:
                    post.highlight_x_interval(f.ax, interval[0], interval[1], interval[2], interval[3], interval[4])

    plt.savefig(f'{"/".join(path.split("/")[:-1])}/{info_fig["image_name"]}.png')

# TODO : Mettre un array de t au lieu de multiplier l'indice à dt
# ajouter un paramètre bin pour binariser comme je le souhaite
# TODO : Corriger les fuites mémoires qui font tout planter quand je lance en série
def heatmap_video_function(name_function, function, path_video, dt, n_cells, legend, fps, bin_value, color="Greys_r"):
    # path_stat = stat_list_to_path(list_stats)
    # title_stat = stat_list_to_title(list_stats)

    c = 1
    #for name_function, function in dict_functions.items():
    #print(f"Function : {name_function} {c}/{len(dict_functions.items())}")
    print(f"Function : {name_function}")
    function = function[:, :, ::bin_value]
    max_value = function[:, :, :].max()
    min_value = function[:, :, :].min()

    if color == "RdBu_r": # Make symmetrical heatmap bar on 0
        if max_value > abs(min_value):
            min_value = -max_value
        else:
            max_value = -min_value

    list_frames = []
    for i in range(function.shape[2]):
        if i in range(0, function.shape[2], int(function.shape[2] / 10)):
            print(f"Progress : {np.round(i / function.shape[2] * 100, 0)}%")
        fig, ax_plot = plt.subplots(1, 1, figsize=(n_cells[0], n_cells[1]))
        params_fig = dict(wspace=0.15, hspace=0.4)

        info_fig = {"title": f"Heatmap {name_function}\nt={round(bin_value * i * dt, 4)}s", "subtitles": "",
                    "xlabel": "X coordinate", "ylabel": "Y coordinate", "colorbar_label": legend,
                    "sharex": True, "sharey": False}

        font_size = {"main_title": 35, "subtitle": 25, "xlabel": 25, "ylabel": 25, "g_xticklabel": 15,
                     "g_yticklabel": 15, "legend": 15, "global": 25}

        params_plot = {"grid_color": "lightgray", "grid_width": 4, "ticklength": 5, "tickwidth": 3, "labelpad": 15,
                       "col_map": color}

        mpl.rcParams.update({"font.size": font_size["global"]})
        sns.dark_palette("#69d", reverse=True, as_cmap=True)

        plot = sns.heatmap(function[:, :, i], vmin=min_value, vmax=max_value,
                           cbar_kws={'label': info_fig["colorbar_label"]}, cmap=params_plot["col_map"], ax=ax_plot)
        ax_plot.set_facecolor('black')
        [ax_plot.spines[side].set_visible(True) for side in ax_plot.spines]
        [ax_plot.spines[side].set_linewidth(2) for side in ax_plot.spines]
        ax_plot.tick_params(axis="x", which="both", labelsize=font_size["xlabel"], color="black",
                            length=params_plot["ticklength"], width=params_plot["tickwidth"])
        ax_plot.tick_params(axis="y", which="both", labelsize=font_size["ylabel"], color="black",
                            length=params_plot["ticklength"], width=params_plot["tickwidth"])
        ax_plot.set_xlabel("X coordinate", fontsize=25)
        ax_plot.set_ylabel("Y coordinate", fontsize=25)
        ax_plot.set_title(info_fig["title"], fontsize=30, pad=25)

        ax_plot.set_xticks(np.array([x for x in range(0, n_cells[0] - 1, 2)]))
        ax_plot.set_xticklabels([str(round(x * 0.225,2)) + "°" for x in range(0, n_cells[0] - 1, 2)])
        ax_plot.set_yticks(np.array([y for y in range(n_cells[1] - 2, -1, -2)]))
        ax_plot.set_yticklabels([str(round(x * 0.225,2)) + "°" for x in range(0, n_cells[1] - 1, 2)])

        # plt.savefig(f"{path}/frames/t{i}.png")

        canvas = FigureCanvas(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        X = np.frombuffer(s, np.uint8).reshape((height, width, 4))

        list_frames.append(X[:, :, 2::-1])
        plt.close()

        c += 1
    # TODO : Changer pour mettre moviepy
    hf.images_to_video_cv2(f"{name_function}_newVSDI.mp4", list_frames, c=True, path_video=path_video, fps=fps)

# TODO
#def heatmap_video_activity_region_function(name_function, function, path_video, dt, n_cells, legend, fps, bin_value):


def heatmap_picture_function(list_frame_to_select, dict_functions, path_output, dt, n_cells):
    bin_value = 14

    for name_function, function in dict_functions.items():
        function = function[:, :, ::]
        max_value = function[:, :, :].max()
        min_value = function[:, :, :].min()
        min_value = -0.05
        max_value = 0.05
        print(list_frame_to_select)
        for i_frame in list_frame_to_select:
            print(i_frame)
            fig, ax_plot = plt.subplots(1, 1, figsize=(15, 10))
            params_fig = dict(wspace=0.15, hspace=0.4)

            info_fig = {"title": f"{int(round(i_frame * dt, 4) * 1000)}ms", "subtitles": "",
                        "xlabel": "", "ylabel": "", "colorbar_label": "VSDI 60Hz - VSDI 1440Hz",
                        "sharex": True, "sharey": False}

            font_size = {"main_title": 35, "subtitle": 25, "xlabel": 25, "ylabel": 25, "g_xticklabel": 15,
                         "g_yticklabel": 15, "legend": 15, "global": 25}

            params_plot = {"grid_color": "lightgray", "grid_width": 4, "ticklength": 5, "tickwidth": 3, "labelpad": 15,
                           "col_map": "RdBu_r"}

            mpl.rcParams.update({"font.size": font_size["global"]})
            sns.dark_palette("#69d", reverse=True, as_cmap=True)

            plot = sns.heatmap(function[:, :, i_frame], vmin=min_value, vmax=max_value,
                               cbar_kws={'label': info_fig["colorbar_label"]}, cmap=params_plot["col_map"], ax=ax_plot)
            ax_plot.set_facecolor('black')
            [ax_plot.spines[side].set_visible(True) for side in ax_plot.spines]
            [ax_plot.spines[side].set_linewidth(2) for side in ax_plot.spines]
            ax_plot.tick_params(axis="x", which="both", labelsize=font_size["xlabel"], color="black",
                                length=params_plot["ticklength"], width=params_plot["tickwidth"])
            ax_plot.tick_params(axis="y", which="both", labelsize=font_size["ylabel"], color="black",
                                length=params_plot["ticklength"], width=params_plot["tickwidth"])
            # ax_plot.set_xlabel("X coordinate",fontsize=25)
            # ax_plot.set_ylabel("Y coordinate",fontsize=25)
            ax_plot.set_title(info_fig["title"], fontsize=40, pad=25)

            ax_plot.set_xticks(np.array([x for x in range(0, n_cells[0] - 1, 2)]))
            # ax_plot.set_xticklabels(np.array([x for x in range(0,n_cells[1]-1,2)]))
            ax_plot.set_xticklabels([str(x * 0.225) + "°" for x in range(0, n_cells[1] - 1, 2)])
            ax_plot.set_yticks(np.array([y for y in range(n_cells[1] - 2, -1, -2)]))
            # ax_plot.set_yticklabels(np.array([y for y in range(n_cells[1]-2,-1,-2)])*0.225)
            ax_plot.set_yticklabels([str(x * 0.225) + "°" for x in range(0, n_cells[1] - 1, 2)])

            plt.axis('off')
            plt.savefig(f"{path_output}t{i_frame}.png")


def make_multiple_graph_peak_latency_delays(path, list_df_latency_delays, list_df_peak_delays, list_caract, params_sim,
                                            params_plot):

    for df_latency_delays, df_peak_delays, list_value_caract in zip(list_df_latency_delays, list_df_peak_delays,
                                                                    list_caract[1]):
        make_graph_peak_latency_delays(path, df_latency_delays, df_peak_delays, list_caract[0], list_value_caract,
                                       list_caract[2], params_sim, params_plot)


def make_graph_peak_latency_delays(path, df_latency, df_ttp, list_name_caract, list_value_caract, list_unit_caract,
                                   params_sim, params_plot):
    # Plot
    print("Make plot...", end="")
    n_main_axis = params_sim["n_cells_X"] - 1
    list_color = [(0, 0, (i / (n_main_axis / 2))) if i < (n_main_axis / 2) else
                  (0, (i - n_main_axis / 2) / (n_main_axis / 2), 1.0)
                  for i in range(0, n_main_axis + 1, 1)][6:n_main_axis - 3]

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.10, top=0.89)
    ax.plot(df_ttp, c="black")
    plt.scatter(df_ttp.index, df_ttp.iloc[:, 0], label="Time to peak", marker='^', s=200, c=list_color)

    ax.plot(df_latency, c="black")
    plt.scatter(df_latency.index, df_latency.iloc[:, 0], label="Latency", marker="o", s=200, c=list_color)

    leg = ax.legend(fontsize=25)
    leg.legendHandles[0].set_color(list_color[-1])
    leg.legendHandles[1].set_color(list_color[-1])

    if list_name_caract[0] == "barSpeed":
        title = f"Latency and time to peak delays as function of\ncortical space with white bar moving at {list_value_caract[0]}°/s"
    else:
        end_title = ""
        for i, caract in enumerate(list_name_caract):
            end_title += f" {caract} {list_value_caract[i]}{list_unit_caract[i]}"
        title = f"Latency and time to peak delays as function of\ncortical spacewith white bar moving at {params_sim['speed']}°/s\n{end_title}"

    ax.set_title(title, fontsize=35, fontweight="bold", pad=40)
    ax.set_xlabel("Delay to center bar on center RF (ms)", fontsize=25, labelpad=20)
    ax.set_ylabel("Cortical space (degrees)", fontsize=25, labelpad=20)
    ax.tick_params(axis="x", which="both", labelsize=25, color="black", length=7, width=2)
    ax.tick_params(axis="y", which="both", labelsize=25, color="black", length=7, width=2)

    # Set customizable xlim
    if len(params_plot["Xlim"]) == 1:
        ax.xaxis.set_ticks(np.array(
            [i for i in range(params_plot["Xlim"][0][0], params_plot["Xlim"][0][1], params_plot["Xlim"][0][2])]))
        ax.set_xlim(params_plot["Xlim"][0][0], params_plot["Xlim"][0][1])
    elif len(params_plot["Xlim"]) > 1:
        ax.xaxis.set_ticks(np.array(
            [i for i in range(params_plot["Xlim"][i][0], params_plot["Xlim"][i][1], params_plot["Xlim"][i][2])]))
        ax.set_xlim(params_plot["Xlim"][i][0], params_plot["Xlim"][i][1])
    print("Done!")

    if len(list_name_caract) > 1:
        ext_filename = ""
        for i, caract in enumerate(list_name_caract):
            ext_filename += f" {caract}{list_value_caract[i]}{list_unit_caract[i]}"
    else:
        ext_filename = f" {caract}{list_value_caract[i]}{list_unit_caract[i]}"

    plt.savefig(f"{path}/TTP_latency_{ext_filename}_newVSDI_rfCenter.png", bbox_inches='tight')


def make_multiple_graph_peak_latency_fit_time(path, list_df_latency, list_df_sttp, list_pred, list_caract,
                                              params_sim, params_plot):
    for i in range(len(list_df_latency)):
        make_graph_peak_latency_fit_time(path, list_df_latency[i], list_df_sttp[i], list_pred[i], list_caract[0],
                                         list_caract[1][i], list_caract[2], params_sim, params_plot)


def make_graph_peak_latency_fit_time(path, df_latency, df_ttp, dict_pred, list_name_caract, list_value_caract,
                                     list_unit_caract, params_sim, params_plot):
    """
    FUNCTION TO PLOT LATENCY AND STTP DURATION WITH A DEFAULT CURVE OF THE SPEED GAVE IN PARAMS_SIM.
    """
    # Plot
    print("Make plot...", end="")

    # Make title and speed depending on parameters change between csv files
    end_title = ""
    if list_name_caract[0] == "barSpeed":
        speed = list_value_caract[0]
        if "," in speed:
            speed = float(speed.replace(",","."))
        else:
            speed = int(speed)
    else:
        speed = params_sim["speed"]
        for i, caract in enumerate(list_name_caract):
            end_title += f" {caract} {list_value_caract[i]}{list_unit_caract[i]}"

    title = f"Latency and peak time as function of\ncortical space with white bar moving at {speed}°/s\n{end_title}"

    # Make color map
    n_main_axis = params_sim["n_cells_X"] - 1
    list_color = [(0, 0, (i / (n_main_axis / 2))) if i < (n_main_axis / 2) else
                  (0, (i - n_main_axis / 2) / (n_main_axis / 2), 1.0)
                  for i in range(0, n_main_axis + 1, 1)][6:n_main_axis-3]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # Plot peak time
    ax.plot(df_ttp, c="black")
    plt.scatter(df_ttp.index, df_ttp.iloc[:, 0], label="Peak", marker='^', s=200, c=list_color)

    # Plot latency time
    ax.plot(df_latency, c="black")
    plt.scatter(df_latency.index, df_latency.iloc[:, 0], label="Latency", marker="o", s=200, c=list_color)

    # Plot default speed
    df_default_speed = pd.DataFrame(df_latency.iloc[:,0].values, index=df_latency.iloc[:,0].values/(speed/1000))
    ax.plot(df_default_speed, c="black", ls = "--", lw = 5, label = f"{speed}°/s bar")

    # Plot latency prediction
    ax.plot(dict_pred["x pred latency"], dict_pred["y pred latency"], c="orange", lw=5, label=f"Latency fit")

    # Plot peak prediction
    ax.plot(dict_pred["x pred peak"], dict_pred["y pred peak"], c="purple", lw=5, label=f"Peak fit")


    # Set legend
    leg = ax.legend(fontsize=25)
    leg.legendHandles[0].set_color(list_color[-1])
    leg.legendHandles[1].set_color(list_color[-1])

    # Plot settings
    ax.set_title(title, fontsize=35, fontweight="bold", pad=0)
    ax.set_xlabel("Time (ms)", fontsize=25, labelpad=20)
    ax.set_ylabel("Cortical space (degrees)", fontsize=25, labelpad=20)
    ax.tick_params(axis="x", which="both", labelsize=25, color="black", length=7, width=2)
    ax.tick_params(axis="y", which="both", labelsize=25, color="black", length=7, width=2)

    # Set customizable xlim
    if len(params_plot["Xlim"]) == 1:
        ax.xaxis.set_ticks(np.array(
            [i for i in range(params_plot["Xlim"][0][0], params_plot["Xlim"][0][1], params_plot["Xlim"][0][2])]))
        ax.set_xlim(params_plot["Xlim"][0][0], params_plot["Xlim"][0][1])
    elif len(params_plot["Xlim"]) > 1:
        ax.xaxis.set_ticks(np.array(
            [i for i in range(params_plot["Xlim"][i][0], params_plot["Xlim"][i][1], params_plot["Xlim"][i][2])]))
        ax.set_xlim(params_plot["Xlim"][i][0], params_plot["Xlim"][i][1])
    print("Done!")

    # Set file name extension
    if len(list_name_caract) > 1:
        ext_filename = ""
        for i, caract in enumerate(list_name_caract):
            ext_filename += f" {caract}{list_value_caract[i]}{list_unit_caract[i]}"
    else:
        ext_filename = f" {caract}{list_value_caract[i]}{list_unit_caract[i]}"

    fig.tight_layout(pad=0)

    # Save figure
    plt.savefig(f"{path}/duration_STTP_latency_{ext_filename}_newVSDI_rfCenter.png", bbox_inches='tight')


def make_peak_latency_summary_graph(fig, ax, path, df_measures, list_caract, xlabel):
    # Plot time graph
    ax[0].set_ylabel("Time (ms)", fontsize=25, labelpad=5)
    ax[0].plot(df_measures.loc[:, "Stationary peak delay"], c="purple", marker="o", markersize=10,
               label="Stationary peak delay")
    ax[0].plot(df_measures.loc[:, "Stationary end latency delay"], c="red", marker="o", markersize=10,
               label="Stationary end latency delay")
    ax[0].yaxis.set_ticks(np.array([i for i in range(-800, 300, 200)]))

    # Plot spatial graph
    ax[1].set_ylabel("Distance (°)", fontsize=25, labelpad=5)
    ax[1].plot(df_measures.loc[:, "Inflexion point"], c="green", marker="o", markersize=10,
               label="Inflexion point")
    ax[1].yaxis.set_ticks(np.array([i for i in range(0, 9, 1)]))

    # Plot speed graph
    ax[2].set_ylabel("Speed (°/s)", fontsize=25, labelpad=5)
    ax[2].plot(df_measures.loc[:, "Start latency speed"], c="deepskyblue", marker="o", markersize=10,
               label="Start latency speed")
    ax[2].plot(df_measures.loc[:, "End latency speed"], c="red", marker="o", markersize=10,
               label="End latency speed")
    ax[2].plot(df_measures.loc[:, "Peak speed"], c="purple", marker="o", markersize=10,
               label="Peak speed")
    ax[2].yaxis.set_ticks(np.array([i for i in range(0, 31, 10)]))

    # Set global parameters values
    for axe in [ax[0], ax[1], ax[2]]:
        axe.set_xlabel(xlabel, fontsize=25, labelpad=20)
        axe.legend(fontsize=15)
        axe.tick_params(axis="x", which="both", labelsize=25, color="black", length=7, width=2)
        axe.tick_params(axis="y", which="both", labelsize=25, color="black", length=7, width=2)

    fig.tight_layout(pad=5)

    # Save figure
    str_save = ""
    for c, caract in enumerate(list_caract[0]):
        str_save += f"_{list_caract[0][c]}_{list_caract[1][0][c]}to{list_caract[1][-1][c]}{list_caract[2][c]}"
    plt.savefig(f"{path}/TTP_latency_means{str_save}_newVSDI.png", bbox_inches='tight')


def mean_section_graph(ax, function, index, frame, axis, params_sim, info_fig, params_plot, font_size, x_lim=False, arr_stim=False, ax_stim=False):
    """
    Graph for get a mean horizontal or vertical section of one frame
    """
    # FOR TEST
    #test = np.ones((40, 15, 1))
    #incr = 0
    #for i in range(test.shape[0]):
    #    test[i, :] += incr
    #    incr += 1

    # Mean section graph
    index_section = np.array([i for i in range(40)]) * 0.225

    if x_lim != False:
        idx_x_lim_min = (np.abs(index_section - x_lim[0])).argmin()
        idx_x_lim_max = (np.abs(index_section - x_lim[1])).argmin()
        index_section = index_section[idx_x_lim_min:idx_x_lim_max+1]
        function = function[idx_x_lim_min:idx_x_lim_max+1, :, :]

    arr_section_mean = np.zeros((function[:, :, frame].shape[axis]))
    for i in range(function[:, :, frame].shape[axis]):
        section_sum = 0
        if axis:
            threshold = function[:, i, frame].max() / 2  # 50% of the maximum response
        else:
            threshold = function[i, :, frame].max() / 2  # 50% of the maximum response

        for j in range(function[:, :, frame].shape[1 - axis]):
            if axis:
                if function[j, i, frame] > threshold:
                    section_sum += function[j, i, frame]
            else:
                if function[i, j, frame] > threshold:
                    section_sum += function[i, j, frame]

        arr_section_mean[i] = section_sum / function[:, :, frame].shape[1 - axis]

    ax.plot(index_section, arr_section_mean, lw=4, color="Blue", label=f"{round(index[frame],4)}s")
    ax.set_xlabel(info_fig["xlabel"], fontsize=font_size["xlabel"], labelpad=params_plot["labelpad"])
    ax.set_ylabel(info_fig["ylabel"], fontsize=font_size["ylabel"])
    #ax.yaxis.set_ticks(np.array([i for i in np.linspace(0, function.max()*0.3, 10).round(4)]))
    ax.yaxis.set_ticks(np.array([i for i in np.linspace(0, 0.0002, 10).round(4)]))
    ax.tick_params(axis="x", which="both", labelsize=font_size["g_xticklabel"], color="black", length=params_plot["ticklength"], width=params_plot["tickwidth"])
    ax.tick_params(axis="y", which="both", labelsize=font_size["g_yticklabel"], color="black", length=params_plot["ticklength"], width=params_plot["tickwidth"])
    ax.legend(fontsize=15)

    # Supplementary stimuli if wanted
    if arr_stim!=False:
        #arr_stim = arr_stim[params_sim["n_transient_frame"]:-(len(arr_stim) - params_sim["n_transient_frame"] - function.shape[-1])]
        ax_stim.imshow(arr_stim[frame], aspect='auto', cmap='Greys_r')


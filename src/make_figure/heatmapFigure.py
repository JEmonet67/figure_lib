import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from figure_lib.src.make_figure.figure import Figure
from figure_lib.src.data_transform.listMatrix2D import listMatrix2D


class heatMapFigure(Figure):
    def __init__(self, list_data, dimX=-1, dimY=-1, sizeX=30, sizeY=10, dict_info_fig=None, dict_font_size=None, dict_params_fig=None, dict_params_plot=None):
        super().__init__(list_data, dimX, dimY, sizeX, sizeY, dict_info_fig, dict_font_size, dict_params_fig, dict_params_plot)

        mpl.rcParams.update({"font.size":self.dict_font_size["global"]})
        sns.dark_palette("#69d", reverse=True, as_cmap=True)
        self.list_fig = len(self.list_data[0].t)*[0]
        plt.close()
        # self.list_plot = self.set_plot()
        

    # def __repr__(self):


    def set_one_plot(self, df, ax_plot, max_value):
        # plt.clf()
        plot = sns.heatmap(df,vmin=0,vmax=max_value,cbar_kws={'label': r"Firing Rate"},cmap="Greens_r",ax=ax_plot) 
        ax_plot.set_facecolor('black')
        [ax_plot.spines[side].set_visible(True) for side in ax_plot.spines]
        [ax_plot.spines[side].set_linewidth(2) for side in ax_plot.spines]
        ax_plot.tick_params(axis="x", which="both", labelsize=self.dict_font_size["xlabel"], color="black", length=self.dict_params_plot["ticklength"], width=self.dict_params_plot["tickwidth"])
        ax_plot.tick_params(axis="y", which="both", labelsize=self.dict_font_size["ylabel"], color="black", length=self.dict_params_plot["ticklength"], width=self.dict_params_plot["tickwidth"])
        
        # plt.title(f"Firing rate of excitatory cells for conduction speed of {VC} mm/s",fontsize=25,fontweight="bold",pad=30)
        # plt.xlabel(r"Position X (mm)",fontsize=25,labelpad=15)
        # plt.ylabel(r"Position Y (mm)",fontsize=25,labelpad=15)

        return plot
    
    # def saving_heatmap():

    # def saving_video_heatmap():

    def set_plot(self, path_save=False):
        self.list_plot = []
        
        for t_ind in range(len(self.list_data[0].t)):
            t = self.list_data[0].t[t_ind]
            self.fig = self.list_fig[t_ind]
            self.fig,self.ax = plt.subplots(self.dim[0], self.dim[1], figsize = (self.size[0],self.size[1]),
            sharex=self.dict_info_fig["sharex"],sharey=self.dict_info_fig["sharey"],gridspec_kw=self.dict_params_fig)
            self.list_ax = self.ax.reshape(self.dim[0]*self.dim[1]).tolist()
            for i in range(len(self.list_data)):
                data = self.list_data[i][str(t)]
                ax_plot = self.list_ax[i]
                # if self.dim[0]==1 and self.dim[1]==1:
                #     ax_plot = self.ax
                # elif self.dim[0]==1 or self.dim[1]==1:
                #     ax_plot = self.ax[i]
                # else:
                #     ax_plot = self.ax[i//self.dim[1]][i%self.dim[1]]
                # display(data)
                self.list_plot += [self.set_one_plot(data, ax_plot, self.list_data[i].max_value)]
            plt.text(-5.2,-0.5, f"{t:.4f} s", c="black", weight="bold")
            self.set_titles()
            self.set_labels()

            self.list_fig[t_ind] = self.fig
            if type(path_save==str):
                self.fig.savefig(f"{path_save}/{t}.png")
            #plt.clf()
            plt.close()
                



    # def get_t_frame(self.t):
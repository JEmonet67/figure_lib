import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
# from src.make_figure.curve import Curve
# from src.make_figure.graph import Graph

import figure_lib.src.make_figure.curve as cu
import figure_lib.src.make_figure.graph as gr
from figure_lib.src.make_figure.figure import Figure

class graphFigure(Figure):
    def __init__(self, list_data, i=1, j=1, sizeX=10, sizeY=10, dict_info_fig=None, dict_font_size=None, dict_params_fig=None, dict_params_plot=None):
        super().__init__(list_data, i, j, sizeX, sizeY, dict_info_fig, dict_font_size, dict_params_fig, dict_params_plot)
        
        if self.dim[0]*self.dim[1] > len(self.list_data):
            print("Warning : Graph dimension is superior to dataframes dimensions")

        self.set_plot()
        self.set_titles()
        self.set_labels()
        self.set_figure_legend((self.dim[0],self.dim[1]),fontsize=self.dict_font_size["legend"])

        
    def set_plot(self):
        '''
        -------------
        Description :  
                
        -------------
        Arguments :
                var -- type, Descr
        -------------
        Returns :
                
        '''
        self.list_graph = []
        try:
            for k in range(len(self.list_data)):
                coord = (k//self.dim[1],k%self.dim[1])
                if self.dim[0]==1 and self.dim[1]==1:
                    if len(self.list_data)==1:
                        self.list_graph += [gr.Graph(self.fig, self.ax, self.list_data[k])]
                    else:
                        raise IndexError
                elif self.dim[0]==1 and self.dim[1]!=1:
                    self.list_graph += [gr.Graph(self.fig, self.ax[k%self.dim[1]], self.list_data[k],dict_params_plot=self.dict_params_plot, dict_font_size=self.dict_font_size)]
                elif self.dim[0]!=1 and self.dim[1]==1:
                    self.list_graph += [gr.Graph(self.fig, self.ax[k//self.dim[1]], self.list_data[k],dict_params_plot=self.dict_params_plot, dict_font_size=self.dict_font_size)]
                else:
                    self.list_graph += [gr.Graph(self.fig, self.ax[k//self.dim[1]][k%self.dim[1]], self.list_data[k], dict_params_plot=self.dict_params_plot, dict_font_size=self.dict_font_size)]
        
        except IndexError:
            print("{0}\n/!\/!\ Graph dimension doesn't match with pd.DataFrames dimensions /!\/!\\".format(IndexError))
            self.fig.clear()


    def set_figure_legend(self,subfigs,loc="best", anchor=None, fontsize=20, edgecolor="w"):
        '''
        -------------
        Description :  
                
        -------------
        Arguments :
                var -- type, Descr
        -------------
        Returns :
                
        '''

        if [True for i in range(len(subfigs)) if type(subfigs[i])==int] == [True]*len(subfigs):
            if len(subfigs)>len(self.list_graph):
                print("{0}\n/!\/!\ Number of subfigures to legend superior to real number of subfigures /!\/!\\".format(IndexError))
            else:
                subloc = loc
                subanchor = anchor
                subfontsize = fontsize
                subedgecolor = edgecolor

                for s in subfigs:
                    if type(loc)==list:
                        subloc = loc[s-1]
                    if type(anchor)==list:
                        subanchor = anchor[s-1]
                    if type(fontsize)==list:
                        subfontsize = fontsize[s-1]
                    if type(edgecolor)==list:
                        subedgecolor = edgecolor[s-1]


                    self.list_graph[s-1].set_graph_legend(loc=subloc, anchor=subanchor,fontsize=subfontsize, edgecolor=subedgecolor)
        else:
            print("{0}\n/!\/!\ Subfigure index have to be int /!\/!\\".format(TypeError))


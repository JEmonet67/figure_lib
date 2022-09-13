import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
# from src.make_figure.curve import Curve
# from src.make_figure.graph import Graph

import figure_lib.src.make_figure.curve as cu
import figure_lib.src.make_figure.graph as gr

class Figure():
    def __init__(self, list_gdf, i, j, dimX=10, dimY=10, dict_info_fig=None, dict_font_size=None, dict_params_fig=None, dict_params_plot=None):
        self.dim = (i,j)
        self.list_gdf = list_gdf

        if type(dict_params_fig)!=dict and dict_params_fig!=None:
            print("{0}\n/!\/!\ Parameters fig information have to be given in a dictionnary /!\/!\\".format(TypeError))
        else:
            self.dict_params_fig=dict_params_fig

        if type(dict_font_size)!=dict and dict_font_size!=None:
            print("{0}\n/!\/!\ Font size information have to be given in a dictionnary /!\/!\\".format(TypeError))
            self.dict_font_size = {"main_title":35, "subtitle":25, "xlabel":25, "ylabel":25,"g_xticklabel":25, "g_yticklabel":25}
        elif dict_font_size==None:
            self.dict_font_size = {"main_title":35, "subtitle":25, "xlabel":25, "ylabel":25,"g_xticklabel":25, "g_yticklabel":25}
        else:
            self.dict_font_size=dict_font_size
            

        if type(dict_info_fig)!=dict and dict_info_fig!=None:
            print("{0}\n/!\/!\ Figure information have to be given in a dictionnary /!\/!\\".format(TypeError))
            self.dict_info_fig = {"sharex":False, "sharey":False}
        elif dict_info_fig == None:
            self.dict_info_fig = {"sharex":False, "sharey":False}
        else:
            self.dict_info_fig=dict_info_fig
        
        self.fig,self.ax = plt.subplots(self.dim[0], self.dim[1], figsize = (dimX,dimY),
        sharex=self.dict_info_fig["sharex"],sharey=self.dict_info_fig["sharey"],gridspec_kw=dict_params_fig)

        if self.dim[0]*self.dim[1] > len(self.list_gdf):
            print("Warning : Graph dimension is superior to dataframes dimensions")
        
        self.list_graph = []
        try:
            for k in range(len(self.list_gdf)):
                coord = (k//self.dim[1],k%self.dim[1])
                if self.dim[0]==1 and self.dim[1]==1:
                    if len(self.list_gdf)==1:
                        self.list_graph += [gr.Graph(self.fig, self.ax, self.list_gdf[k])]
                    else:
                        raise IndexError
                elif self.dim[0]==1 and self.dim[1]!=1:
                    self.list_graph += [gr.Graph(self.fig, self.ax[k%self.dim[1]], self.list_gdf[k],dict_params_plot=dict_params_plot, dict_font_size=self.dict_font_size)]
                elif self.dim[0]!=1 and self.dim[1]==1:
                    self.list_graph += [gr.Graph(self.fig, self.ax[k//self.dim[1]], self.list_gdf[k],dict_params_plot=dict_params_plot, dict_font_size=self.dict_font_size)]
                else:
                    self.list_graph += [gr.Graph(self.fig, self.ax[k//self.dim[1]][k%self.dim[1]], self.list_gdf[k], dict_params_plot=dict_params_plot, dict_font_size=self.dict_font_size)]
        
        except IndexError:
            print("{0}\n/!\/!\ Graph dimension doesn't match with pd.DataFrames dimensions /!\/!\\".format(IndexError))
            self.fig.clear()

        self.set_titles()
        self.set_labels()

        
        

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

    
    def set_titles(self):
        '''
        -------------
        Description :  
                
        -------------
        Arguments :
                var -- type, Descr
        -------------
        Returns :
                
        '''
        
        try:
            title = self.dict_info_fig["title"]
            if type(title)==str:
                self.fig.suptitle(title, fontsize=self.dict_font_size["main_title"], fontweight="bold")
            else:
                print("{0}\n/!\/!\ Title name have to be str /!\/!\\".format(TypeError))
        except KeyError:
            pass
        
        try:
            subtitles = self.dict_info_fig["subtitles"]
            if type(subtitles)==list and [True for i in range(len(subtitles)) if type(subtitles[i])==str] == [True]*len(subtitles):
                if len(subtitles)<len(self.list_graph):
                    print("{0}\n/!\/!\ Number of subfigures titles inferior to the real number of subfigures /!\/!\\".format(IndexError))
                else:
                    for i in range(len(subtitles)):
                        self.list_graph[i].ax.set_title(subtitles[i],fontsize=self.dict_font_size["subtitle"])
            elif subtitles=="":
                for i in range(len(subtitles)):
                    self.list_graph[i].ax.set_title(subtitles,fontsize=self.dict_font_size["subtitle"])
            else:
                print("{0}\n/!\/!\ Subtitles names have to be a list of str /!\/!\\".format(TypeError))
        except KeyError:
            pass


    def set_labels(self):
        '''
        -------------
        Description :  
                
        -------------
        Arguments :
                var -- type, Descr
        -------------
        Returns :
                
        '''

        try:
            xlabel = self.dict_info_fig["xlabel"]
            if type(xlabel)==list and [True for i in range(len(xlabel)) if type(xlabel[i])==str] == [True]*len(xlabel):
                if len(xlabel)==1:
                    for i in range(len(self.list_graph)):
                        self.list_graph[i].ax.set_xlabel(xlabel=xlabel[0], fontsize=self.dict_font_size["xlabel"])
                elif len(xlabel)<len(self.list_graph):
                    print("{0}\n/!\/!\ Number of X labels inferior to the real number of subfigures /!\/!\\".format(IndexError))
                else:
                    
                    for i in range(len(xlabel)):
                        self.list_graph[i].ax.set_xlabel(xlabel=xlabel[i], fontsize=self.dict_font_size["xlabel"])
            elif type(xlabel)==str:
                for i in range(len(self.list_graph)):
                    self.list_graph[i].ax.set_xlabel(xlabel=xlabel, fontsize=self.dict_font_size["xlabel"])
            else:
                
                print("{0}\n/!\/!\ X labels names have to be a list of str /!\/!\\".format(TypeError))
        except KeyError:
            pass

        try:
            ylabel = self.dict_info_fig["ylabel"]
            if type(ylabel)==list and [True for i in range(len(ylabel)) if type(ylabel[i])==str] == [True]*len(ylabel):
                if len(ylabel)==1:
                    for i in range(len(self.list_graph)):
                        self.list_graph[i].ax.set_ylabel(ylabel=ylabel[0],fontsize=self.dict_font_size["ylabel"])
                elif len(ylabel)<len(self.list_graph):
                    print("{0}\n/!\/!\ Number of Y labels inferior to the real number of subfigures /!\/!\\".format(IndexError))
                else:
                    for i in range(len(ylabel)):
                        self.list_graph[i].ax.set_ylabel(ylabel=ylabel[i],fontsize=self.dict_font_size["ylabel"])
            elif type(ylabel)==str:
                for i in range(len(self.list_graph)):
                    self.list_graph[i].ax.set_ylabel(ylabel=ylabel,fontsize=self.dict_font_size["ylabel"])
            else:
                print("{0}\n/!\/!\ Y labels names have to be a list of str /!\/!\\".format(TypeError))
        except KeyError:
            pass
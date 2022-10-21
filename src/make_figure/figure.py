import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
# from src.make_figure.curve import Curve
# from src.make_figure.graph import Graph


class Figure(ABC):
    def __init__(self, list_data, i, j, sizeX=10, sizeY=10, dict_info_fig=None, dict_font_size=None, dict_params_fig=None, dict_params_plot=None):
        self.list_data = list_data
        self.dim = (i,j)
        self.size = (sizeX, sizeY)

        if type(dict_info_fig)==dict:
            self.dict_info_fig=dict_info_fig
        else:
            print("\n-- Default values taken for figure informations --")
            self.dict_info_fig = {"sharex":False, "sharey":False}
            
        if type(dict_font_size)==dict:
            self.dict_font_size=dict_font_size
        else:
            print("\n-- Default values taken for font size parameters --")
            self.dict_font_size = {"main_title":35, "subtitle":25, "xlabel":25, "ylabel":25,"g_xticklabel":25, "g_yticklabel":25, "legend":10}
            
        if type(dict_params_plot)==dict:
            self.dict_params_plot=dict_params_plot
        else:
            print("\n-- Default values taken for plot parameters --")
            self.dict_params_plot = {"ticklength":7,"tickwidth":3}
            
        if type(dict_params_fig)==dict:
            self.dict_params_fig=dict_params_fig
        else:
            print("\n-- Default values taken for figure parameters --")
            self.dict_params_fig={}

        self.fig,self.ax = plt.subplots(self.dim[0], self.dim[1], figsize = (self.size[0],self.size[1]),
        sharex=self.dict_info_fig["sharex"],sharey=self.dict_info_fig["sharey"],gridspec_kw=self.dict_params_fig)
        
        

    
    @abstractmethod
    def set_plot(self):
        pass


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
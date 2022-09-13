import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from figure_lib.src.data_transform.GraphDF import GraphDF
from figure_lib.src.make_figure.curve import Curve

import figure_lib.src.make_figure.curve as cu


class Graph():
    def __init__(self, fig, ax, gdf,dict_params_plot=None, dict_font_size=None):
        self.fig, self.ax = fig,ax
        self.gdf = gdf

        if type(dict_params_plot)!=dict and dict_params_plot!=None:
            print("{0}\n/!\/!\ Plots display parameters have to be given in a dictionnary /!\/!\\".format(TypeError))
            self.dict_params_plot = {"grid_color":"lightgray","grid_width":4,"ticklength":7,"tickwidth":3}
        elif dict_params_plot==None:
            self.dict_params_plot = {"grid_color":"lightgray","grid_width":4,"ticklength":7,"tickwidth":3}
        else:
            self.dict_params_plot=dict_params_plot


        self.ax.grid(color=self.dict_params_plot["grid_color"], linewidth=self.dict_params_plot["grid_width"])
        if str(type(gdf)).split(".")[-1][0:-2] == "GraphDF":
        # if str(type(gdf)) == "<class 'src.data_transform.GraphDF.GraphDF'>":
            self.list_curve = [cu.Curve(self.ax, self.fig, gdf.list_col[i]) for i in range(len(gdf.list_col))]

        elif type(gdf) == list and str(type(gdf[0])).split(".")[-1][0:-2] == "GraphDF":
        # elif type(gdf) == list and str(type(gdf[0]))=="<class 'src.data_transform.GraphDF.GraphDF'>":
            for elt in gdf:
                self.list_curve = [cu.Curve(self.ax, self.fig, elt.list_col[i]) for i in range(len(elt.list_col))]
    
        try:
            self.ax.tick_params(axis="x", which="both", labelsize=dict_font_size["g_xticklabel"], color="black", length=self.dict_params_plot["ticklength"], width=self.dict_params_plot["tickwidth"])
            self.ax.tick_params(axis="y", which="both", labelsize=dict_font_size["g_yticklabel"], color="black", length=self.dict_params_plot["ticklength"], width=self.dict_params_plot["tickwidth"])
        except:
            self.ax.tick_params(axis="x", which="both", labelsize=25, color="black", length=self.dict_params_plot["ticklength"], width=self.dict_params_plot["tickwidth"])
            self.ax.tick_params(axis="y", which="both", labelsize=25, color="black", length=self.dict_params_plot["ticklength"], width=self.dict_params_plot["tickwidth"])

        # self.ax.spines["bottom"].set_color("black")


    # def replot(self):

    def add_curves(self,gdf):
        '''
        -------------
        Description :  
                
        -------------
        Arguments :
                var -- type, Descr
        -------------
        Returns :
                
        '''
        if str(type(gdf)).split(".")[-1][0:-2] == "GraphDF":
        # if str(type(gdf)) == "<class 'src.data_transform.GraphDF.GraphDF'>":
            self.list_curve += [cu.Curve(self.ax, self.fig, gdf.list_col[i]) for i in range(len(gdf.list_col))]
            self.gdf = self.gdf + gdf
        elif str(type(gdf)).split(".")[-1][0:-2] == "GraphColumn":
            self.list_curve += [cu.Curve(self.ax, self.fig, gdf)]
            self.gdf = self.gdf + gdf
        elif type(gdf) == pd.DataFrame:
            try:
                n_cells_x= int(input("n_cells_x = "))
            except:
                n_cells_x = self.gdf.n_cells[0]
            try:
                n_cells_y= int(input("n_cells_y = "))
            except:
                n_cells_y = self.gdf.n_cells[1]

            if n_cells_y<=0 or n_cells_x<=0:
                print("{0}\n/!\/!\ number of cells index in X and Y should be positive /!\/!\\".format(ValueError))  
            else:
                gdf = GraphDF(gdf,self.gdf.dt,self.gdf.frame_rate,n_cells_x,n_cells_y)
                self.list_curve += [cu.Curve(self.ax, self.fig, gdf.list_col[i]) for i in range(len(gdf.list_col))]
                self.gdf = self.gdf + gdf

        else:
            print("{0}\n/!\/!\ Element to add have to be a GraphDF, GraphColumn or pd.DataFrame object /!\/!\\".format(TypeError))


    def add_theoretical_curve(self,function,start=None,end=None, dt=None):
        if type(function)==str:
            if start == None:
                start = self.gdf.data.index.min()
            if end == None:
                end = self.gdf.data.index.max()

            if dt == None:
                n = len(self.gdf.data.index)
            else:
                n = end/dt
            try:
                name_function = function.split(".")[1]
            except:
                name_function = function

            x = np.linspace(start, end, n)
            y = eval(function)
            df = pd.DataFrame(y,pd.Index(x,name="Time"),[name_function])

            self.list_curve += [cu.Curve(self.ax, self.fig, df)]

        else:
            print("{0}\n/!\/!\ Theoretical data function have to be str /!\/!\\".format(TypeError))

        return df


    def resize(self,xmin=None,xmax=None,ymin=None,ymax=None):
        '''
        -------------
        Description :  
                
        -------------
        Arguments :
                var -- type, Descr
        -------------
        Returns :
                
        '''
        if xmin==None:
            ymin=self.data.index.min()
        if xmax==None:
            ymin=self.data.index.max()
        if ymin==None:
            ymin=self.data.min().min()-5
        if ymax==None:
            ymax=self.data.max().max()+5

        self.ax.set_xlim(xmin,xmax)
        self.ax.set_ylim(ymin,ymax)


    def get_list_plot(self):
        '''
        -------------
        Description :  
                
        -------------
        Arguments :
                var -- type, Descr
        -------------
        Returns :
                
        '''
        return [curve.plot[0] for curve in self.list_curve]


    def set_graph_legend(self,loc="best", anchor=None, fontsize=20, edgecolor="w"):
        '''
        -------------
        Description :  
                
        -------------
        Arguments :
                var -- type, Descr
        -------------
        Returns :
                
        '''
        list_legend = [curve.legend for curve in self.list_curve]
        list_plot = self.get_list_plot()

        self.ax.legend(list_plot,list_legend, loc=loc, bbox_to_anchor=anchor,fontsize=fontsize,edgecolor=edgecolor)

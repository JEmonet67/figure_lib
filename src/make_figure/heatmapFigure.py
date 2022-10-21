import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from figure_lib.src.make_figure.figure import Figure
from figure_lib.src.data_transform.matrix2D import matrix2D


class heatmapFigure(Figure):
    def __init__(self, list_gdf, i=1, j=1, sizeX=10, sizeY=10, dict_info_fig=None, dict_font_size=None, dict_params_fig=None, dict_params_plot=None):
        super().__init__(list_gdf, i, j, sizeX, sizeY, dict_info_fig, dict_font_size, dict_params_fig, dict_params_plot)


    def set_plot(self):

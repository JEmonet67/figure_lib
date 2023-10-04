import pandas as pd
from figure_lib.src.get_infos.info_cell import InfoCell

class GraphColumn():
    def __init__(self,df,n_cells_x,n_cells_y,num_col=None):
        #Condition to copy columns
        if str(type(df)).split(".")[-1][0:-2]=="GraphColumn" and num_col==None:
            self.data = df.data.copy()
            self.max = df.max
            self.tmax = df.tmax
            self.name = df.name
            self.cell = df.cell
            self.n_cells =(n_cells_x,n_cells_y)

        #Condition to create columns from scratch
        else:
            if type(df) == pd.DataFrame and num_col!=None:
                self.data = df.loc[:,[df.columns[num_col]]]
            elif str(type(df)).split(".")[-1][0:-2]=="GraphDF" and num_col!=None:
            # elif str(type(df))=="<class 'src.data_transform.GraphDF.GraphDF'>" and num_col!=None:
                self.data = df.data.loc[:,[df.data.columns[num_col]]]
            else:
                print("{0}\n/!\/!\ data should be GraphDF or pd.DataFrame /!\/!\\".format(TypeError))
            
            self.max = self.get_maximums_col()["Ymax"]
            self.tmax = self.get_maximums_col()["tmax"]
            self.name = self.data.columns[0]
            self.n_cells =(n_cells_x,n_cells_y)
            self.cell = InfoCell(self.name, self.n_cells[0],self.n_cells[1])

    
    def __repr__(self):
        print("---------------------------------------------")
        print("\n",self.cell)
        print("- max = {0} at {1}s\n- Grid dimension : {2}x{3} cells".format(self.max,self.tmax,self.n_cells[0],self.n_cells[1]))
        display(self.data)
        return ""

    def __eq__(self,col):
        return self.data.equals(col.data)


    def equals(self,col, verbose=True):
            '''
            -------------
            Description :  
                    
            -------------
            Arguments :
                    var -- type, Descr
            -------------
            Returns :
                    
            '''
            if verbose:
                print("---------------------------------------------")
                print("### Equality test of GraphColumn :")
            if type(col)==type(self):
                    equality = 1
                    for attr in self.__dict__:
                            if type(self.__dict__[attr]) == pd.DataFrame:
                                    test = self.__dict__[attr].equals(col.__dict__[attr])
                            else:
                                    test = self.__dict__[attr] == col.__dict__[attr]
                            if verbose:
                                print("- Testing {0} : {1}".format(attr,test))
                            equality *= test
                    if equality:
                            if verbose:
                                print("=====> Both objects are identicals\n")
                            return True
                    else:
                            if verbose:
                                print("=====> Both objects are differents\n")
                            return False
            else:
                    if verbose:
                        print("=====> Objects are differents by their type\n")
                    return False

    def get_maximums_col(self):
        '''
        -------------
        Description :  
                Function to calcul the maximum value of a data column, the time when it happens and the index of the corresponding dataframe.
        -------------
        Arguments :
                self.data -- pandas.DataFrame, Dataframe composed of one time column and one of data.
        -------------
        Returns :
                Give a dictionary with the maximum value of the column (Ymax), the time of this pic (tmax) and the index (i_Ymax).
        '''
        dict_max = {}
        dict_max["Ymax"] = self.data.iloc[:,0].max()
        dict_max["tmax"] = self.data[self.data.iloc[:,0] == dict_max["Ymax"]].index[0]
        
        dict_max["Ymax"] = round(dict_max["Ymax"],3)
        dict_max["tmax"] = round(dict_max["tmax"],3)

        return dict_max

    # def tmax_centering_col(self):
    #     '''
    #         -------------
    #     Description :  
    #             Function to center on pic values depending from time.
    #     -------------
    #     Arguments :
    #             self.data -- pandas.DataFrame, two columns dataframe contening Time and data values of the column to center.
    #             self.tmax -- float/int, time value corresponding to the data max value.
    #     -------------
    #     Returns :
    #             Return a pandas Dataframe contening a Time column center on pic and the input data column.
    #     '''
    #     print("tmax", self.tmax)
    #     self.data.index = self.data.index-self.tmax


    def tmax_centering_col(self):
        '''
            -------------
        Description :  
                Function to center on pic values depending from time.
        -------------
        Arguments :
                self.data -- pandas.DataFrame, two columns dataframe contening Time and data values of the column to center.
                self.tmax -- float/int, time value corresponding to the data max value.
        -------------
        Returns :
                Return a pandas Dataframe contening a Time column center on pic and the input data column.
        '''
        centered_col = self.copy()
        centered_col.data.index = centered_col.data.index-centered_col.tmax

        return centered_col

    def rf_centering_col(self, pos_col, speed, size_bar, dt):
        '''
            -------------
        Description :
                Function to center on rf center.
        -------------
        Arguments :
                self.data -- pandas.DataFrame, two columns dataframe contening Time and data values of the column to center.
                self.tmax -- float/int, time value corresponding to the data max value.
        -------------
        Returns :
                Return a pandas Dataframe contening a Time column center on RF center and the input data column.
        '''
        centered_col = self.copy()
        centered_col.data.index = centered_col.data.index - (((pos_col + size_bar/2)/speed-dt)*1000)

        return centered_col
    
    def copy(self):
        '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
        return GraphColumn(self,self.n_cells[0],self.n_cells[1])

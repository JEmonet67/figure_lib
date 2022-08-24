import pandas as pd
import math
import re
from src.data_transform.GraphColumn import GraphColumn

class GraphDF():
        @classmethod 
        def preparation(cls, df, dt, frame_rate=60):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
                df.loc[:,"Time"] = df.loc[:,"Time"]*frame_rate*dt 
                df = df.set_index("Time")
                
                return df

        def __init__(self, df,dt,frame_rate,n_cells_x,n_cells_y):
                self.dt = dt
                self.frame_rate = frame_rate
                self.n_cells = (n_cells_x,n_cells_y)
                try:
                        if type(df)==str:
                                self.data = self.preparation(pd.read_csv(df),self.dt,self.frame_rate)
                        elif type(df)==pd.DataFrame:
                                self.data = self.preparation(df,self.dt,self.frame_rate)
                        else:
                                raise TypeError("/!\/!\ Path must be a str or a pd.DataFrame /!\/!\\")
                except KeyError:
                        if type(df)==str:
                                self.data = pd.read_csv(df)
                        elif type(df)==pd.DataFrame:
                                self.data = df
                        else:
                                raise TypeError("/!\/!\ Path must be a str or a pd.DataFrame /!\/!\\")

                self.list_col = [GraphColumn(self.data,self.n_cells[0],self.n_cells[1],i) for i in range(len(self.data.columns))]


        def __repr__(self):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        "<class 'src.data_transform.GraphColumn.GraphColumn'>"
                '''
                print("---------------------------------------------")
                print("\n### Parameters :\n  - dt = {0}s\n  - frame rate = {1}Hz\n  - Grid dimension : {2}x{3} cells".format(self.dt,self.frame_rate,self.n_cells[0],self.n_cells[1]))
                display(self.data)
                return ""
        

        
        def __add__(self, df):
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
                        add_df = self.copy()
                        if type(df) == pd.DataFrame:
                                try:
                                        n_cells_x= int(input("n_cells_x = "))
                                except:
                                        n_cells_x = self.n_cells[0]
                                try:
                                        n_cells_y= int(input("n_cells_y = "))
                                except:
                                        n_cells_y = self.n_cells[1]

                                if n_cells_y<=0 or n_cells_x<=0:
                                        print("{0}\n/!\/!\ number of cells index in X and Y should be positive /!\/!\\".format(ValueError))  
                                else:
                                        add_df.data = self.data.join(df)
                                        add_df.list_col += [GraphColumn(df,n_cells_x,n_cells_y,i) for i in range(len(df.columns))]

                        elif type(df) == type(self):
                                add_df.data = self.data.join(df.data)
                                add_df.list_col += df.list_col
                        elif str(type(df)) == "<class 'src.data_transform.GraphColumn.GraphColumn'>":
                                add_df.data = self.data.join(df.data)
                                add_df.list_col += [df]
                        else:
                                print("{0}\n/!\/!\ GraphDF should be add to GraphDF, GraphColumn or pd.DataFrame /!\/!\\".format(TypeError))
                                return self
                except ValueError:
                        print("{0}\n/!\/!\ Columns name are in multiple versions, suppress or rename it before merging GraphDF /!\/!\\".format(ValueError))

                return add_df.sort_by_cell_number()


        def __eq__(self,df):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''                
                if type(df)==type(self):
                        return self.data.equals(df.data)
                elif type(df) == pd.DataFrame:
                        return self.data.equals(df)
                else:
                        print("GraphDF should be compare to GraphDF or pd.DataFrame.")
                        return None

        def equals(self,gdf,verbose=True):
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
                        print("### Equality test of GraphDF :")
                if type(gdf)==type(self):
                        equality = 1
                        for elt in self.__dict__:
                                attr = self.__dict__[elt]
                                attr_other = gdf.__dict__[elt]

                                if type(attr) == pd.DataFrame:
                                        test = attr.equals(attr_other)

                                elif type(attr) == list:
                                        if len(attr)==len(attr_other) and str(type(attr[0])) == "<class 'src.data_transform.GraphColumn.GraphColumn'>":
                                                equality_list = 1
                                                for i in range(len(attr)):
                                                        equality_list *= attr[i].equals(attr_other[i],False)
                                                test = bool(equality_list)
                                        else:
                                                test = False

                                else:
                                        test = attr == attr_other
                                
                                if verbose:
                                        print("- Testing {0} : {1}".format(elt,test))
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
                return GraphDF(self.data.copy(),self.dt,self.frame_rate,self.n_cells[0],self.n_cells[1])


        def isolate_dataframe_columns(self):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
                print("\n==================== CHOICES ====================\n")
                for i in range(len(self.data.columns)):
                        print("     ",i+1," - ", self.data.columns[i])
                print("\n=================================================")

                choice = input("Select your columns by their id :\n")
                
                select_col=[]
                for i in choice:
                        if re.match(r'[\d]',i) and int(i)<=len(self.data.columns) and self.data.columns[int(i)-1] not in select_col:
                                select_col+=[self.data.columns[int(i)-1]]

                new_gdf = GraphDF(self.data.loc[:,select_col],self.dt,self.frame_rate, self.n_cells[0], self.n_cells[1])

                return new_gdf.sort_by_cell_number()

        def crop(self,Xmin,Xmax):
                '''
                -------------
                Description :  
                        Function to crop the beginning and/or the end of a data column.
                -------------
                Arguments :
                        df_col -- pandas.DataFrame, Dataframe composed of one time column and one of data.
                        Xmin -- float or int, X value corresponding to the new minimum value of the plot cropped.
                        Xmax -- float or int, X value corresponding to the new maxmimum value of the plot cropped.
                        seuil -- float, 
                -------------
                Returns :
                        Give a cropped DataFrame contening Time column and the input column.
                '''

                try:
                        if isinstance(Xmin,float) or isinstance(Xmin,int) and isinstance(Xmax,float) or isinstance(Xmax,int) and Xmin<Xmax:  
                                gdf = self.copy()
                                gdf.data = self.data[(self.data.index>=Xmin)&(self.data.index<=Xmax)]

                                return gdf
                                
                except (TypeError, ValueError):
                        print("/!\/!\ Wrong Xmin or Xmax values /!\/!\\")

    # def crop_seuil(self,Xmin,Xmax, seuil=0):
    #     if str(Xmax).upper()=="SEUIL":
    #         for colonne in self.data.columns:
    #             df_col=self.data.loc[:,[colonne]]

    #             serie_col = df_col.iloc[:,0]
    #             min_df=df_col[df_col.index>=Xmin]
    #             Xmin=min_df.iloc[[0],[0]].index[0]
    #             inflex_serie = serie_col[serie_col<seuil]
    #             inflex_X = inflex_serie.iloc[[inflex_serie.shape[0]-1]].index[0]
    #             df_col=df_col.iloc[Xmin:inflex_X,[0,1]]

    #             self.data.loc[:,colonne] = df_col[:,colonne]

    #     else:
    #         print("Wrong Xmin or Xmax values")

        
        def sort_by_cell_number(self):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
                data_sort = pd.DataFrame()

                dict_col = {re.search(r'[\d]{4}',name)[0]:self.data.loc[:,[name]] for name in self.data.columns if re.search(r'[\d]{4}',name)!=None}
                list_numbers = [key for key in dict_col]
                list_numbers.sort()

                for number in list_numbers:
                        if data_sort.equals(pd.DataFrame()):
                                data_sort = dict_col[number]
                        else:
                                data_sort = data_sort.join(dict_col[number])

                return GraphDF(data_sort,self.dt,self.frame_rate,self.n_cells[0],self.n_cells[1])


                # def sort_by_cell_number(self):
                # '''
                # -------------
                # Description :  
                        
                # -------------
                # Arguments :
                #         var -- type, Descr
                # -------------
                # Returns :
                        
                # '''
                # data_sort = pd.DataFrame()

                # dict_col = {re.search(r'[\d]{4}',name)[0]:self.data.loc[:,[name]] for name in self.data.columns if re.search(r'[\d]{4}',name)!=None}
                # list_numbers = [key for key in dict_col]
                # list_numbers.sort()

                # for number in list_numbers:
                #         if data_sort.equals(pd.DataFrame()):
                #                 data_sort = dict_col[number]
                #         else:
                #                 data_sort = data_sort.join(dict_col[number])

                # return GraphDF(data_sort,self.dt,self.frame_rate,self.n_cells[0],self.n_cells[1])


        def tmax_centering_df(self):
                '''
                -------------
                Description :  
                        Function to center on pic values depending from time.
                -------------
                Arguments :
                        df_col -- pandas.DataFrame, two columns dataframe contening Time and data values of the column to center.
                        tmax -- float/int, time value corresponding to the data max value.
                -------------
                Returns :
                        Return a list containing all DataFrame columns centered on pic.
                '''
                df_centered=[self.copy()]*len(self.list_col)

                for i in range(len(self.list_col)):
                        centered_col = self.list_col[i].tmax_centering_col()
                        df_centered[i].data = centered_col.data
                        df_centered[i].list_col = [centered_col]
                
                return df_centered


        # def add_theoretical_data(self, function):
        #         if type(function)==str:
        #                 self.data =
        #                 self.list_col =
        #         else:
        #                 print("{0}\n/!\/!\ Theoretical data function have to be str /!\/!\\".format(TypeError))
    



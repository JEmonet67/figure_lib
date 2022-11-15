import pandas as pd
import numpy as np
import math
import re
from figure_lib.src.data_transform.GraphColumn import GraphColumn
from figure_lib.src.data_transform.listMatrix2D import listMatrix2D

class GraphDF():
        @classmethod 
        def preparation(cls, df):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
                #df.loc[:,"Time"] = df.loc[:,"Time"]*frame_rate*dt 
                df = df.set_index("Time")
                
                return df

        def __init__(self, df,dt,frame_rate,n_cells_x,n_cells_y):
                self.dt = dt
                self.frame_rate = frame_rate
                self.n_cells = (n_cells_x,n_cells_y)
                try:
                        if type(df)==str:
                                self.data = self.preparation(pd.read_csv(df))
                        elif type(df)==pd.DataFrame:
                                self.data = self.preparation(df)
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
                        "GraphColumn"
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
                        elif str(type(df)).split(".")[-1][0:-2] == "GraphColumn":
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
                                        if len(attr)==len(attr_other) and str(type(attr[0])).split(".")[-1][0:-2] == "GraphColumn":
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


        def isolate_dataframe_bytype(self,choice=None):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
                
                if choice==None:
                        print("\n==================== CHOICES ====================\n")
                        list_todisplay = []
                        for i in range(len(self.list_col)):
                                col = self.list_col[i]
                                if col.cell.type not in list_todisplay:
                                        list_todisplay += [col.cell.type]
                                        print(f"     -{col.cell.type}")
                        print("\n=================================================")
                        choice = input("Select your columns by their id :\n")
                
                select_col=[]
                list_nametype = []
                for nametype in choice.replace(" ","").split(","):
                        if nametype not in list_nametype:
                                list_nametype += [nametype]
                                for col in self.list_col:
                                        if  nametype==col.cell.type:
                                                select_col+=[col.name]

                new_gdf = GraphDF(self.data.loc[:,select_col],self.dt,self.frame_rate, self.n_cells[0], self.n_cells[1])

                return new_gdf.sort_by_cell_number()

        def isolate_dataframe_byoutputs(self,choice=None):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
                
                if choice==None:
                        print("\n==================== CHOICES ====================\n")
                        list_todisplay = []
                        for i in range(len(self.list_col)):
                                col = self.list_col[i]
                                if col.cell.output not in list_todisplay:
                                        list_todisplay += [col.cell.output]
                                        print(f"     -{col.cell.output}")
                        print("\n=================================================")
                        choice = input("Select your columns by their id :\n")
                
                select_col=[]
                list_name_output = []
                for name_output in choice.replace(" ","").split(","):
                        if name_output not in list_name_output:
                                list_name_output += [name_output]
                                for col in self.list_col:
                                        if  name_output==col.cell.output:
                                                select_col+=[col.name]

                new_gdf = GraphDF(self.data.loc[:,select_col],self.dt,self.frame_rate, self.n_cells[0], self.n_cells[1])

                return new_gdf.sort_by_cell_number()

        def isolate_dataframe_columns_bynum(self,choice=None):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
                
                if choice==None:
                        print("\n==================== CHOICES ====================\n")
                        for i in range(len(self.list_col)):
                                col = self.list_col[i]
                                X = col.cell.coord["X"]
                                Y = col.cell.coord["Y"]
                                Z = col.cell.coord["Z"]
                                print(f"     {col.name}  :  ({X},{Y},{Z})")
                        print("\n=================================================")
                        choice = input("Select your columns by their id :\n")
                
                select_col=[]
                for i in choice.replace(" ","").split(","):
                        for col in self.list_col:
                                if col.cell.num == int(i) and col.name not in select_col:
                                        select_col+=[col.name]

                new_gdf = GraphDF(self.data.loc[:,select_col],self.dt,self.frame_rate, self.n_cells[0], self.n_cells[1])

                return new_gdf.sort_by_cell_number()


        # def isolate_dataframe_columns_bycoord(self,choice=None):
        #         '''
        #         -------------
        #         Description :  
                        
        #         -------------
        #         Arguments :
        #                 var -- type, Descr
        #         -------------
        #         Returns :
                        
        #         '''
                
        #         if choice==None:
        #                 print("\n==================== CHOICES ====================\n")
        #                 for i in range(len(self.list_col)):
        #                         col = self.list_col[i]
        #                         X = col.cell.coord["X"]
        #                         Y = col.cell.coord["Y"]
        #                         Z = col.cell.coord["Z"]
        #                         print(f"     {col.name}  :  ({X},{Y},{Z})")
        #                 print("\n=================================================")
        #                 choice = input("Select your columns by their id :\n")
                
        #         select_col=[]
        #         for i in choice.replace(" ","").split(","):
        #                 if re.match(r'^(\d){1,4}$',i) and int(i)<=len(self.list_col) and self.list_col[int(i)-1].name not in select_col:
        #                         select_col+=[self.data.columns[int(i)-1]]

        #         new_gdf = GraphDF(self.data.loc[:,select_col],self.dt,self.frame_rate, self.n_cells[0], self.n_cells[1])

        #         return new_gdf.sort_by_cell_number()


        def isolate_dataframe_columns_byidx(self,choice=None):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
                
                if choice==None:
                        print("\n==================== CHOICES ====================\n")
                        for i in range(len(self.list_col)):
                                col = self.list_col[i]
                                X = col.cell.coord["X"]
                                Y = col.cell.coord["Y"]
                                Z = col.cell.coord["Z"]
                                print(f"     {i+1} - {col.name}  :  ({X},{Y},{Z})")
                        print("\n=================================================")
                        choice = input("Select your columns by their id :\n")
                
                select_col=[]
                for i in choice.replace(" ","").split(","):
                        if re.match(r'^(\d){1,3}$',i) and int(i)<=len(self.list_col) and self.list_col[int(i)-1].name not in select_col:
                                select_col+=[self.list_col[int(i)-1].name]

                new_gdf = GraphDF(self.data.loc[:,select_col],self.dt,self.frame_rate, self.n_cells[0], self.n_cells[1])

                return new_gdf.sort_by_cell_number()

        def crop(self,Xmin,Xmax=-1):
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
                if Xmax ==-1:
                        Xmax = self.data.index[-1]
                try:
                        if isinstance(Xmin,float) or isinstance(Xmin,int) and isinstance(Xmax,float) or isinstance(Xmax,int) and Xmin<Xmax:  
                                gdf = self.copy()
                                gdf.data = self.data[(self.data.index>=Xmin)&(self.data.index<=Xmax)]
                                gdf.data.index = gdf.data.index - gdf.data.index[0]
                                
                                return gdf
                                
                except (TypeError, ValueError):
                        print("/!\/!\ Wrong Xmin or Xmax values /!\/!\\")


        def isolate_dataframe_rows_byname(self,choice=None):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        
                '''
                
                if choice==None:
                        print("\n==================== CHOICES ====================\n")
                        for i in self.data.index:
                                print(f"     {i}")
                        print("\n=================================================")
                        choice = input("Select your rows by their name :\n")
                
                select_row=[]
                for i in choice.replace(" ","").split(","):
                        for row in self.data.index:
                                if row == float(i) and row not in select_row:
                                        select_row+=[row]

                new_gdf = GraphDF(self.data.loc[select_row],self.dt,self.frame_rate, self.n_cells[0], self.n_cells[1])

                return new_gdf.sort_by_cell_number()
        
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

                dict_col = {}
                for name in self.data.columns:
                        if re.search(r'[\d]{1,4}',name)!=None:
                                if re.search(r'[\d]{1,4}',name)[0] not in dict_col:
                                        dict_col[re.search(r'[\d]{1,4}',name)[0]] = [self.data.loc[:,[name]]]
                                else:
                                        dict_col[re.search(r'[\d]{1,4}',name)[0]] = dict_col[re.search(r'[\d]{1,4}',name)[0]] + [self.data.loc[:,[name]]]


                # dict_col = {re.search(r'[\d]{1,4}',name)[0]:self.data.loc[:,[name]] for name in self.data.columns if re.search(r'[\d]{1,4}',name)!=None}
                list_numbers = [int(key) for key in dict_col]
                list_numbers.sort()

                for number in list_numbers:
                        if data_sort.equals(pd.DataFrame()):
                                if len(dict_col[str(number)]) > 1:
                                        data_sort = dict_col[str(number)][0]
                                        for i_col in range(1,len(dict_col[str(number)])):
                                                data_sort = data_sort.join(dict_col[str(number)][i_col])
                                else:
                                        data_sort = dict_col[str(number)][0]

                        else:
                                if len(dict_col[str(number)]) > 1:
                                        for i_col in range(len(dict_col[str(number)])):
                                                data_sort = data_sort.join(dict_col[str(number)][i_col])

                                else:
                                        data_sort = data_sort.join(dict_col[str(number)][0])
                                # data_sort = data_sort.join(dict_col[str(number)])

                return GraphDF(data_sort,self.dt,self.frame_rate,self.n_cells[0],self.n_cells[1])


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
                # li=[]
                # for i in range(len(self.list_col)):
                #         li+=[self.isolate_dataframe_columns(str(i))]
                df_centered=[self.isolate_dataframe_columns(str(i)) for i in range(len(self.list_col))]
                # df_centered=[self.copy()]*len(self.list_col)

                for i in range(len(self.list_col)):
                        centered_col = self.list_col[i].tmax_centering_col()
                        df_centered[i].data = centered_col.data

                        df_centered[i].list_col = [centered_col]

                return df_centered

        def row_to_2Dmatrix(self, i_row):
                row = self.data.iloc[i_row]
                t = round(self.data.iloc[i_row].name,4)

                x = [round(i,2) for i in np.linspace(20,0,20)]
                y = [round(i,2) for i in np.linspace(0,20,20)]

                mtx2D=pd.DataFrame(0,index=x,columns=y)
                mtx2D.index.name = "Y"
                mtx2D.columns.name = "X"

                for x in range(self.n_cells[0]):
                        for y in range(self.n_cells[1]):
                                mtx2D.iloc[self.n_cells[1]-1-y,x] = row.iloc[self.n_cells[0]*x+y]
                
                return mtx2D, t

        def export_rows_to_2DmatrixList(self):
                list_2Dmatrix = []
                list_t = []
                max_value = self.data.max().max()
                min_value = self.data.min().min()
                for i_row in range(self.data.shape[0]):
                        mtxt2D,t = self.row_to_2Dmatrix(i_row)
                        list_2Dmatrix += [mtxt2D]
                        list_t += [t]

                return list_2Dmatrix, list_t, min_value, max_value
    



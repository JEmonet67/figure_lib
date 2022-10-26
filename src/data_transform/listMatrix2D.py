import pandas as pd

class listMatrix2D():
        def __init__(self, list_df, max_value, list_t=None):
                if len(list_df) == len(list_t):
                        self.data = list_df
                        self.t = list_t
                        self.max_value = max_value
                else:
                        print(f"{ValueError}\n/!\/!\ : Le nombre de DataFrame ({len(list_df)}) ne correspond pas au nombre de pas de temps ({len(list_t)}) /!\/!\\")

        def __repr__(self):
                '''
                -------------
                Description :  
                        
                -------------
                Arguments :
                        var -- type, Descr
                -------------
                Returns :
                        ""
                '''
                print("---------------------------------------------")
                print(f"\n### Parameters :\n - t = {min(self.t)} --> {max(self.t)}")
                if type(self.data[0])==list:
                        print(f" - dim = {len(self)} heatmap of {self.data[0][0].shape}")
                else:
                        print(f" - dim = {len(self)} heatmap of {self.data[0].shape}")
                print(f" - max = {self.max_value}")
                #display(self.data)
                return ""

        def __len__(self):
                return len(self.data)

        def __getitem__(self,i):
                if type(i)==str:
                        index_t = self.t.index(float(i))
                        return self.data[index_t]
                elif type(i)==int:
                        return listMatrix2D([self.data[i]],self.max_value, [self.t[i]])


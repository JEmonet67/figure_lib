import pandas as pd

class matrix2D():
    def __init__(self, df, t=None):
        self.data = df
        self.t = t

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
                print("\n### Parameters :\n  - t = {0}s".format(self.t))
                display(self.data)
                return ""
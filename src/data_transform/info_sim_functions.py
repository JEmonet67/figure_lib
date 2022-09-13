# libraries

# def info_sim():
#     print()

#Code de calcul et création d'un .txt de tous les paramètres d'un modèle à partir des paramètres d'entrés.

def compute_speed_stim_param(dx, dt, bar_traj_len, bar_traj_duration):
    '''
    -------------
    Description :  
            Function to compute bar speed with stimulus parameters.
    -------------
    Arguments :
            delta_x -- int, Conversion factor between pixels and degrees.
            delta_t -- int, Conversion factor between frames and time in ms.
            bar_traj_len -- int, Bar trajectory length in pixels.
            bar_traj_duration -- int, Bar trajectory duration in frames.
    -------------
    Returns :
            Display speed bar value in mm/s.
    '''

    d = round(bar_traj_len/dx*0.3,3) #0.3 mm/degree correspond to the conversion factor between degrees of retina and mm.
    t = round(bar_traj_duration*dt/1000,3)

    print("For a time duration of", t, "s and a distance of", d, "mm\nthe speed computed with stimulus parameters is", round(d/t,3), "mm/s")


def compute_pic_speed(t1, t2, dSpacing, stim_len, n_cells, dt):
    '''
    -------------
    Description :  
            Function to compute pic speed generated by a stimulus on retina.
    -------------
    Arguments :
            t1 -- int, Time video of the first pic.
            t2 -- int, Time video of the second pic.
            dSpacing -- int, Number of spacing between the two cells (Difference between the two cells positions).
            stim_len -- int, Length of the stimulus (mm).
            n_cells -- int, Number of cells in the x axis of the grid cell.
            dt -- int, Time conversion factor (ms/frame).
    -------------
    Returns :
            Return pic speed value.
    '''
    
    dist = stim_len/n_cells*dSpacing
    time = (t2-t1)*60*dt/1000
    print("Distance of ", round(dist,4), "mm in ", round(time,4), "s", "consisting in a pic speed of :", round(dist/time,4))


def compute_tau(df):
    '''
    -------------
    Description :  
            Function to compute tau value by doing difference between time of the pic and time of the start of the activity.
    -------------
    Arguments :
            df -- pd.DataFrame, Dataframe contening data in function of time.
    -------------
    Returns :
            Return tau value.
    '''

    t_debut_pic = df[df.iloc[:,0]>0].index[0]
    t_pic = df[df.iloc[:,0]==df.iloc[:,0].max()].index[0]
    
    return t_pic - t_debut_pic
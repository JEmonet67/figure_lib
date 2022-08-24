

class InfoStim():
    @classmethod 
    def deg_to_mm(cls,deg):
        return round(deg*0.3,3) #0.3mm=1deg

    @classmethod 
    def mm_to_deg(cls,mm):
        return round(mm/0.3,3) #3.33deg=1mm

    def __init__(self, frame_rate, width_stim_pix, height_stim_pix, dx_deg, duration_frames, dt,traj_len_pix,traj_duration_frames):
        self.dx_deg, self.dx_SI = dx_deg, self.deg_to_mm(1/dx_deg)
        self.width_stim_pix, self.width_stim_deg, self.width_stim_SI = width_stim_pix, round(width_stim_pix/dx_deg,2), round(width_stim_pix*self.dx_SI,2)
        self.height_stim_pix, self.height_stim_deg, self.height_stim_SI = height_stim_pix, round(height_stim_pix/dx_deg,2), round(height_stim_pix*self.dx_SI,2)

        self.frame_rate = frame_rate
        self.duration_frames = duration_frames
        self.dt = dt
        self.duration_SI = round(duration_frames*self.dt,2)

        self.traj_len_pix, self.traj_len_deg, self.traj_len_mm = traj_len_pix, round(traj_len_pix/self.dx_deg,2), round(traj_len_pix*self.dx_SI,2)
        self.traj_duration_frames, self.traj_dur_SI = traj_duration_frames, round(traj_duration_frames*self.dt,2)

        self.speed_pixels = self.compute_speed_stim_param("pixels/frame")
        self.speed_deg = self.compute_speed_stim_param("deg/s")
        self.speed_mm = self.compute_speed_stim_param("mm/s")

    def __repr__(self):
        text = "\n\t--- STIMULUS ---\n### Space :\n"
        text += "Height => {0} pixels || {1} deg || {2} mm\n".format(self.width_stim_pix, self.width_stim_deg, self.width_stim_SI)
        text += "Width => {0} pixels || {1} deg || {2} mm\n".format(self.height_stim_pix, self.height_stim_deg, self.height_stim_SI)
        text += "dx => {0} deg/pixels || {1} mm/pixels\n".format(1/self.dx_deg,self.dx_SI)
        text += "\n### Object trajectory :\nLength => {0} pixels || {1} deg || {2} mm\n".format(self.traj_len_pix, self.traj_len_deg, self.traj_len_mm)
        text += "Duration => {0} frames || {1} s\n".format(self.traj_duration_frames, self.traj_dur_SI)
        text += "\n### Time :\nDuration => {0} frames || {1} s\n".format(self.duration_frames,self.duration_SI)
        text += "Frame_rate => {0} frames/s\ndt => {1} s/frame\n".format(self.frame_rate,self.dt)
        text += "\n### Speed :\nSpeed => {0} pixels/frame || {1} deg/s || {2} mm/s\n\n".format(self.speed_pixels,self.speed_deg,self.speed_mm)

        return text




    def compute_speed_stim_param(self, units):
        '''
        -------------
        Description :  
                Function to compute bar speed with stimulus parameters.
        -------------
        Arguments :
                delta_x -- int, Conversion factor between pixels and degrees.
                delta_t -- int, Conversion factor between frames and time in s.
                bar_traj_len -- int, Bar trajectory length in pixels.
                bar_traj_duration -- int, Bar trajectory duration in frames.
        -------------
        Returns :
                Display speed bar value in mm/s.
        '''
        

        if units.split("/")[0]=="mm":
            dx = self.dx_SI
            bar_traj_len = self.traj_len_mm
        elif units.split("/")[0]=="deg":
            dx = self.dx_deg
            bar_traj_len = self.traj_len_deg
        elif units.split("/")[0]=="pixels":
            dx = 1
            bar_traj_len = self.traj_len_pix

        if units.split("/")[1]=="s":
            dt = self.dt
            bar_traj_duration = self.traj_dur_SI
        elif units.split("/")[1]=="frame":
            dt = 1
            bar_traj_duration = self.traj_duration_frames

        d = round(bar_traj_len,3)
        t = round(bar_traj_duration,3)

        # print("For a time duration of", t, "s and a distance of", d, "mm\nthe speed computed with stimulus parameters is", round(d/t,3), "mm/s")
        return round(d/t,3)









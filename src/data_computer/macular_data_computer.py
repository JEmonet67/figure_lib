from src.data_computer.data_computer import DataComputer


class MacularDataComputer(DataComputer):
    def crop_t(self, dict_array_3d):
        return dict_array_3d[dict_array_3d.index >= (self.n_transient_frame+1) * self.delta_t]

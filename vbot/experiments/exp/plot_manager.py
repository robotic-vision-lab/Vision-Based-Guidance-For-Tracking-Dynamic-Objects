import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.patches as patches
from mpl_toolkits import mplot3d


class PlotManager:
    def __init__(self, exp_manager):
        self.exp_manager = exp_manager
        plt.ion()

        # create figure and axes 
        # focal points
        self.fig_fp, self.axs_fp = plt.subplots()
        self.fig_fp.suptitle('Focal points')


    def plot_fp(self):
        fp1x = self.exp_manager.tracking_manager.ellipse_params_est[0]
        fp1y = self.exp_manager.tracking_manager.ellipse_params_est[3]
        fp2x = self.exp_manager.tracking_manager.ellipse_params_est[6]
        fp2y = self.exp_manager.tracking_manager.ellipse_params_est[9]

        self.axs_fp.plot(fp1x, fp1y, color='blue', marker='.', markersize=3)
        self.axs_fp.plot(fp2x, fp2y, color='red', marker='.', markersize=3)
        plt.pause(0.00001)





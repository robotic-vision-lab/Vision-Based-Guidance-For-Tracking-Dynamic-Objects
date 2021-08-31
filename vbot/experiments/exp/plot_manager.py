import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.patches as patches
from mpl_toolkits import mplot3d


class PlotManager:
    def __init__(self, exp_manager):
        self.exp_manager = exp_manager
        self.skip_count = 0
        self.skip_step = 50
        plt.ion()

        # self.focal_points_plotter = FocalPointsPlotter(self, exp_manager, 'Focal points')
        self.uas_focal_points_plotter = UASFocalPointsPlotter(self, exp_manager, 'UAS and Focal points')


    def plot_focal_points(self):
        self.focal_points_plotter.get_data()
        self.focal_points_plotter.plot()

    def plot_uas_focal_points(self):
        self.uas_focal_points_plotter.get_data()
        self.uas_focal_points_plotter.plot3D()

    
    def plot(self):
        if self.skip_count==self.skip_step:
            self.plot_uas_focal_points()
            self.skip_count = 0

        plt.pause(0.000001)
        self.skip_count += 1



class FocalPointsPlotter:
    def __init__(self, plot_manager, exp_manager, title='Focal points'):
        self.plot_manager = plot_manager
        self.exp_manager = exp_manager
        self.title = title

        self.init_plot()
   

    def init_plot(self):
        # create figure, axis etc
        self.fig, self.axs = plt.subplots()
        self.fig.suptitle(self.title)
        self.axs.grid(True)


    def get_data(self):
        self.fp1x = self.exp_manager.tracking_manager.ellipse_params_est[0]
        self.fp1y = self.exp_manager.tracking_manager.ellipse_params_est[3]
        self.fp2x = self.exp_manager.tracking_manager.ellipse_params_est[6]
        self.fp2y = self.exp_manager.tracking_manager.ellipse_params_est[9]

    def plot(self):
        self.axs.plot(self.fp1x, self.fp1y, color='blue', marker='.', markersize=1)
        self.axs.plot(self.fp2x, self.fp2y, color='red', marker='.', markersize=1)



class UASFocalPointsPlotter:
    def __init__(self, plot_manager, exp_manager, title='UAS and Focal points'):
        self.plot_manager = plot_manager
        self.exp_manager = exp_manager
        self.title = title

        self.init_plot()
   

    def init_plot(self):
        # create figure, axis etc
        self.fig = plt.figure()
        self.axs = self.fig.add_subplot(111, projection='3d')
        self.axs.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        # self.fig, self.axs = plt.subplots()
        # self.axs.set(xlabel='X (m)', ylabel='Y (m)')
        self.fig.suptitle(self.title)
        self.axs.grid(True)


    def get_data(self):
        # focal point 1
        self.fp1x = self.exp_manager.tracking_manager.ellipse_params_est[0]
        self.fp1y = self.exp_manager.tracking_manager.ellipse_params_est[3]
        
        # focal point 2
        self.fp2x = self.exp_manager.tracking_manager.ellipse_params_est[6]
        self.fp2y = self.exp_manager.tracking_manager.ellipse_params_est[9]
        
        # uas
        self.drone_x = self.exp_manager.simulator.camera.origin[0]
        self.drone_y = self.exp_manager.simulator.camera.origin[1]
        self.drone_z = self.exp_manager.simulator.camera.altitude


    def plot3D(self):
        self.axs.scatter3D(self.fp1x, self.fp1y, 0, color='blue', marker='.', s=5, alpha=0.6)
        self.axs.scatter3D(self.fp2x, self.fp2y, 0, color='red', marker='.', s=5, alpha=0.6)
        self.axs.scatter3D(self.drone_x, self.drone_y, self.drone_z, color='green', marker='x', s=5, alpha=0.7)
        self.axs.scatter3D(self.drone_x, self.drone_y, 0, color='gray', marker='x', s=5, alpha=0.5)

    def plot2D(self):
        self.axs.plot(self.fp1x, self.fp1y, color='blue', marker='.', markersize=1, alpha=0.8)
        self.axs.plot(self.fp2x, self.fp2y, color='red', marker='.', markersize=1, alpha=0.8)
        self.axs.plot(self.drone_x, self.drone_y, color='darkgray', marker='x', markersize=2, alpha=0.8)
        self.fig.suptitle(f'{self.title}\n altitude={self.drone_z:0.2f}m')




class QuadrotorStatesPlotter:
    def __init__(self, plot_manager, exp_manager, title='Quadrotor States'):
        self.plot_manager = plot_manager
        self.exp_manager = exp_manager
        self.title = title

        self.init_plot()
        

    def init_plot(self):
        pass


        







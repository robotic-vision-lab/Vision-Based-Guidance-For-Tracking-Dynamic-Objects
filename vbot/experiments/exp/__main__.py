import os
import sys
import time
import shutil
from math import degrees, atan2
import numpy as np


from .exp_manager import ExperimentManager
from .settings import *

from .my_imports import _prep_temp_folder, bf
from .plotter import *


if __name__ == '__main__':

    EXPERIMENT_SAVE_MODE_ON = 0  # pylint: disable=bad-whitespace
    WRITE_PLOT = 0  # pylint: disable=bad-whitespace
    CONTROL_ON = 1  # pylint: disable=bad-whitespace
    TRACKER_ON = 1  # pylint: disable=bad-whitespace
    TRACKER_DISPLAY_ON = 1  # pylint: disable=bad-whitespace
    USE_TRUE_KINEMATICS = 0  # pylint: disable=bad-whitespace
    USE_REAL_CLOCK = 0  # pylint: disable=bad-whitespace
    DRAW_OCCLUSION_BARS = 0  # pylint: disable=bad-whitespace

    RUN_EXPERIMENT = 0  # pylint: disable=bad-whitespace
    RUN_TRACK_PLOT = 1  # pylint: disable=bad-whitespace

    RUN_VIDEO_WRITER = 0  # pylint: disable=bad-whitespace

    if RUN_EXPERIMENT:
        EXPERIMENT_MANAGER = ExperimentManager(save_on=EXPERIMENT_SAVE_MODE_ON,
                                               write_plot=WRITE_PLOT,
                                               control_on=CONTROL_ON,
                                               tracker_on=TRACKER_ON,
                                               tracker_display_on=TRACKER_DISPLAY_ON,
                                               use_true_kin=USE_TRUE_KINEMATICS,
                                               use_real_clock=USE_REAL_CLOCK,
                                               draw_occlusion_bars=DRAW_OCCLUSION_BARS)
        print(bf(f'\nExperiment started. [{time.strftime("%H:%M:%S")}]\n'))
        EXPERIMENT_MANAGER.run()

        print(bf(f'\n\nExperiment finished. [{time.strftime("%H:%M:%S")}]\n'))

    if RUN_TRACK_PLOT:
        FILE = open('plot_info.csv', 'r')
        
        # plot switches
        SHOW_ALL = 1    # set to 1 to show all plots 

        SHOW_CARTESIAN_PLOTS = 1
        SHOW_LOS_KIN_1 = 1
        SHOW_LOS_KIN_2 = 1
        SHOW_ACCELERATIONS = 1
        SHOW_TRAJECTORIES = 1
        SHOW_SPEED_HEADING = 1
        SHOW_ALTITUDE_PROFILE = 0
        SHOW_3D_TRAJECTORIES = 1
        SHOW_DELTA_TIME_PROFILE = 0
        SHOW_Y1_Y2 = 0

        # _TIME = []
        # _R = []
        # _THETA = []
        # _V_THETA = []
        # _V_R = []
        # _DRONE_POS_X = []
        # _DRONE_POS_Y = []
        # _CAR_POS_X = []
        # _CAR_POS_Y = []
        # _DRONE_ACC_X = []
        # _DRONE_ACC_Y = []
        # _DRONE_ACC_LAT = []
        # _DRONE_ACC_LNG = []
        # _CAR_VEL_X = []
        # _CAR_VEL_Y = []
        # _TRACKED_CAR_POS_X = []
        # _TRACKED_CAR_POS_Y = []
        # _TRACKED_CAR_VEL_X = []
        # _TRACKED_CAR_VEL_Y = []
        # _CAM_ORIGIN_X = []
        # _CAM_ORIGIN_Y = []
        # _DRONE_SPEED = []
        # _DRONE_ALPHA = []
        # _DRONE_VEL_X = []
        # _DRONE_VEL_Y = []
        # _MEASURED_CAR_POS_X = []
        # _MEASURED_CAR_POS_Y = []
        # _MEASURED_CAR_VEL_X = []
        # _MEASURED_CAR_VEL_Y = []
        # _DRONE_ALTITUDE = []
        # _ABS_DEN = []
        # _MEASURED_R = []
        # _MEASURED_THETA = []
        # _MEASURED_V_R = []
        # _MEASURED_V_THETA = []
        # _TRUE_R = []
        # _TRUE_THETA = []
        # _TRUE_V_R = []
        # _TRUE_V_THETA = []
        # _DELTA_TIME = []
        # _Y1 = []
        # _Y2 = []
        # _CAR_SPEED = []
        # _CAR_HEADING = []
        # _TRUE_Y1 = []
        # _TRUE_Y2 = []
        # _OCC_CASE = []


        TIME = []

        FP_1_X = []
        FP_1_Y = []
        FP_1_VX = []
        FP_1_VY = []
        FP_1_AX = []
        FP_1_AY = []
        FP_1_R = []
        FP_1_THETA = []
        FP_1_V_R = []
        FP_1_V_THETA = []
        FP_1_SPEED = []
        FP_1_HEADING = []
        FP_1_ACC = []
        FP_1_DELTA = []
        FP_2_X = []
        FP_2_Y = []
        FP_2_VX = []
        FP_2_VY = []
        FP_2_AX = []
        FP_2_AY = []
        FP_2_R = []
        FP_2_THETA = []
        FP_2_V_R = []
        FP_2_V_THETA = []
        FP_2_SPEED = []
        FP_2_HEADING = []
        FP_2_ACC = []
        FP_2_DELTA = []
        Y_1 = []
        Y_2 = []
        A_LAT = []
        A_LNG = []
        S = []
        C = []
        Z_W = []
        S_DOT = []
        C_DOT = []
        Z_W_DOT = []
        AZ_S = []
        AZ_C = []
        AZ_Z = []
        AZ = []

        T_1_OCCLUSION_CASE = []
        T_1_X_MEAS = []
        T_1_Y_MEAS = []
        T_1_R_MEAS = []
        T_1_THETA_MEAS = []
        T_1_X_EST = []
        T_1_Y_EST = []
        T_1_VX_EST = []
        T_1_VY_EST = []
        T_1_AX_EST = []
        T_1_AY_EST = []
        T_1_R_EST = []
        T_1_THETA_EST = []
        T_1_V_R_EST = []
        T_1_V_THETA_EST = []
        T_1_SPEED_EST = []
        T_1_BETA_EST = []
        T_1_ACC_EST = []
        T_1_DELTA_EST = []
        T_1_TRUE_R = []
        T_1_TRUE_THETA = []
        T_1_TRUE_V_R = []
        T_1_TRUE_V_THETA = []

        T_2_OCCLUSION_CASE = []
        T_2_X_MEAS = []
        T_2_Y_MEAS = []
        T_2_R_MEAS = []
        T_2_THETA_MEAS = []
        T_2_X_EST = []
        T_2_Y_EST = []
        T_2_VX_EST = []
        T_2_VY_EST = []
        T_2_AX_EST = []
        T_2_AY_EST = []
        T_2_R_EST = []
        T_2_THETA_EST = []
        T_2_V_R_EST = []
        T_2_V_THETA_EST = []
        T_2_SPEED_EST = []
        T_2_BETA_EST = []
        T_2_ACC_EST = []
        T_2_DELTA_EST = []
        T_2_TRUE_R = []
        T_2_TRUE_THETA = []
        T_2_TRUE_V_R = []
        T_2_TRUE_V_THETA = []

        T_3_OCCLUSION_CASE = []
        T_3_X_MEAS = []
        T_3_Y_MEAS = []
        T_3_R_MEAS = []
        T_3_THETA_MEAS = []
        T_3_X_EST = []
        T_3_Y_EST = []
        T_3_VX_EST = []
        T_3_VY_EST = []
        T_3_AX_EST = []
        T_3_AY_EST = []
        T_3_R_EST = []
        T_3_THETA_EST = []
        T_3_V_R_EST = []
        T_3_V_THETA_EST = []
        T_3_SPEED_EST = []
        T_3_BETA_EST = []
        T_3_ACC_EST = []
        T_3_DELTA_EST = []
        T_3_TRUE_R = []
        T_3_TRUE_THETA = []
        T_3_TRUE_V_R = []
        T_3_TRUE_V_THETA = []

        DRONE_POS_X = []
        DRONE_POS_Y = []
        DRONE_VEL_X = []
        DRONE_VEL_Y = []
        CAM_ORIGIN_X = []
        CAM_ORIGIN_Y = []
        DRONE_POS_X_W = []
        DRONE_POS_Y_W = []
        DRONE_SPEED = []
        DRONE_ALPHA = []

        # get all the data in memory
        for line in FILE.readlines():
            # skip first line (header)
            if line.split(',')[0].strip().lower()=='time':
                continue
            data = tuple(map(float, list(map(str.strip, line.strip().split(',')))))
            TIME.append(data[0])
            FP_1_X.append(data[1])
            FP_1_Y.append(data[2])
            FP_1_VX.append(data[3])
            FP_1_VY.append(data[4])
            FP_1_AX.append(data[5])
            FP_1_AY.append(data[6])
            FP_1_R.append(data[7])
            FP_1_THETA.append(data[8])
            FP_1_V_R.append(data[9])
            FP_1_V_THETA.append(data[10])
            FP_1_SPEED.append(data[11])
            FP_1_HEADING.append(data[12])
            FP_1_ACC.append(data[13])
            FP_1_DELTA.append(data[14])
            FP_2_X.append(data[15])
            FP_2_Y.append(data[16])
            FP_2_VX.append(data[17])
            FP_2_VY.append(data[18])
            FP_2_AX.append(data[19])
            FP_2_AY.append(data[20])
            FP_2_R.append(data[21])
            FP_2_THETA.append(data[22])
            FP_2_V_R.append(data[23])
            FP_2_V_THETA.append(data[24])
            FP_2_SPEED.append(data[25])
            FP_2_HEADING.append(data[26])
            FP_2_ACC.append(data[27])
            FP_2_DELTA.append(data[28])
            Y_1.append(data[29])
            Y_2.append(data[30])
            A_LAT.append(data[31])
            A_LNG.append(data[32])
            S.append(data[33])
            C.append(data[34])
            Z_W.append(data[35])
            S_DOT.append(data[36])
            C_DOT.append(data[37])
            Z_W_DOT.append(data[38])
            AZ_S.append(data[39])
            AZ_C.append(data[40])
            AZ_Z.append(data[41])
            AZ.append(data[42])
            T_1_OCCLUSION_CASE.append(data[43])
            T_1_X_MEAS.append(data[44])
            T_1_Y_MEAS.append(data[45])
            T_1_R_MEAS.append(data[46])
            T_1_THETA_MEAS.append(data[47])
            T_1_X_EST.append(data[48])
            T_1_Y_EST.append(data[49])
            T_1_VX_EST.append(data[50])
            T_1_VY_EST.append(data[51])
            T_1_AX_EST.append(data[52])
            T_1_AY_EST.append(data[53])
            T_1_R_EST.append(data[54])
            T_1_THETA_EST.append(data[55])
            T_1_V_R_EST.append(data[56])
            T_1_V_THETA_EST.append(data[57])
            T_1_SPEED_EST.append(data[58])
            T_1_BETA_EST.append(data[59])
            T_1_ACC_EST.append(data[60])
            T_1_DELTA_EST.append(data[61])
            T_1_TRUE_R.append(data[62])
            T_1_TRUE_THETA.append(data[63])
            T_1_TRUE_V_R.append(data[64])
            T_1_TRUE_V_THETA.append(data[65])
            T_2_OCCLUSION_CASE.append(data[66])
            T_2_X_MEAS.append(data[67])
            T_2_Y_MEAS.append(data[68])
            T_2_R_MEAS.append(data[69])
            T_2_THETA_MEAS.append(data[70])
            T_2_X_EST.append(data[71])
            T_2_Y_EST.append(data[72])
            T_2_VX_EST.append(data[73])
            T_2_VY_EST.append(data[74])
            T_2_AX_EST.append(data[75])
            T_2_AY_EST.append(data[76])
            T_2_R_EST.append(data[77])
            T_2_THETA_EST.append(data[78])
            T_2_V_R_EST.append(data[79])
            T_2_V_THETA_EST.append(data[80])
            T_2_SPEED_EST.append(data[81])
            T_2_BETA_EST.append(data[82])
            T_2_ACC_EST.append(data[83])
            T_2_DELTA_EST.append(data[84])
            T_2_TRUE_R.append(data[85])
            T_2_TRUE_THETA.append(data[86])
            T_2_TRUE_V_R.append(data[87])
            T_2_TRUE_V_THETA.append(data[88])
            T_3_OCCLUSION_CASE.append(data[89])
            T_3_X_MEAS.append(data[90])
            T_3_Y_MEAS.append(data[91])
            T_3_R_MEAS.append(data[92])
            T_3_THETA_MEAS.append(data[93])
            T_3_X_EST.append(data[94])
            T_3_Y_EST.append(data[95])
            T_3_VX_EST.append(data[96])
            T_3_VY_EST.append(data[97])
            T_3_AX_EST.append(data[98])
            T_3_AY_EST.append(data[99])
            T_3_R_EST.append(data[100])
            T_3_THETA_EST.append(data[101])
            T_3_V_R_EST.append(data[102])
            T_3_V_THETA_EST.append(data[103])
            T_3_SPEED_EST.append(data[104])
            T_3_BETA_EST.append(data[105])
            T_3_ACC_EST.append(data[106])
            T_3_DELTA_EST.append(data[107])
            T_3_TRUE_R.append(data[108])
            T_3_TRUE_THETA.append(data[109])
            T_3_TRUE_V_R.append(data[110])
            T_3_TRUE_V_THETA.append(data[111])
            DRONE_POS_X.append(data[112])
            DRONE_POS_Y.append(data[113])
            DRONE_VEL_X.append(data[114])
            DRONE_VEL_Y.append(data[115])
            CAM_ORIGIN_X.append(data[116])
            CAM_ORIGIN_Y.append(data[117])
            DRONE_POS_X_W.append(data[118])
            DRONE_POS_Y_W.append(data[119])
            DRONE_SPEED.append(data[120])
            DRONE_ALPHA.append(data[121])

        FILE.close()

        # plot
        if len(TIME) < 5:
            print('Not enough data to plot.')
            sys.exit()
        import matplotlib.pyplot as plt
        import scipy.stats as st

        _PATH = f'./sim_outputs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        _prep_temp_folder(os.path.realpath(_PATH))

        # copy the plot_info file to the where plots figured will be saved
        shutil.copyfile('plot_info.csv', f'{_PATH}/plot_info.csv')
        plt.style.use(['seaborn-paper', 'fast'])

        los_1_plotter = LOS1DataPlotter(_PATH,
                                        TIME,
                                        T_1_TRUE_R,
                                        T_1_R_MEAS,
                                        T_1_R_EST,
                                        T_2_TRUE_R,
                                        T_2_R_MEAS,
                                        T_2_R_EST,
                                        T_3_TRUE_R,
                                        T_3_R_MEAS,
                                        T_3_R_EST,
                                        FP_1_R,
                                        FP_2_R,
                                        T_1_TRUE_THETA,
                                        T_1_THETA_MEAS,
                                        T_1_THETA_EST,
                                        T_2_TRUE_THETA,
                                        T_2_THETA_MEAS,
                                        T_2_THETA_EST,
                                        T_3_TRUE_THETA,
                                        T_3_THETA_MEAS,
                                        T_3_THETA_EST,
                                        FP_1_THETA,
                                        FP_2_THETA
                                        )

        los_1_plotter.plot()

        los_2_plotter = LOS2DataPlotter(_PATH,
                                        TIME,
                                        T_1_TRUE_V_R,
                                        T_1_V_R_EST,
                                        T_2_TRUE_V_R,
                                        T_2_V_R_EST,
                                        T_3_TRUE_V_R,
                                        T_3_V_R_EST,
                                        FP_1_V_R,
                                        FP_2_V_R,
                                        T_1_TRUE_V_THETA,
                                        T_1_V_THETA_EST,
                                        T_2_TRUE_V_THETA,
                                        T_2_V_THETA_EST,
                                        T_3_TRUE_V_THETA,
                                        T_3_V_THETA_EST,
                                        FP_1_V_THETA,
                                        FP_2_V_THETA
                                        )

        los_2_plotter.plot()

        accl_comm_plotter = AccelerationCommandDataPlotter(_PATH,
                                                           TIME,
                                                           A_LAT,
                                                           A_LNG,
                                                           AZ
                                                           )

        accl_comm_plotter.plot()

        obj_func_plotter = ObjectiveFunctionDataPlotter(_PATH,
                                                        TIME,
                                                        Y_1,
                                                        Y_2
                                                        )

        obj_func_plotter.plot()

        speeds_headings_plotter = SpeedsHeadingsDataPlotter(_PATH,
                                                            TIME,
                                                            T_1_SPEED_EST,
                                                            T_2_SPEED_EST,
                                                            T_3_SPEED_EST,
                                                            FP_1_SPEED,
                                                            FP_2_SPEED,
                                                            DRONE_SPEED,
                                                            T_1_BETA_EST,
                                                            T_2_BETA_EST,
                                                            T_3_BETA_EST,
                                                            FP_1_HEADING,
                                                            FP_2_HEADING,
                                                            DRONE_ALPHA
                                                            )

        speeds_headings_plotter.plot()

        trajectory_world_plotter = TrajectoryWorldDataPlotter(_PATH,
                                                              TIME,
                                                              T_1_X_EST,
                                                              T_1_Y_EST,
                                                              T_2_X_EST,
                                                              T_2_Y_EST,
                                                              T_3_X_EST,
                                                              T_3_Y_EST,
                                                              FP_1_X,
                                                              FP_1_Y,
                                                              FP_2_X,
                                                              FP_2_Y,
                                                              DRONE_POS_X_W,
                                                              DRONE_POS_Y_W,
                                                              )

        trajectory_world_plotter.plot()

        plt.show()

        # # -------------------------------------------------------------------------------- figure 1
        # # line of sight kinematics 1
        # if SHOW_ALL or SHOW_LOS_KIN_1:
        #     f0, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.25})
        #     if SUPTITLE_ON:
        #         f0.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)

        #     # t vs r
        #     axs[0].plot(
        #         _TIME,
        #         _MEASURED_R,
        #         color='forestgreen',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$measured\ r$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         _R,
        #         color='royalblue',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$estimated\ r$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         _TRUE_R,
        #         color='red',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$true\ r$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         [i*10 for i in _OCC_CASE],
        #         color='orange',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$case\ r$',
        #         alpha=0.9)

        #     axs[0].legend(loc='upper right')
        #     axs[0].set(ylabel=r'$r\ (m)$')
        #     axs[0].set_title(r'$\mathbf{r}$', fontsize=SUB_TITLE_FONT_SIZE)

        #     # t vs θ
        #     axs[1].plot(
        #         _TIME,
        #         _MEASURED_THETA,
        #         color='forestgreen',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$measured\ \theta$',
        #         alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         _THETA,
        #         color='royalblue',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$estimated\ \theta$',
        #         alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         _TRUE_THETA,
        #         color='red',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$true\ \theta$',
        #         alpha=0.9)

        #     axs[1].legend(loc='upper right')
        #     axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$\theta\ (^{\circ})$')
        #     axs[1].set_title(r'$\mathbf{\theta}$', fontsize=SUB_TITLE_FONT_SIZE)

        #     f0.savefig(f'{_PATH}/1_los1.png', dpi=300)
        #     f0.show()

        # # -------------------------------------------------------------------------------- figure 2
        # # line of sight kinematics 2
        # if SHOW_ALL or SHOW_LOS_KIN_2:
        #     f1, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.25})
        #     if SUPTITLE_ON:
        #         f1.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ II}$', fontsize=TITLE_FONT_SIZE)

        #     # t vs vr
        #     axs[0].plot(
        #         _TIME,
        #         _MEASURED_V_R,
        #         color='palegreen',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$measured\ V_{r}$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         _V_R,
        #         color='royalblue',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_2,
        #         label=r'$estimated\ V_{r}$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         _TRUE_V_R,
        #         color='red',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$true\ V_{r}$',
        #         alpha=0.9)

        #     axs[0].legend(loc='upper right')
        #     axs[0].set(ylabel=r'$V_{r}\ (\frac{m}{s})$')
        #     axs[0].set_title(r'$\mathbf{V_{r}}$', fontsize=SUB_TITLE_FONT_SIZE)

        #     # t vs vtheta
        #     axs[1].plot(
        #         _TIME,
        #         _MEASURED_V_THETA,
        #         color='palegreen',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$measured\ V_{\theta}$',
        #         alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         _V_THETA,
        #         color='royalblue',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_2,
        #         label=r'$estimated\ V_{\theta}$',
        #         alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         _TRUE_V_THETA,
        #         color='red',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$true\ V_{\theta}$',
        #         alpha=0.9)

        #     axs[1].legend(loc='upper right')
        #     axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$V_{\theta}\ (\frac{m}{s})$')
        #     axs[1].set_title(r'$\mathbf{V_{\theta}}$', fontsize=SUB_TITLE_FONT_SIZE)

        #     f1.savefig(f'{_PATH}/1_los2.png', dpi=300)
        #     f1.show()

        # # -------------------------------------------------------------------------------- figure 2
        # # acceleration commands
        # if SHOW_ALL or SHOW_ACCELERATIONS:
        #     f2, axs = plt.subplots()
        #     if SUPTITLE_ON:
        #         f2.suptitle(r'$\mathbf{Acceleration\ commands}$', fontsize=TITLE_FONT_SIZE)

        #     axs.plot(
        #         _TIME,p
        #         _DRONE_ACC_LAT,
        #         color='forestgreen',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$a_{lat}$',
        #         alpha=0.9)
        #     axs.plot(
        #         _TIME,
        #         _DRONE_ACC_LNG,
        #         color='deeppink',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$a_{long}$',
        #         alpha=0.9)
        #     axs.legend()
        #     axs.set(xlabel=r'$time\ (s)$', ylabel=r'$acceleration\ (\frac{m}{s_{2}})$')

        #     f2.savefig(f'{_PATH}/2_accel.png', dpi=300)
        #     f2.show()

        # # -------------------------------------------------------------------------------- figure 3
        # # trajectories
        # if SHOW_ALL or SHOW_TRAJECTORIES:
        #     f3, axs = plt.subplots(2, 1, gridspec_kw={'hspace': 0.4})
        #     if SUPTITLE_ON:
        #         f3.suptitle(
        #             r'$\mathbf{Vehicle\ and\ UAS\ True\ Trajectories}$',
        #             fontsize=TITLE_FONT_SIZE)

        #     ndx = np.array(_DRONE_POS_X) + np.array(_CAM_ORIGIN_X)
        #     ncx = np.array(_CAR_POS_X) + np.array(_CAM_ORIGIN_X)
        #     ndy = np.array(_DRONE_POS_Y) + np.array(_CAM_ORIGIN_Y)
        #     ncy = np.array(_CAR_POS_Y) + np.array(_CAM_ORIGIN_Y)

        #     axs[0].plot(
        #         ndx,
        #         ndy,
        #         color='darkslategray',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$UAS$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         ncx,
        #         ncy,
        #         color='limegreen',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_2,
        #         label=r'$Vehicle$',
        #         alpha=0.9)
        #     axs[0].set(ylabel=r'$y\ (m)$')
        #     axs[0].set_title(r'$\mathbf{World\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs[0].legend()

        #     ndx = np.array(_DRONE_POS_X)
        #     ncx = np.array(_CAR_POS_X)
        #     ndy = np.array(_DRONE_POS_Y)
        #     ncy = np.array(_CAR_POS_Y)

        #     x_pad = (max(ncx) - min(ncx)) * 0.05
        #     y_pad = (max(ncy) - min(ncy)) * 0.05
        #     xl = max(abs(max(ncx)), abs(min(ncx))) + x_pad
        #     yl = max(abs(max(ncy)), abs(min(ncy))) + y_pad
        #     axs[1].plot(
        #         ndx,
        #         ndy,
        #         color='darkslategray',
        #         marker='+',
        #         markersize=10,
        #         label=r'$UAS$',
        #         alpha=0.7)
        #     axs[1].plot(
        #         ncx,
        #         ncy,
        #         color='limegreen',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_2,
        #         label=r'$Vehicle$',
        #         alpha=0.9)
        #     axs[1].set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$')
        #     axs[1].set_title(r'$\mathbf{Camera\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs[1].legend(loc='lower right')
        #     axs[1].set_xlim(-xl, xl)
        #     axs[1].set_ylim(-yl, yl)
        #     f3.savefig(f'{_PATH}/3_traj.png', dpi=300)
        #     f3.show()

        # # -------------------------------------------------------------------------------- figure 4
        # # true and estimated trajectories
        # if SHOW_ALL or SHOW_CARTESIAN_PLOTS:
        #     f4, axs = plt.subplots()
        #     if SUPTITLE_ON:
        #         f4.suptitle(
        #             r'$\mathbf{Vehicle\ True\ and\ Estimated\ Trajectories}$',
        #             fontsize=TITLE_FONT_SIZE)

        #     axs.plot(
        #         _TRACKED_CAR_POS_X,
        #         _TRACKED_CAR_POS_Y,
        #         color='darkturquoise',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$estimated\ trajectory$',
        #         alpha=0.9)
        #     axs.plot(
        #         _CAR_POS_X,
        #         _CAR_POS_Y,
        #         color='crimson',
        #         linestyle=':',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$true\ trajectory$',
        #         alpha=0.9)
        #     axs.set_title(r'$\mathbf{camera\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs.legend()
        #     axs.axis('equal')
        #     axs.set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$')
        #     f4.savefig(f'{_PATH}/4_traj_comp.png', dpi=300)
        #     f4.show()

        # # -------------------------------------------------------------------------------- figure 5
        # # true and tracked pos
        # if SHOW_ALL or SHOW_CARTESIAN_PLOTS:
        #     f4, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.4})
        #     if SUPTITLE_ON:
        #         f4.suptitle(
        #             r'$\mathbf{Vehicle\ True\ and\ Estimated\ Positions}$',
        #             fontsize=TITLE_FONT_SIZE)

        #     axs[0].plot(
        #         _TIME,
        #         _TRACKED_CAR_POS_X,
        #         color='rosybrown',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$estimated\ x$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         _CAR_POS_X,
        #         color='red',
        #         linestyle=':',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$true\ x$',
        #         alpha=0.9)
        #     axs[0].set(ylabel=r'$x\ (m)$')
        #     axs[0].set_title(r'$\mathbf{x}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs[0].legend()
        #     axs[1].plot(
        #         _TIME,
        #         _TRACKED_CAR_POS_Y,
        #         color='mediumseagreen',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$estimated\ y$',
        #         alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         _CAR_POS_Y,
        #         color='green',
        #         linestyle=':',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$true\ y$',
        #         alpha=0.9)
        #     axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$y\ (m)$')
        #     axs[1].set_title(r'$\mathbf{y}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs[1].legend()
        #     f4.savefig(f'{_PATH}/5_pos_comp.png', dpi=300)
        #     f4.show()

        # # -------------------------------------------------------------------------------- figure 6
        # # true and tracked velocities
        # if SHOW_ALL or SHOW_CARTESIAN_PLOTS:
        #     f5, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.4})
        #     if SUPTITLE_ON:
        #         f5.suptitle(
        #             r'$\mathbf{True,\ Measured\ and\ Estimated\ Vehicle\ Velocities}$',
        #             fontsize=TITLE_FONT_SIZE)

        #     axs[0].plot(
        #         _TIME,
        #         _MEASURED_CAR_VEL_X,
        #         color='paleturquoise',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$measured\ V_x$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         _TRACKED_CAR_VEL_X,
        #         color='darkturquoise',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$estimated\ V_x$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         _CAR_VEL_X,
        #         color='crimson',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_2,
        #         label=r'$true\ V_x$',
        #         alpha=0.7)
        #     axs[0].set(ylabel=r'$V_x\ (\frac{m}{s})$')
        #     axs[0].set_title(r'$\mathbf{V_x}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs[0].legend(loc='upper right')

        #     axs[1].plot(
        #         _TIME,
        #         _MEASURED_CAR_VEL_Y,
        #         color='paleturquoise',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$measured\ V_y$',
        #         alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         _TRACKED_CAR_VEL_Y,
        #         color='darkturquoise',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$estimated\ V_y$',
        #         alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         _CAR_VEL_Y,
        #         color='crimson',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_2,
        #         label=r'$true\ V_y$',
        #         alpha=0.7)
        #     axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$V_y\ (\frac{m}{s})$')
        #     axs[1].set_title(r'$\mathbf{V_y}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs[1].legend(loc='upper right')

        #     f5.savefig(f'{_PATH}/6_vel_comp.png', dpi=300)
        #     f5.show()

        # # -------------------------------------------------------------------------------- figure 7
        # # speed and heading
        # if SHOW_ALL or SHOW_SPEED_HEADING:
        #     f6, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.4})
        #     if SUPTITLE_ON:
        #         f6.suptitle(
        #             r'$\mathbf{Vehicle\ and\ UAS,\ Speed\ and\ Heading}$',
        #             fontsize=TITLE_FONT_SIZE)
        #     c_speed = (CAR_INITIAL_VELOCITY[0]**2 + CAR_INITIAL_VELOCITY[1]**2)**0.5
        #     c_heading = degrees(atan2(CAR_INITIAL_VELOCITY[1], CAR_INITIAL_VELOCITY[0]))

        #     axs[0].plot(_TIME,
        #                 _CAR_SPEED,
        #                 color='lightblue',
        #                 linestyle='-',
        #                 linewidth=LINE_WIDTH_1,
        #                 label=r'$|V_{vehicle}|$',
        #                 alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         _DRONE_SPEED,
        #         color='blue',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$|V_{UAS}|$',
        #         alpha=0.9)
        #     axs[0].set(ylabel=r'$|V|\ (\frac{m}{s})$')
        #     axs[0].set_title(r'$\mathbf{speed}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs[0].legend()

        #     axs[1].plot(_TIME, _CAR_HEADING, color='lightgreen',
        #                 linestyle='-', linewidth=LINE_WIDTH_2, label=r'$\angle V_{vehicle}$', alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         _DRONE_ALPHA,
        #         color='green',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$\angle V_{UAS}$',
        #         alpha=0.9)
        #     axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$\angle V\ (^{\circ})$')
        #     axs[1].set_title(r'$\mathbf{heading}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs[1].legend()

        #     f6.savefig(f'{_PATH}/7_speed_head.png', dpi=300)
        #     f6.show()

        # # -------------------------------------------------------------------------------- figure 7
        # # altitude profile
        # if SHOW_ALL or SHOW_ALTITUDE_PROFILE:
        #     f7, axs = plt.subplots()
        #     if SUPTITLE_ON:
        #         f7.suptitle(r'$\mathbf{Altitude\ profile}$', fontsize=TITLE_FONT_SIZE)
        #     axs.plot(
        #         _TIME,
        #         _DRONE_ALTITUDE,
        #         color='darkgoldenrod',
        #         linestyle='-',
        #         linewidth=2,
        #         label=r'$altitude$',
        #         alpha=0.9)
        #     axs.set(xlabel=r'$time\ (s)$', ylabel=r'$z\ (m)$')

        #     f7.savefig(f'{_PATH}/8_alt_profile.png', dpi=300)
        #     f7.show()

        # # -------------------------------------------------------------------------------- figure 7
        # # 3D Trajectories
        # ndx = np.array(_DRONE_POS_X) + np.array(_CAM_ORIGIN_X)
        # ncx = np.array(_CAR_POS_X) + np.array(_CAM_ORIGIN_X)
        # ndy = np.array(_DRONE_POS_Y) + np.array(_CAM_ORIGIN_Y)
        # ncy = np.array(_CAR_POS_Y) + np.array(_CAM_ORIGIN_Y)

        # if SHOW_ALL or SHOW_3D_TRAJECTORIES:
        #     f8 = plt.figure()
        #     if SUPTITLE_ON:
        #         f8.suptitle(r'$\mathbf{3D\ Trajectories}$', fontsize=TITLE_FONT_SIZE)
        #     axs = f8.add_subplot(111, projection='3d')
        #     axs.plot3D(
        #         ncx,
        #         ncy,
        #         0,
        #         color='limegreen',
        #         linestyle='-',
        #         linewidth=2,
        #         label=r'$Vehicle$',
        #         alpha=0.9)
        #     axs.plot3D(
        #         ndx,
        #         ndy,
        #         _DRONE_ALTITUDE,
        #         color='darkslategray',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$UAS$',
        #         alpha=0.9)

        #     for point in zip(ndx, ndy, _DRONE_ALTITUDE):
        #         x = [point[0], point[0]]
        #         y = [point[1], point[1]]
        #         z = [point[2], 0]
        #         axs.plot3D(x, y, z, color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.1)
        #     axs.plot3D(ndx, ndy, 0, color='silver', linestyle='-', linewidth=1, alpha=0.9)
        #     axs.scatter3D(ndx, ndy, _DRONE_ALTITUDE, c=_DRONE_ALTITUDE, cmap='plasma', alpha=0.3)

        #     axs.set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$', zlabel=r'$z\ (m)$')
        #     axs.view_init(elev=41, azim=-105)
        #     # axs.view_init(elev=47, azim=-47)
        #     axs.set_title(r'$\mathbf{World\ frame}$', fontsize=SUB_TITLE_FONT_SIZE)
        #     axs.legend()

        #     f8.savefig(f'{_PATH}/9_3D_traj.png', dpi=300)
        #     f8.show()

        # # -------------------------------------------------------------------------------- figure 7
        # # delta time
        # if SHOW_ALL or SHOW_DELTA_TIME_PROFILE:
        #     f9, axs = plt.subplots(2, 1, gridspec_kw={'hspace': 0.4})
        #     if SUPTITLE_ON:
        #         f9.suptitle(r'$\mathbf{Time\ Delay\ profile}$', fontsize=TITLE_FONT_SIZE)
        #     axs[0].plot(
        #         _TIME,
        #         _DELTA_TIME,
        #         color='darksalmon',
        #         linestyle='-',
        #         linewidth=2,
        #         label=r'$\Delta\ t$',
        #         alpha=0.9)
        #     axs[0].set(xlabel=r'$time\ (s)$', ylabel=r'$\Delta t\ (s)$')

        #     _NUM_BINS = 300
        #     _DIFF = max(_DELTA_TIME) - min(_DELTA_TIME)
        #     _BAR_WIDTH = _DIFF/_NUM_BINS if USE_REAL_CLOCK else DELTA_TIME * 0.1
        #     _RANGE = (min(_DELTA_TIME), max(_DELTA_TIME)) if USE_REAL_CLOCK else (-2*abs(DELTA_TIME), 4*abs(DELTA_TIME))
        #     _HIST = np.histogram(_DELTA_TIME, bins=_NUM_BINS, range=_RANGE, density=1) if USE_REAL_CLOCK else np.histogram(_DELTA_TIME, bins=_NUM_BINS, density=1)
        #     axs[1].bar(_HIST[1][:-1], _HIST[0]/sum(_HIST[0]), width=_BAR_WIDTH*0.9, 
        #                 color='lightsteelblue', label=r'$Frequentist\ PMF\ distribution$', alpha=0.9)
        #     if not USE_REAL_CLOCK:
        #         axs[1].set_xlim(-2*abs(DELTA_TIME), 4*abs(DELTA_TIME))
            
        #     if USE_REAL_CLOCK:
        #         _MIN, _MAX = axs[1].get_xlim()
        #         axs[1].set_xlim(_MIN, _MAX)
        #         _KDE_X = np.linspace(_MIN, _MAX, 301)
        #         _GAUSS_KER = st.gaussian_kde(_DELTA_TIME)
        #         _PDF_DELTA_T = _GAUSS_KER.pdf(_KDE_X)
        #         axs[1].plot(_KDE_X, _PDF_DELTA_T/sum(_PDF_DELTA_T), color='royalblue', linestyle='-',
        #                     linewidth=2, label=r'$Gaussian\ Kernel\ Estimate\ PDF$', alpha=0.8)
        #     axs[1].set(ylabel=r'$Probabilities$', xlabel=r'$\Delta t\ values$')
        #     axs[1].legend(loc='upper left')

        #     f9.savefig(f'{_PATH}/9_delta_time.png', dpi=300)

        #     f9.show()

        # # -------------------------------------------------------------------------------- figure 7
        # # y1, y2
        # if SHOW_ALL or SHOW_Y1_Y2:
        #     f10, axs = plt.subplots(2, 1, gridspec_kw={'hspace': 0.4})
        #     if SUPTITLE_ON:
        #         f10.suptitle(r'$\mathbf{Objectives}$', fontsize=TITLE_FONT_SIZE)
        #     axs[0].plot(
        #         _TIME,
        #         _TRUE_Y1,
        #         color='red',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$true\ y_1$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         _Y1,
        #         color='royalblue',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$estimated\ y_1$',
        #         alpha=0.9)
        #     axs[0].plot(
        #         _TIME,
        #         [K_W for _ in _TIME],
        #         color='orange',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$w_1$',
        #         alpha=0.9)
        #     axs[0].legend(loc='upper right')
        #     axs[0].set(ylabel=r'$y_1$')
        #     axs[0].set_title(r'$\mathbf{y_1}$', fontsize=SUB_TITLE_FONT_SIZE)

        #     axs[1].plot(
        #         _TIME,
        #         _TRUE_Y2,
        #         color='red',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$true\ y_2$',
        #         alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         _Y2,
        #         color='royalblue',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$estimated\ y_2$',
        #         alpha=0.9)
        #     axs[1].plot(
        #         _TIME,
        #         [0.0 for _ in _TIME],
        #         color='orange',
        #         linestyle='-',
        #         linewidth=LINE_WIDTH_1,
        #         label=r'$w_2$',
        #         alpha=0.9)

        #     axs[1].legend(loc='upper right')
        #     axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$y_2$')
        #     axs[1].set_title(r'$\mathbf{y_2}$', fontsize=SUB_TITLE_FONT_SIZE)


        #     f10.savefig(f'{_PATH}/10_objectives.png', dpi=300)

        #     f10.show()
        # plt.show()

    if RUN_VIDEO_WRITER:
        EXPERIMENT_MANAGER = ExperimentManager()
        # create folder path inside ./sim_outputs
        _PATH = f'./sim_outputs/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        _prep_temp_folder(os.path.realpath(_PATH))
        VID_PATH = f'{_PATH}/sim_track_control.avi'
        print('Making video.')
        EXPERIMENT_MANAGER.make_video(VID_PATH, SIMULATOR_TEMP_FOLDER)

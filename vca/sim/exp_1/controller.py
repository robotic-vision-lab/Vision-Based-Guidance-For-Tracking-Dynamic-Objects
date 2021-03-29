import os
import numpy as np
from datetime import timedelta
from math import atan2, degrees, cos, sin, pi
from settings import *

class Controller:
    def __init__(self, manager):
        self.manager = manager
        self.plot_info_file = 'plot_info.csv'
        self.R = CAR_RADIUS
        self.f = None
        self.a_ln = 0.0
        self.a_lt = 0.0
        # self.est_def = False


    @staticmethod
    def sat(x, bound):
        return min(max(x, -bound), bound)

    def generate_acceleration(self, kin):
        # unpack kinematics of UAS and vehicle
        drone_pos_x, drone_pos_y = kin[0]
        drone_vel_x, drone_vel_y = kin[1]
        car_pos_x, car_pos_y = kin[2]
        car_vel_x, car_vel_y = kin[3]

        # convert kinematics to inertial frame
        cam_origin_x, cam_origin_y = self.manager.get_cam_origin()
        # positions translated by camera origin
        drone_pos_x = cam_origin_x
        drone_pos_y = cam_origin_y
        car_pos_x += cam_origin_x
        car_pos_y += cam_origin_y

        # compute speeds of drone and car
        drone_speed = (drone_vel_x**2 + drone_vel_y**2)**0.5
        car_speed = (car_vel_x**2 + car_vel_y**2)**0.5

        # heading angle of drone
        drone_alpha = atan2(drone_vel_y, drone_vel_x)

        # heading angle of car
        car_beta = atan2(car_vel_y, car_vel_x)

        # distance between the drone and car
        r = ((car_pos_x - drone_pos_x)**2 + (car_pos_y - drone_pos_y)**2)**0.5

        # angle of LOS from drone to car
        theta = atan2(car_pos_y - drone_pos_y, car_pos_x - drone_pos_x)

        # compute Vr and Vθ
        Vr = car_speed*cos(car_beta - theta) - drone_speed*cos(drone_alpha - theta)
        Vtheta = car_speed*sin(car_beta - theta) - drone_speed*sin(drone_alpha - theta)

        # save measured as r_m, θ_m, Vr_m, Vθ_m
        r_m = r
        theta_m = theta
        Vr_m = Vr
        Vtheta_m = Vtheta

        if not CLEAN_CONSOLE:
            print(f'CCCm >> r_m:{r_m:0.2f} | theta_m:{theta_m:0.2f} | alpha_m:{drone_alpha:0.2f} | beta_m:{car_beta:0.2f} | car_vel_x:{car_vel_x:0.2f} | car_vel_y:{car_vel_y:0.2f} | drone_speed:{drone_speed:0.2f} | Vr_m:{Vr_m:0.2f} | Vtheta_m:{Vtheta_m:0.2f} ')

        # this point on r, θ, Vr, Vθ etc are estimated ones

        # EKF filtering [r, theta, Vr, Vtheta] 
        if not self.manager.use_true_kin and (USE_EXTENDED_KALMAN or USE_NEW_EKF):
            if USE_NEW_EKF:
                self.manager.EKF.add(r, theta, Vr, Vtheta, drone_alpha, self.a_lt, self.a_ln, car_pos_x, car_pos_y, car_vel_x, car_vel_y)
                r, theta, Vr, Vtheta, deltaB_est, estimated_acceleration = self.manager.EKF.get_estimated_state()
            else: # the old EKF may not be used anymore, here only for backward compat
                self.manager.EKF.add(r, theta, Vr, Vtheta, drone_alpha, self.a_lt, self.a_ln)
                r, theta, Vr, Vtheta = self.manager.EKF.get_estimated_state()

        # calculate y from drone to car
        y2 = Vtheta**2 + Vr**2
        y1 = r**2 * Vtheta**2 - y2 * self.R**2


        # compute desired acceleration
        w = K_W
        K1 = K_1 * np.sign(-Vr)    # lat
        K2 = K_2                   # long

        # compute lat and long accelerations
        _D = 2 * Vr * Vtheta * r**2


        if abs(_D) < 0.01:
            a_lat = 0.0
            a_long = 0.0
        else:
            if self.manager.use_true_kin:
                car_acc_x, car_acc_y = self.manager.simulator.car.acceleration
                estimated_acceleration = (car_acc_x**2 + car_acc_y**2)**0.5
                deltaB_est = atan2(car_acc_y, car_acc_x)
                # estimated_acceleration = 0.0
                # deltaB_est = 0.0

            # a_lat = (K1 * Vr * y1 * cos(drone_alpha - theta) - K1 * Vr * w * cos(drone_alpha - theta) - K1 * Vtheta * w * sin(drone_alpha - theta) + K1 * Vtheta * y1 * sin(drone_alpha - theta)
            #             - 2*Vr*Vtheta*estimated_acceleration*r**2*sin(drone_alpha - deltaB_est) +
            #             K2 * self.R**2 * Vr * y2 * cos(drone_alpha - theta) + K2 * self.R**2 * Vtheta * y2 * sin(drone_alpha - theta) - K2 * Vtheta * r**2 * y2 * sin(drone_alpha - theta)) / _D
            # a_long = (K1 * Vtheta * w * cos(drone_alpha - theta) - K1 * Vtheta * y1 * cos(drone_alpha - theta) - K1 * Vr * w * sin(drone_alpha - theta) + K1 * Vr * y1 * sin(drone_alpha - theta)
            #             + 2*Vr*Vtheta*estimated_acceleration*r**2*cos(drone_alpha - deltaB_est) -
            #             K2 * self.R**2 * Vtheta * y2 * cos(drone_alpha - theta) + K2 * self.R**2 * Vr * y2 * sin(drone_alpha - theta) + K2 * Vtheta * r**2 * y2 * cos(drone_alpha - theta)) / _D
            K_a = 1
            dax = 0 #self.manager.simulator.camera.acceleration[0]
            day = 0 #self.manager.simulator.camera.acceleration[1]
            a_lat = ((K1 * (y1-w) * (Vr * cos(drone_alpha - theta) + Vtheta * sin(drone_alpha - theta))
                        + K2 * y2 *	( self.R**2 * Vr * cos(drone_alpha - theta) - (r**2 - self.R**2) * Vtheta * sin(drone_alpha - theta) )
                        ) / _D 
                        - K_a*(-dax+estimated_acceleration)*sin(drone_alpha - deltaB_est))

            # print(f'est_acc: {estimated_acceleration}, drone_alpha: {degrees(drone_alpha)}, deltaB_est: {degrees(deltaB_est)}, sin: {sin(drone_alpha - theta)}, cos: {cos(drone_alpha - theta)}')
            a_long = ((K1 * (y1-w) * ( Vr * sin(drone_alpha - theta) - Vtheta * cos(drone_alpha - theta) )
                        + K2 * y2 *	( self.R**2 * Vr * sin(drone_alpha - theta) + (r**2 - self.R**2) * Vtheta * cos(drone_alpha - theta) )
                        ) / _D
                        + K_a*(-day+estimated_acceleration)*cos(drone_alpha - deltaB_est))
            

        a_long_bound = 10
        a_lat_bound = 10

        a_long = self.sat(a_long, a_long_bound)
        a_lat = self.sat(a_lat, a_lat_bound)

        self.a_ln = a_long
        self.a_lt = a_lat

        # compute acceleration command
        delta = drone_alpha + pi / 2
        ax = a_lat * cos(delta) + a_long * cos(drone_alpha)
        ay = a_lat * sin(delta) + a_long * sin(drone_alpha)

        if not CLEAN_CONSOLE:
            print(f'CCC0 >> r:{r:0.2f} | theta:{theta:0.2f} | alpha:{drone_alpha:0.2f} | car_vel_x:{car_vel_x:0.2f} | car_vel_y:{car_vel_y:0.2f} | drone_speed:{drone_speed:0.2f} | Vr:{Vr:0.2f} | Vtheta:{Vtheta:0.2f} | y1:{y1:0.2f} | y2:{y2:0.2f} | a_lat:{a_lat:0.2f} | a_long:{a_long:0.2f} | _D:{_D:0.2f}')

        tru_kin = self.manager.get_true_kinematics()
        tX, tY = tru_kin[0]
        tVx, tVy = tru_kin[1]
        tcar_x, tcar_y = tru_kin[2]
        tcar_vel_x, tcar_vel_y = tru_kin[3]
        car_S = (tcar_vel_x**2 + tcar_vel_y**2)**0.5
        tS = (tVx**2 + tVy**2) ** 0.5
        tr = ((tcar_x - tX)**2 + (tcar_y - tY)**2)**0.5
        ttheta = atan2(tcar_y - tY, tcar_x - tX)
        tbeta = atan2(tcar_vel_y, tcar_vel_x)
        tVr = car_S * cos(tbeta - ttheta) - tS * cos(drone_alpha - ttheta)
        tVtheta = car_S * sin(tbeta - ttheta) - tS * sin(drone_alpha - ttheta)
        car_head = atan2(tcar_vel_y, tcar_vel_x)
        ty2 = tVtheta**2 + tVr**2
        ty1 = tr**2 * tVtheta**2 - ty2 * self.R**2
        
        occ_case = self.manager.tracker.target_occlusion_case_new

        tra_kin = self.manager.get_tracked_kinematics()
        # vel = self.manager.simulator.camera.velocity
        if not CLEAN_CONSOLE:
            print(
                f'CCCC >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:[{drone_pos_x:0.2f}, {drone_pos_y:0.2f}] | v:[{drone_vel_x:0.2f}, {drone_vel_y:0.2f}] | CAR - x:[{car_pos_x:0.2f}, {car_pos_y:0.2f}] | v:[{car_vel_x:0.2f}, {car_vel_y:0.2f}] | COMMANDED a:[{ax:0.2f}, {ay:0.2f}] | TRACKED x:[{tra_kin[2][0]:0.2f},{tra_kin[2][1]:0.2f}] | v:[{tra_kin[3][0]:0.2f},{tra_kin[3][1]:0.2f}]')
        if self.manager.write_plot:
            self.f.write(
                f'{self.manager.simulator.time},' +                 # _TIME
                f'{r},' +                                           # _R (est)
                f'{degrees(theta)},' +                              # _THETA (est)
                f'{Vtheta},' +                                      # _V_THETA (est)
                f'{Vr},' +                                          # _V_R (est)
                f'{tru_kin[0][0]},' +                               # _DRONE_POS_X
                f'{tru_kin[0][1]},' +                               # _DRONE_POS_Y
                f'{tru_kin[2][0]},' +                               # _CAR_POS_X
                f'{tru_kin[2][1]},' +                               # _CAR_POS_Y
                f'{ax},' +                                          # _DRONE_ACC_X
                f'{ay},' +                                          # _DRONE_ACC_Y
                f'{a_lat},' +                                       # _DRONE_ACC_LAT
                f'{a_long},' +                                      # _DRONE_ACC_LNG
                f'{tru_kin[3][0]},' +                               # _CAR_VEL_X
                f'{tru_kin[3][1]},' +                               # _CAR_VEL_Y
                f'{tra_kin[2][0]},' +                               # _TRACKED_CAR_POS_X
                f'{tra_kin[2][1]},' +                               # _TRACKED_CAR_POS_Y
                f'{tra_kin[3][0]},' +                               # _TRACKED_CAR_VEL_X
                f'{tra_kin[3][1]},' +                               # _TRACKED_CAR_VEL_Y
                f'{self.manager.simulator.camera.origin[0]},' +     # _CAM_ORIGIN_X
                f'{self.manager.simulator.camera.origin[1]},' +     # _CAM_ORIGIN_Y
                f'{drone_speed},' +                                 # _DRONE_SPEED
                f'{degrees(drone_alpha)},' +                        # _DRONE_ALPHA
                f'{tru_kin[1][0]},' +                               # _DRONE_VEL_X
                f'{tru_kin[1][1]},' +                               # _DRONE_VEL_Y
                f'{tra_kin[2][0]},' +                               # _MEASURED_CAR_POS_X
                f'{tra_kin[2][1]},' +                               # _MEASURED_CAR_POS_Y
                f'{tra_kin[3][0]},' +                               # _MEASURED_CAR_VEL_X
                f'{tra_kin[3][1]},' +                               # _MEASURED_CAR_VEL_Y
                f'{self.manager.simulator.camera.altitude},' +      # _DRONE_ALTITUDE
                f'{abs(_D)},' +                                     # _ABS_DEN
                f'{r_m},' +                                         # _MEASURED_R
                f'{degrees(theta_m)},' +                            # _MEASURED_THETA
                f'{Vr_m},' +                                        # _MEASURED_V_R
                f'{Vtheta_m},' +                                    # _MEASURED_V_THETA
                f'{tr},' +                                          # _TRUE_R
                f'{degrees(ttheta)},' +                             # _TRUE_THETA
                f'{tVr},' +                                         # _TRUE_V_R
                f'{tVtheta},' +                                     # _TRUE_V_THETA
                f'{self.manager.simulator.dt},' +                   # _DELTA_TIME
                f'{y1},' +                                          # _Y1 (est)
                f'{y2},' +                                          # _Y2 (est)
                f'{car_S},' +                                       # _CAR_SPEED
                f'{degrees(car_head)},' +                           # _CAR_HEADING
                f'{ty1},' +                                         # _TRUE_Y1
                f'{ty2},' +                                          # _TRUE_Y2
                f'{occ_case}\n')                                    # _OCC_CASE

        if not self.manager.control_on:
            ax, ay = pygame.Vector2((0.0, 0.0))

        return ax, ay

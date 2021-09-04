import os
import numpy as np
from datetime import timedelta
from math import atan2, degrees, cos, sin, pi, pow
from .settings import *
from .my_imports import bf, rb, mb, gb, yb, bb, cb,  r, m, g, y, b, c, colored, cprint

class Controller:
    def __init__(self, manager):
        self.manager = manager
        self.plot_info_file = 'plot_info.csv'
        self.stored_data = None
        self.R = CAR_RADIUS
        self.f = None

        self.e_s_prev = 0.0
        self.e_c_prev = 0.0
        self.e_z_prev = 0.0
        self.e_s_sum = 0.0
        self.e_c_sum = 0.0
        self.e_z_sum = 0.0

        self.scz_ind_prev = 0

    @staticmethod
    def sat(x, bound):
        return min(max(x, -bound), bound)

    def generate_acceleration_(self, kin):
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
        if not self.manager.use_true_kin:
            self.manager.EKF.add(r, theta, Vr, Vtheta, drone_alpha, car_pos_x, car_pos_y, car_vel_x, car_vel_y)
            r, theta, Vr, Vtheta, deltaB_est, estimated_acceleration = self.manager.EKF.get_estimated_state()

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

    def generate_acceleration(self, ellipse_focal_points_est_state, ellipse_major_axis_len):
        """Uses esitmated state of ellipse focal points and major axis to generate lateral and logintudinal accleration commands for dronecamera.
        Returns ax, ay

        Args:
            ellipse_focal_points_est_state (tuple): State estimations for both focal points
            ellipse_major_axis_len (float): Major axis of enclosing ellipse

        Returns:
            tuple : ax, ay
        """
        cam_origin_x, cam_origin_y = self.manager.get_cam_origin()

        # drone (known)
        drone_pos_x, drone_pos_y = self.manager.get_true_drone_position()
        drone_vel_x, drone_vel_y = self.manager.get_true_drone_velocity()
        drone_pos_x += cam_origin_x
        drone_pos_y += cam_origin_y

        drone_speed = (drone_vel_x**2 + drone_vel_y**2)**0.5
        drone_alpha = atan2(drone_vel_y, drone_vel_x)

        # collect estimated focal point state 
        fp1_x, fp1_vx, fp1_ax, fp1_y, fp1_vy, fp1_ay, fp2_x, fp2_vx, fp2_ax, fp2_y, fp2_vy, fp2_ay = ellipse_focal_points_est_state

        # compute r and θ for both focal points
        r1 = ((fp1_x - drone_pos_x)**2 + (fp1_y - drone_pos_y)**2)**0.5
        r2 = ((fp2_x - drone_pos_x)**2 + (fp2_y - drone_pos_y)**2)**0.5

        theta1 = atan2(fp1_y - drone_pos_y, fp1_x - drone_pos_x)
        theta2 = atan2(fp2_y - drone_pos_y, fp2_x - drone_pos_x)

        # compute focal points speed and heading for both focal points
        fp1_speed = (fp1_vx**2 + fp1_vy**2)**0.5
        fp2_speed = (fp2_vx**2 + fp2_vy**2)**0.5

        fp1_heading = atan2(fp1_vy, fp1_vx)
        fp2_heading = atan2(fp2_vy, fp2_vx)

        # compute acceleration magnitude and direction for both focal points
        fp1_acc = (fp1_ax**2 + fp1_ay**2)**0.5
        fp2_acc = (fp2_ax**2 + fp2_ay**2)**0.5

        fp1_delta = atan2(fp1_ay, fp1_ax)
        fp2_delta = atan2(fp2_ay, fp2_ax)

        # compute Vr and Vθ for both focal points
        Vr1 = fp1_speed*cos(fp1_heading - theta1) - drone_speed*cos(drone_alpha - theta1)
        Vr2 = fp2_speed*cos(fp2_heading - theta2) - drone_speed*cos(drone_alpha - theta2)

        Vtheta1 = fp1_speed*sin(fp1_heading - theta1) - drone_speed*sin(drone_alpha - theta1)
        Vtheta2 = fp2_speed*sin(fp2_heading - theta2) - drone_speed*sin(drone_alpha - theta2)

        # compute objective function
        y1, y2 = self.compute_objective_functions(r1, r2, Vr1, Vr2, Vtheta1, Vtheta2, ellipse_major_axis_len)

        # compute objective function derivatives
        dy1dVr1, dy1dVtheta1, dy1dVr2, dy1dVtheta2, dy2dVr1, dy2dVtheta1 = self.compute_y1_y2_derivative(r1, r2, Vr1, Vr2, Vtheta1, Vtheta2)

        # set gains
        K1 = K_1 * np.sign(-Vr1)
        K2 = K_2
        w = K_W


        # compute acceleration commands
        # precompute variables for numeric efficiency
        f1dt1 = fp1_delta - theta1
        f2dt2 = fp2_delta - theta2
        dalt1 = drone_alpha - theta1
        dalt2 = drone_alpha - theta2
        dalf1d = drone_alpha-fp1_delta
        t1t2 = theta1 - theta2
        dalf1dmt1t2 = drone_alpha+fp1_delta-theta1-theta2
        dalf2dmt1t2 = drone_alpha+fp2_delta-theta1-theta2
        dalmf1dt1mt2 = drone_alpha-fp1_delta+theta1-theta2
        dalmf2dmt1t2 = drone_alpha-fp2_delta-theta1+theta2
        cf1dt1 = cos(f1dt1)
        sf1dt1 = sin(f1dt1)
        cf2dt2 = cos(f2dt2)
        sf2dt2 = sin(f2dt2)
        cdalt1 = cos(dalt1)
        sdalt1 = sin(dalt1)
        cdalt2 = cos(dalt2)
        sdalt2 = sin(dalt2)
        ct1t2 = cos(t1t2)
        st1t2 = sin(t1t2)
        cdalf1d = cos(dalf1d)
        cdalf1dmt1t2 = cos(dalf1dmt1t2)
        sdalf1dmt1t2 = sin(dalf1dmt1t2)
        cdalf2dmt1t2 = cos(dalf2dmt1t2)
        sdalf2dmt1t2 = sin(dalf2dmt1t2)
        cdalmf1dt1mt2 = cos(dalmf1dt1mt2)
        sdalmf1dt1mt2 = sin(dalmf1dt1mt2)
        cdalmf2dmt1t2 = cos(dalmf2dmt1t2)
        sdalmf2dmt1t2 = sin(dalmf2dmt1t2)
        denom_sub = dy1dVtheta1*dy2dVr1-dy1dVr1*dy2dVtheta1
        y1vt1dy2vr1 = dy1dVtheta2*dy2dVr1
        y1vr2dy2vt1 = dy1dVr2*dy2dVtheta1
        y1vr2dy2vr1 = dy1dVr2*dy2dVr1
        y1vt2dy2vt1 = dy1dVtheta2*dy2dVtheta1
        denom = (denom_sub+(y1vt1dy2vr1-y1vr2dy2vt1)*ct1t2-(y1vr2dy2vr1+y1vt2dy2vt1)*st1t2)

        a_lat = -(
            (K2*y2+fp1_acc*dy2dVr1*cf1dt1+fp1_acc*dy2dVtheta1*sf1dt1)
            *(dy1dVr1*cdalt1+dy1dVr2*cdalt2+dy1dVtheta1*sdalt1+dy1dVtheta2*sdalt2)
            -(dy2dVr1*cdalt1+dy2dVtheta1*sdalt1)
            *(-K1*w+K1*y1+fp1_acc*dy1dVr1*cf1dt1+fp2_acc*dy1dVr2*cf2dt2+fp1_acc*dy1dVtheta1*sf1dt1+fp2_acc*dy1dVtheta2*sf2dt2)
            ) / denom

        a_long = (1/2)*(
            2*fp1_acc*(denom_sub)*cdalf1d
            +2*(dy2dVtheta1*K1*(w-y1)+dy1dVtheta1*K2*y2)*cdalt1
            +2*dy1dVtheta2*K2*y2*cdalt2
            -2*dy2dVr1*K1*w*sdalt1
            +2*dy2dVr1*K1*y1*sdalt1
            -2*dy1dVr1*K2*y2*sdalt1
            -2*dy1dVr2*K2*y2*sdalt2
            +y1vt1dy2vr1*fp1_acc*cdalf1dmt1t2
            +y1vr2dy2vt1*fp1_acc*cdalf1dmt1t2
            +y1vt1dy2vr1*fp1_acc*cdalmf1dt1mt2
            -y1vr2dy2vt1*fp1_acc*cdalmf1dt1mt2
            -y1vr2dy2vr1*fp1_acc*sdalf1dmt1t2
            +y1vt2dy2vt1*fp1_acc*sdalf1dmt1t2
            -y1vr2dy2vr1*fp1_acc*sdalmf1dt1mt2
            -y1vt2dy2vt1*fp1_acc*sdalmf1dt1mt2
            -y1vt1dy2vr1*fp2_acc*cdalf2dmt1t2
            -y1vr2dy2vt1*fp2_acc*cdalf2dmt1t2
            +y1vt1dy2vr1*fp2_acc*cdalmf2dmt1t2
            -y1vr2dy2vt1*fp2_acc*cdalmf2dmt1t2
            +y1vr2dy2vr1*fp2_acc*sdalf2dmt1t2
            -y1vt2dy2vt1*fp2_acc*sdalf2dmt1t2
            +y1vr2dy2vr1*fp2_acc*sdalmf2dmt1t2
            +y1vt2dy2vt1*fp2_acc*sdalmf2dmt1t2
            ) / denom

        # clip acceleration commands
        a_long_bound = 10
        a_lat_bound = 10

        a_long = self.sat(a_long, a_long_bound)
        a_lat = self.sat(a_lat, a_lat_bound)

        # compute acceleration commands ax and ay
        delta = drone_alpha + pi / 2
        ax = a_lat * cos(delta) + a_long * cos(drone_alpha)
        ay = a_lat * sin(delta) + a_long * sin(drone_alpha)

        # compute az
        x_min, y_min = self.manager.tracking_manager.p1
        x_max, y_max = self.manager.tracking_manager.p2
        X = x_max - x_min
        Y = y_max - y_min

        S = np.linalg.norm((X,Y))
        # C = (((WIDTH - x_min - x_max)**2 + (HEIGHT - y_min - y_max)**2)**0.5)/2
        C = abs((((WIDTH - x_min - x_max) + (HEIGHT - y_min - y_max)))/2)
        Z_W = self.manager.simulator.camera.altitude

        S_W = S*self.manager.simulator.pxm_fac
        C_W = C*self.manager.simulator.pxm_fac

        self.manager.tracking_manager.bounding_area_EKF.add(S, C, Z_W)
        S, C, Z_W, S_dot, C_dot, Z_W_dot = self.manager.tracking_manager.bounding_area_EKF.get_estimated_state()

        KP_s = 150
        KP_c = 300
        KP_z = 200

        KD_s = 15
        KD_c = 10
        KD_z = 10
        
        KI_s = 0.5
        KI_c = 3
        KI_z = 0.5

        # X_d = WIDTH*0.3
        # Y_d = WIDTH*0.3
        S_d = S_DES
        C_d = C_DES
        Z_d = Z_DES

        e_s = S_d - S
        e_c = min(0, C_d - C)
        e_Z_W = Z_d - Z_W if abs(Z_d - Z_W) > Z_DELTA else 0.0

        vz = self.manager.simulator.camera.vz
        vz_2 = vz**2 * np.sign(vz)

        
        FS = ((FOCAL_LENGTH * S_W) / S**2)
        FC = ((FOCAL_LENGTH * C_W) / C**2)

        az_s = -FS * KP_s * (e_s) + FS * KD_s * S_dot + 2 * FS * S_dot**2 / S 
        az_c = -FC * KP_c * (e_c) + FC * KD_c * C_dot + 2 * FC * C_dot**2 / C #- FC * KI_c * self.e_c_sum
        az_z = KP_z * e_Z_W - KD_z * Z_W_dot if not e_Z_W==0.0 else 0.0

        

        self.e_s_prev = e_s
        self.e_c_prev = e_c
        self.e_z_prev = e_Z_W
        self.e_s_sum += e_s
        self.e_c_sum += e_c
        self.e_z_sum += e_Z_W

        a = np.array([az_s, az_c, az_z])
        scz_ind = np.argmax(abs(a))
        az = a[scz_ind]
        if not self.scz_ind_prev==scz_ind:
            if scz_ind == 0:
                self.e_s_sum = 0.0
                # pass
            elif scz_ind == 1:
                self.e_c_sum = 0.0
                # pass
            else:
                self.e_z_sum = 0.0

        self.scz_ind_prev = scz_ind

        # az = az_x + az_y + 0

        az = self.sat(az, 10)

        print(f'{g("            SCZ_des-")}{gb(f"[{S_d:.2f}, {C_d:.2f}, {Z_d:.2f}]")}{g(", SCZ_meas-")}{gb(f"[{S:.2f}, {C:.2f}, {Z_W:.2f}]")}{g(", vz=")}{gb(f"{vz:.2f}")}', end='')
        print(f'{g(", az_s=")}{gb(f"{az_s:.4f}")}', end=' ')
        print(f'{g("+ az_c=")}{gb(f"{az_c:.4f}")}', end=' ')
        print(f'{g("+ az_z=")}{gb(f"{az_z:.4f} ")}{g("=> comm_az=")}{gb(f"{az:.4f}")}')

        if self.manager.write_plot:
            # store vairables if manager needs to write to file
            self.stored_data = np.array([
                fp1_x,          #  0 focal point 1 position x component
                fp1_y,          #  1 focal point 1 position y component
                fp1_vx,         #  2 focal point 1 velocity x component
                fp1_vy,         #  3 focal point 1 velocity y component
                fp1_ax,         #  4 focal point 1 acceleration x component
                fp1_ay,         #  5 focal point 1 acceleration y component
                r1,             #  6 focal point 1 r (LOS)
                theta1,         #  7 focal point 1 theta (LOS)
                Vr1,            #  8 focal point 1 Vr (LOS)
                Vtheta1,        #  9 focal point 1 Vtheta (LOS)
                fp1_speed,      # 10 focal point 1 speed
                fp1_heading,    # 11 focal point 1 heading
                fp1_acc,        # 12 focal point 1 acceleration magnitude
                fp1_delta,      # 13 focal point 1 acceleration angle (delta)
                fp2_x,          # 14 focal point 2 position x component
                fp2_y,          # 15 focal point 2 position y component
                fp2_vx,         # 16 focal point 2 velocity x component
                fp2_vy,         # 17 focal point 2 velocity y component
                fp2_ax,         # 18 focal point 2 acceleration x component
                fp2_ay,         # 19 focal point 2 acceleration y component
                r2,             # 20 focal point 2 r (LOS)
                theta2,         # 21 focal point 2 theta (LOS)
                Vr2,            # 22 focal point 2 Vr (LOS)
                Vtheta2,        # 23 focal point 2 Vtheta (LOS)
                fp2_speed,      # 24 focal point 2 speed
                fp2_heading,    # 25 focal point 2 heading
                fp2_acc,        # 26 focal point 2 acceleration magnitude
                fp2_delta,      # 27 focal point 2 acceleration angle (delta)
                y1,             # 28 objective function y1
                y2,             # 29 objective function y2
                a_lat,          # 30 commanded acceleration a_lat
                a_long,         # 31 commanded acceleration a_long
                S,              # 32 size of control area
                C,              # 33 distance of control area
                Z_W,            # 34 drone altitude 
                S_dot,          # 35 rate of change of size of control area
                C_dot,          # 36 rate of change of distance of control area
                Z_W_dot,        # 37 rate of change of drone altitude 
                az_s,           # 38 commanded acceleration on acount of S
                az_c,           # 39 commanded acceleration on acount of C
                az_z,           # 40 commanded acceleration on acount of Z_W
                az              # 41 aggregated commanded accleration az for altitude control
            ])

        return ax, ay, az



    @staticmethod
    def compute_objective_functions(r1, r2, Vr1, Vr2, Vtheta1, Vtheta2, a):
        V1 = pow((Vtheta1**2 + Vr1**2),0.5)
        V2 = pow((Vtheta2**2 + Vr2**2),0.5)
        A1 = r1*Vtheta1/V1
        A2 = r2*Vtheta2/V2
        tau_num = r1*Vr1/V1**2 - r2*Vr2/V2**2
        tau_den = A1+A2
        tau = (tau_num/tau_den)**2

        y1 = A1**2*(1+tau*V1**2) + A2**2*(1+tau*V2**2) + 2*A1*A2*pow((1+tau*(V1**2+V2**2)+tau**2*V1**2*V2**2),0.5)-4*(a)**2

        y2 = Vtheta1**2 + Vr1**2

        return y1, y2

    @staticmethod
    def compute_y1_y2_derivative(r1, r2, Vr1, Vr2, Vtheta1, Vtheta2):
        def sqrt(x):
            return pow(x, 0.5)

        # Compute the variables needed for derivatives
        V1 = sqrt(Vtheta1**2+Vr1**2); V2 = sqrt(Vtheta2**2+Vr2**2) 
        A1 = r1*Vtheta1/V1 
        A2 = r2*Vtheta2/V2
        tau = (r1*Vr1/V1**2 - r2*Vr2/V2**2)**2/(A1+A2)**2

        # Compute the derivatives needed for chain rule
        dy1dA1 = 2*A1*(1+tau*V1**2)+2*A2*(1+tau**2*V1**2*V2**2+tau*(V1**2+V2**2))**(1/2)
        dA1dVr1 = (-1)*r1*Vr1*Vtheta1*(Vr1**2+Vtheta1**2)**(-3/2)
        dy1dtau = A1**2*V1**2+A2**2*V2**2+A1*A2*(V1**2+V2**2+2*tau*V1**2*V2**2)*(1+tau**2*V1**2*V2**2+tau*(V1**2+V2**2))**(-1/2)
        dy1dV1 = 2*A1**2*tau*V1+A1*A2*(2*tau*V1+2*tau**2*V1*V2**2)*(1+tau**2*V1**2*V2**2+tau*(V1**2+V2**2))**(-1/2)
        dV1dVr1 = Vr1*(Vr1**2+Vtheta1**2)**(-1/2)
        dy1dA2 = 2*A2*(1+tau*V2**2)+2*A1*(1+tau**2*V1**2*V2**2+tau*(V1**2+V2**2))**(1/2)
        dA2dVr2 = (-1)*r2*Vr2*Vtheta2*(Vr2**2+Vtheta2**2)**(-3/2)
        dy1dV2 = 2*A2**2*tau*V2+A1*A2*(2*tau*V2+2*tau**2*V1**2*V2)*(1+tau**2*V1**2*V2**2+tau*(V1**2+V2**2))**(-1/2)
        dV2dVr2 = Vr2*(Vr2**2+Vtheta2**2)**(-1/2)
        dA2dVtheta2 = (-1)*r2*Vtheta2**2*(Vr2**2+Vtheta2**2)**(-3/2)+r2*(Vr2**2+Vtheta2**2)**(-1/2)
        dV2dVtheta2 = Vtheta2*(Vr2**2+Vtheta2**2)**(-1/2)
        dA1dVtheta1 = (-1)*r1*Vtheta1**2*(Vr1**2+Vtheta1**2)**(-3/2)+r1*(Vr1**2+Vtheta1**2)**(-1/2)
        dV1dVtheta1 = Vtheta1*(Vr1**2+Vtheta1**2)**(-1/2)
        dtaudV1 = (-4)*(A1+A2)**(-2)*r1*V1**(-3)*Vr1*(r1*V1**(-2)*Vr1+(-1)*r2*V2**(-2)*Vr2)
        dtaudV2 = 4*(A1+A2)**(-2)*r2*V2**(-3)*Vr2*(r1*V1**(-2)*Vr1+(-1)*r2*V2**(-2)*Vr2)
        dtaudA1 = (-2)*(A1+A2)**(-3)*(r1*V1**(-2)*Vr1+(-1)*r2*V2**(-2)*Vr2)**2
        dtaudA2 = (-2)*(A1+A2)**(-3)*(r1*V1**(-2)*Vr1+(-1)*r2*V2**(-2)*Vr2)**2
        dtaudVr1E = 2*(A1+A2)**(-2)*r1*V1**(-2)*(r1*V1**(-2)*Vr1+(-1)*r2*V2**(-2)*Vr2)
        dtaudVr2E = (-2)*(A1+A2)**(-2)*r2*V2**(-2)*(r1*V1**(-2)*Vr1+(-1)*r2*V2**(-2)*Vr2)

        # Apply chain rule to compute intermediate derivatives
        dtaudVr1 = dtaudV1*dV1dVr1 + dtaudA1*dA1dVr1  + dtaudVr1E
        dtaudVr2 = dtaudV2*dV2dVr2 + dtaudA2*dA2dVr2  + dtaudVr2E
        dtaudVtheta1 = dtaudV1*dV1dVtheta1 + dtaudA1*dA1dVtheta1
        dtaudVtheta2 = dtaudV2*dV2dVtheta2 + dtaudA2*dA2dVtheta2

        # Final Derivatives as per chain rule
        dy1dVr1 = dy1dA1*dA1dVr1 + dy1dtau*dtaudVr1 + dy1dV1*dV1dVr1
        dy1dVr2 = dy1dA2 * dA2dVr2 + dy1dtau * dtaudVr2 + dy1dV2 * dV2dVr2
        dy1dVtheta1 = dy1dA1 * dA1dVtheta1 + dy1dtau * dtaudVtheta1 + dy1dV1 * dV1dVtheta1
        dy1dVtheta2 = dy1dA2 * dA2dVtheta2 + dy1dtau * dtaudVtheta2 + dy1dV2 * dV2dVtheta2

        # Derivatives of y2 -- no chain rule needed
        dy2dVr1 = 2*Vr1
        dy2dVtheta1 = 2*Vtheta1

        return dy1dVr1, dy1dVtheta1, dy1dVr2, dy1dVtheta2, dy2dVr1, dy2dVtheta1




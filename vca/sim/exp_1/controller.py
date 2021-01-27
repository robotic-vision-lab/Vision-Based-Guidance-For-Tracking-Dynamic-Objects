from datetime import timedelta
from math import atan2, degrees, cos, sin pi

class Controller:
    def __init__(self, manager):
        self.manager = manager
        self.plot_info_file = 'plot_info.txt'
        self.R = CAR_RADIUS
        self.f = None
        self.a_ln = 0.0
        self.a_lt = 0.0
        # self.est_def = False


    @staticmethod
    def sat(x, bound):
        return min(max(x, -bound), bound)

    def generate_acceleration(self, kin):
        X, Y = kin[0]
        Vx, Vy = kin[1]
        car_x, car_y = kin[2]
        car_speed, cvy = kin[3]

        if USE_WORLD_FRAME:
            # add camera origin to positions
            orig = self.manager.get_cam_origin()
            X += orig[0]
            Y += orig[1]
            car_x += orig[0]
            car_y += orig[1]

        # speed of drone
        S = (Vx**2 + Vy**2) ** 0.5

        # heading angle of drone wrt x axis
        alpha = atan2(Vy, Vx)

        # heading angle of car
        beta = 0

        # distance between the drone and car
        r = ((car_x - X)**2 + (car_y - Y)**2)**0.5

        # angle of LOS from drone to car
        theta = atan2(car_y - Y, car_x - X)

        # compute Vr and Vθ
        Vr = car_speed * cos(beta - theta) - S * cos(alpha - theta)
        Vtheta = car_speed * sin(beta - theta) - S * sin(alpha - theta)

        # save measured r, θ, Vr, Vθ
        r_ = r
        theta_ = theta
        Vr_ = Vr
        Vtheta_ = Vtheta

        # at this point r, theta, Vr, Vtheta are computed
        # we can consider EKF filtering [r, theta, Vr, Vtheta]
        if not USE_TRUE_KINEMATICS and USE_EXTENDED_KALMAN:
            self.manager.EKF.add(r, theta, Vr, Vtheta, alpha, self.a_lt, self.a_ln)
            r, theta, Vr, Vtheta = self.manager.EKF.get_estimated_state()

        # calculate y from drone to car
        y2 = Vtheta**2 + Vr**2
        y1 = r**2 * Vtheta**2 - y2 * self.R**2
        # y1 = Vtheta**2 * (r**2 - self.R**2) - self.R**2 * Vr**2

        # time to collision from drone to car
        # tm = -vr * r / (vtheta**2 + vr**2)

        # compute desired acceleration
        w = w_
        K1 = K_1 * np.sign(-Vr)    # lat
        K2 = K_2                   # long

        # compute lat and long accelerations
        _D = 2 * Vr * Vtheta * r**2

        if abs(_D) < 0.01:
            a_lat = 0.0
            a_long = 0.0
        else:
            a_lat = (K1 * Vr * y1 * cos(alpha - theta) - K1 * Vr * w * cos(alpha - theta) - K1 * Vtheta * w * sin(alpha - theta) + K1 * Vtheta * y1 * sin(alpha - theta) +
                     K2 * self.R**2 * Vr * y2 * cos(alpha - theta) + K2 * self.R**2 * Vtheta * y2 * sin(alpha - theta) - K2 * Vtheta * r**2 * y2 * sin(alpha - theta)) / _D
            a_long = (K1 * Vtheta * w * cos(alpha - theta) - K1 * Vtheta * y1 * cos(alpha - theta) - K1 * Vr * w * sin(alpha - theta) + K1 * Vr * y1 * sin(alpha - theta) -
                      K2 * self.R**2 * Vtheta * y2 * cos(alpha - theta) + K2 * self.R**2 * Vr * y2 * sin(alpha - theta) + K2 * Vtheta * r**2 * y2 * cos(alpha - theta)) / _D

        a_long_bound = 5
        a_lat_bound = 5

        a_long = self.sat(a_long, a_long_bound)
        a_lat = self.sat(a_lat, a_lat_bound)

        self.a_ln = a_long
        self.a_lt = a_lat

        # compute acceleration command
        delta = alpha + pi / 2
        ax = a_lat * cos(delta) + a_long * cos(alpha)
        ay = a_lat * sin(delta) + a_long * sin(alpha)

        if not CLEAN_CONSOLE:
            print(f'CCC0 >> r:{r:0.2f} | theta:{theta:0.2f} | alpha:{alpha:0.2f} | car_speed:{car_speed:0.2f} | S:{S:0.2f} | Vr:{Vr:0.2f} | Vtheta:{Vtheta:0.2f} | y1:{y1:0.2f} | y2:{y2:0.2f} | a_lat:{a_lat:0.2f} | a_long:{a_long:0.2f} | _D:{_D:0.2f}')

        tru_kin = self.manager.get_true_kinematics()
        tX, tY = tru_kin[0]
        tVx, tVy = tru_kin[1]
        tcar_x, tcar_y = tru_kin[2]
        tcar_speed, tcvy = tru_kin[3]
        tS = (tVx**2 + tVy**2) ** 0.5
        tr = ((tcar_x - tX)**2 + (tcar_y - tY)**2)**0.5
        ttheta = atan2(tcar_y - tY, tcar_x - tX)
        tVr = tcar_speed * cos(beta - ttheta) - tS * cos(alpha - ttheta)
        tVtheta = tcar_speed * sin(beta - ttheta) - tS * sin(alpha - ttheta)

        tra_kin = self.manager.get_tracked_kinematics()
        vel = self.manager.simulator.camera.velocity
        if not CLEAN_CONSOLE:
            print(
                f'CCCC >> {str(timedelta(seconds=self.manager.simulator.time))} >> DRONE - x:[{X:0.2f}, {Y:0.2f}] | v:[{Vx:0.2f}, {Vy:0.2f}] | CAR - x:[{car_x:0.2f}, {car_y:0.2f}] | v:[{car_speed:0.2f}, {cvy:0.2f}] | COMMANDED a:[{ax:0.2f}, {ay:0.2f}] | TRACKED x:[{tra_kin[2][0]:0.2f},{tra_kin[2][1]:0.2f}] | v:[{tra_kin[3][0]:0.2f},{tra_kin[3][1]:0.2f}]')
        if self.manager.write_plot:
            self.f.write(
                f'{self.manager.simulator.time},' +                 # _TIME
                f'{r},' +                                           # _R
                f'{degrees(theta)},' +                              # _THETA
                f'{degrees(Vtheta)},' +                             # _V_THETA
                f'{Vr},' +                                          # _V_R
                f'{tru_kin[0][0]},' +                               # _DRONE_POS_X
                f'{tru_kin[0][1]},' +                               # _DRONE_POS_Y
                f'{tru_kin[2][0]},' +                               # _CAR_POS_X
                f'{tru_kin[2][1]},' +                               # _CAR_POS_y
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
                f'{S},' +                                           # _DRONE_SPEED
                f'{degrees(alpha)},' +                              # _DRONE_ALPHA
                f'{tru_kin[1][0]},' +                               # _DRONE_VEL_X
                f'{tru_kin[1][1]},' +                               # _DRONE_VEL_Y
                f'{tra_kin[2][0]},' +                               # _MEASURED_CAR_POS_X
                f'{tra_kin[2][1]},' +                               # _MEASURED_CAR_POS_Y
                f'{tra_kin[3][0]},' +                               # _MEASURED_CAR_VEL_X
                f'{tra_kin[3][1]},' +                               # _MEASURED_CAR_VEL_Y
                f'{self.manager.simulator.camera.altitude},' +      # _DRONE_ALTITUDE
                f'{abs(_D)},' +                                     # _ABS_DEN
                f'{r_},' +                                          # _MEASURED_R
                f'{degrees(theta_)},' +                             # _MEASURED_THETA
                f'{Vr_},' +                                         # _MEASURED_V_R
                f'{degrees(Vtheta_)},' +                            # _MEASURED_V_THETA
                f'{tr},' +                                          # _TRUE_R
                f'{degrees(ttheta)},' +                             # _TRUE_THETA
                f'{tVr},' +                                         # _TRUE_V_R
                f'{degrees(tVtheta)},' +                            # _TRUE_V_THETA
                f'{self.manager.simulator.dt},' +                   # _DELTA_TIME
                f'{y1},' +                                          # _Y1
                f'{y2}\n')                                          # _Y2

        if not self.manager.control_on:
            ax, ay = pygame.Vector2((0.0, 0.0))

        return ax, ay

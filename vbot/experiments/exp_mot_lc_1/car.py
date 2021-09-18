from math import pi, tau, degrees, atan2, sin, cos
from copy import deepcopy

import pygame

from .settings import *

from .my_imports import load_image_rect, scale_img

class Car(pygame.sprite.Sprite):
    """Defines a car sprite.
    """

    def __init__(self, simulator, x, y, vx=0.0, vy=0.0, ax=0.0, ay=0.0, loaded_image_rect=None, img=CAR_IMG,traj=DEFAULT_TRAJECTORY):
        # assign itself to the all_sprites group
        self.groups = [simulator.all_sprites, simulator.car_sprites]

        # call Sprite initializer with group info
        pygame.sprite.Sprite.__init__(self, self.groups)

        # assign Sprite.image and Sprite.rect attributes for this Sprite
        self.image, self.rect = loaded_image_rect
        self.img = img

        # set kinematics
        self.position = pygame.Vector2(x, y)
        self.initial_velocity = pygame.Vector2(vx, vy)
        self.velocity = pygame.Vector2(vx, vy)
        self.acceleration = pygame.Vector2(ax, ay)
        self.angle = pi/2
        self.traj = traj

        # hold onto the game/simulator reference
        self.simulator = simulator

        if (self.traj == ONE_HOLE_TRAJECTORY or
                self.traj == TWO_HOLE_TRAJECTORY or
                self.traj == SQUIRCLE_TRAJECTORY):
            self.init_x = 0
            self.init_y = 0
            # if self.traj== ONE_HOLE_TRAJECTORY:
            #     CW = -1
            #     ACW = 1
            #     DIRECTION = CW
            #     OFFSET = -pi/2
            #     t = -DELTA_TIME
            #     T = ONE_HOLE_PERIOD
            #     size = ONE_HOLE_SIZE
            #     OMEGA = tau/T
            #     self.velocity[0] = -(OMEGA*size) * sin(OMEGA*t*DIRECTION+OFFSET)
            #     self.velocity[1] = (OMEGA*size) * cos(OMEGA*t*DIRECTION+OFFSET)

            # if self.traj == TWO_HOLE_TRAJECTORY:
            #     CW = -1
            #     ACW = 1
            #     DIRECTION = CW
            #     OFFSET = -pi/2
            #     t = -DELTA_TIME
            #     T = TWO_HOLE_PERIOD
            #     size = TWO_HOLE_SIZE
            #     OMEGA = tau/T
            #     self.velocity[0] = -(OMEGA*size) * sin(OMEGA*t*DIRECTION+OFFSET)
            #     self.velocity[1] = (OMEGA*size) * cos(2*OMEGA*t*DIRECTION+OFFSET)
            
            self.velocity = pygame.Vector2(0, 0)
            self.acceleration = pygame.Vector2(0, 0)
            self.update_kinematics()

        # set initial rect location to position
        self.update_rect()

    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        CW = -1
        ACW = 1
        DIRECTION = ACW
        OFFSET = -pi/2
        if self.traj == ONE_HOLE_TRAJECTORY:
            t = self.simulator.time
            T = ONE_HOLE_PERIOD
            size = ONE_HOLE_SIZE
            OMEGA = tau/T
            self.velocity[0] = -(OMEGA*size) * sin(OMEGA*t*DIRECTION+OFFSET)
            self.velocity[1] = (OMEGA*size) * cos(OMEGA*t*DIRECTION+OFFSET)
            self.acceleration[0] = -(OMEGA**2*size) * cos(OMEGA*t*DIRECTION+OFFSET)
            self.acceleration[1] = -(OMEGA**2*size) * sin(OMEGA*t*DIRECTION+OFFSET)
            # self.velocity += self.acceleration * self.simulator.dt
            self.position += self.velocity * self.simulator.dt
            self.angle = atan2(cos(OMEGA*t*DIRECTION+OFFSET), - sin(OMEGA*t*DIRECTION+OFFSET))

        elif self.traj == TWO_HOLE_TRAJECTORY:
            t = self.simulator.time
            T = TWO_HOLE_PERIOD
            size = TWO_HOLE_SIZE
            OMEGA = tau/T
            self.velocity[0] = -(OMEGA*size) * sin(OMEGA*t*DIRECTION+OFFSET)
            self.velocity[1] = (OMEGA*size) * cos(2*OMEGA*t*DIRECTION+OFFSET)
            self.acceleration[0] = -(OMEGA**2*size) * cos(OMEGA*t*DIRECTION+OFFSET)
            self.acceleration[1] = -(OMEGA**2*size) * sin(2*OMEGA*t*DIRECTION+OFFSET)
            # self.velocity += self.acceleration * self.simulator.dt
            self.position += self.velocity * self.simulator.dt
            self.angle = atan2(cos(2*OMEGA*t*DIRECTION+OFFSET), - sin(OMEGA*t*DIRECTION+OFFSET))
        
        elif self.traj == LANE_CHANGE_TRAJECTORY_1 or self.traj == LANE_CHANGE_TRAJECTORY_2:
            _FLAG = 1 if self.traj == LANE_CHANGE_TRAJECTORY_1 else -1.5
            t = self.simulator.time
            if 5 <= t < 7:
                self.acceleration = pygame.Vector2(0, _FLAG* -1)
                self.velocity += self.acceleration * self.simulator.dt
            elif 7 <= t < 9:
                self.acceleration = pygame.Vector2(0, _FLAG* +1)
                self.velocity += self.acceleration * self.simulator.dt

            elif 15 <= t < 17:
                self.acceleration = pygame.Vector2(0, _FLAG* +1)
                self.velocity += self.acceleration * self.simulator.dt
            elif 17 <= t < 19:
                self.acceleration = pygame.Vector2(0, _FLAG* -1)
                self.velocity += self.acceleration * self.simulator.dt

            elif 25 <= t < 27:
                self.acceleration = pygame.Vector2(0, _FLAG* -1)
                self.velocity += self.acceleration * self.simulator.dt
            elif 27 <= t < 29:
                self.acceleration = pygame.Vector2(0, _FLAG* +1)
                self.velocity += self.acceleration * self.simulator.dt

            elif 31 <= t < 33:
                self.acceleration = pygame.Vector2(0, _FLAG* -1)
                self.velocity += self.acceleration * self.simulator.dt
            elif 33 <= t < 35:
                self.acceleration = pygame.Vector2(0, _FLAG* +1)
                self.velocity += self.acceleration * self.simulator.dt

            elif 39 <= t < 55:
                self.acceleration = pygame.Vector2(0, _FLAG* 0)
                self.velocity = deepcopy(self.initial_velocity)

            elif 55 <= t < 57:
                self.acceleration = pygame.Vector2(0, _FLAG* +1)
                self.velocity += self.acceleration * self.simulator.dt
            elif 57 <= t < 59:
                self.acceleration = pygame.Vector2(0, _FLAG* -1)
                self.velocity += self.acceleration * self.simulator.dt

            elif 75 <= t < 77:
                self.acceleration = pygame.Vector2(0, _FLAG* -1)
                self.velocity += self.acceleration * self.simulator.dt
            elif 77 <= t < 79:
                self.acceleration = pygame.Vector2(0, _FLAG* +1)
                self.velocity += self.acceleration * self.simulator.dt

            elif 115 <= t < 117:
                self.acceleration = pygame.Vector2(0, _FLAG* +1)
                self.velocity += self.acceleration * self.simulator.dt
            elif 117 <= t < 119:
                self.acceleration = pygame.Vector2(0, _FLAG* -1)
                self.velocity += self.acceleration * self.simulator.dt

            else:
                self.acceleration = pygame.Vector2(0, 0)
                self.velocity = deepcopy(self.initial_velocity)

            self.position += self.velocity * self.simulator.dt

        elif self.traj == SQUIRCLE_TRAJECTORY:
            s = SQUIRCLE_PARAM_S
            r = SQUIRCLE_PARAM_R
            t = self.simulator.time
            T = SQUIRCLE_PERIOD
            OMEGA = tau/T
            p = OMEGA*t - pi/2
            n = 1
            srt2 = s* 2**1.5
            # r *= 1 + 1/8 * (sin(2*n*p))**2
            # self.position[0] = r * cos(p)
            # self.position[1] = r * sin(p)

            # x = (r / 2*s) * (2 + srt2* cos(p)  + s**2 * cos(2*p))**.5 - (r / 2*s) * (2 - srt2*cos(p)  + s**2 * cos(2*p))**.5
            # y = (r / 2*s) * (2 + srt2* sin(p)  - s**2 * cos(2*p))**.5 - (r / 2*s) * (2 - srt2*sin(p)  - s**2 * cos(2*p))**.5
            

            vx = ((r*OMEGA)/(4*s)) * (((2+srt2*cos(p)+s**2*cos(2*p))**-0.5*(-srt2*sin(p)-2*s**2*sin(2*p))) - ((2-srt2*cos(p)+s**2*cos(2*p))**-0.5*(srt2*sin(p)-2*s**2*sin(2*p))))
            vy = ((r*OMEGA)/(4*s)) * (((2+srt2*sin(p)-s**2*cos(2*p))**-0.5*(srt2*cos(p)+2*s**2*sin(2*p))) - ((2-srt2*sin(p)-s**2*cos(2*p))**-0.5*(-srt2*cos(p)+2*s**2*sin(2*p))))
            self.velocity[0]=vx
            self.velocity[1]=vy
            self.angle=atan2(vy,vx)
            self.position += self.velocity * self.simulator.dt


        else:   # DEFAULT_TRAJECTORY
            # update velocity and position
            self.velocity += self.acceleration * self.simulator.dt
            self.position += self.velocity * self.simulator.dt
        
    def update_rect(self):
        """Update car sprite's rect position.
        """
        x, y = self.position.elementwise() * (1, -1) / self.simulator.pxm_fac
        self.rect.centerx = int(x)
        self.rect.centery = int(y) + HEIGHT
        self.rect.center += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)

    def update_image_rect(self):
        """Update car image and rect for changes in orientations
        """
        if (self.traj == ONE_HOLE_TRAJECTORY or
                self.traj == TWO_HOLE_TRAJECTORY or 
                self.traj == SQUIRCLE_TRAJECTORY):
            # load the unrotated image and rect
            car_scale = (CAR_LENGTH / (CAR_LENGTH_PX * self.simulator.pxm_fac))
            self.car_img_rect = load_image_rect(CAR_IMG, colorkey=BLACK, alpha=True, scale=car_scale)
            prev_center = self.car_img_rect[0].get_rect(center = self.car_img_rect[0].get_rect().center).center
            rot_img = pygame.transform.rotate(self.car_img_rect[0], degrees(self.angle))
            rot_img = rot_img.convert_alpha()
            rot_rect = rot_img.get_rect(center = prev_center)
            self.car_img_rect = (rot_img, rot_rect)
            self.image, self.rect = self.car_img_rect
            self.update_rect()
        else:
            car_scale = (CAR_LENGTH / (CAR_LENGTH_PX * self.simulator.pxm_fac))
            self.car_img_rect = load_image_rect(self.img, colorkey=BLACK, alpha=True, scale=car_scale)
            self.image, self.rect = self.car_img_rect
            self.update_rect()


    def update(self):
        """ update sprite attributes.
            This will get called in game loop for every frame
        """
        self.update_kinematics()
        # self.update_rect()
        # self.rect.center = self.position + SCREEN_CENTER

    def load(self):
        """Helper function called when altitude is changed. Updates image and rect.
        """
        self.image, self.rect = self.simulator.car_img_rect
        self.update_rect()

    def get_position(self):
        """helper function returns position in world frame"""
        return pygame.Vector2(self.simulator.camera.origin) + pygame.Vector2(self.position)

    def get_true_LOS_kinematics(self):
        x, y = self.get_position()
        vx, vy = self.velocity

        car_speed = (vx**2 + vy**2)**0.5
        car_beta = atan2(vy, vx)

        cam_origin_x, cam_origin_y = self.simulator.manager.get_cam_origin()

        # drone (known)
        drone_pos_x, drone_pos_y = self.simulator.manager.get_true_drone_position()
        drone_vel_x, drone_vel_y = self.simulator.manager.get_true_drone_velocity()
        drone_pos_x += cam_origin_x
        drone_pos_y += cam_origin_y

        drone_speed = (drone_vel_x**2 + drone_vel_y**2)**0.5
        drone_alpha = atan2(drone_vel_y, drone_vel_x)

        r = ((x - drone_pos_x)**2 + (y - drone_pos_y)**2)**0.5
        theta = atan2(y - drone_pos_y, x - drone_pos_x)
        Vr = car_speed * cos(car_beta - theta) - drone_speed * cos(drone_alpha - theta)
        Vtheta = car_speed * sin(car_beta - theta) - drone_speed * sin(drone_alpha - theta)

        return (r, theta, Vr, Vtheta)






    
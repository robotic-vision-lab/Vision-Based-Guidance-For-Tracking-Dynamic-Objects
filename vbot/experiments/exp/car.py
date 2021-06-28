from math import pi, tau, degrees, atan2, sin, cos
from copy import deepcopy

import pygame

from .settings import *

from .my_imports import load_image_rect

class Car(pygame.sprite.Sprite):
    """Defines a car sprite.
    """

    def __init__(self, simulator, x, y, vx=0.0, vy=0.0, ax=0.0, ay=0.0, loaded_image_rect=None, traj=DEFAULT_TRAJECTORY):
        # assign itself to the all_sprites group
        self.groups = [simulator.all_sprites, simulator.car_sprites]

        # call Sprite initializer with group info
        pygame.sprite.Sprite.__init__(self, self.groups)

        # assign Sprite.image and Sprite.rect attributes for this Sprite
        self.image, self.rect = loaded_image_rect

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
        
        elif self.traj == LANE_CHANGE_TRAJECTORY:
            t = self.simulator.time
            if t >= 5 and t < 7:
                self.acceleration = pygame.Vector2(0, -1)
                self.velocity += self.acceleration * self.simulator.dt
            elif t >= 7 and t < 9:
                self.acceleration = pygame.Vector2(0, +1)
                self.velocity += self.acceleration * self.simulator.dt

            elif t >= 15 and t < 17:
                self.acceleration = pygame.Vector2(0, +1)
                self.velocity += self.acceleration * self.simulator.dt
            elif t >= 17 and t < 19:
                self.acceleration = pygame.Vector2(0, -1)
                self.velocity += self.acceleration * self.simulator.dt

            elif t >= 25 and t < 27:
                self.acceleration = pygame.Vector2(0, -1)
                self.velocity += self.acceleration * self.simulator.dt
            elif t >= 27 and t < 29:
                self.acceleration = pygame.Vector2(0, +1)
                self.velocity += self.acceleration * self.simulator.dt

            elif t >= 35 and t < 37:
                self.acceleration = pygame.Vector2(0, -1)
                self.velocity += self.acceleration * self.simulator.dt
            elif t >= 37 and t < 39:
                self.acceleration = pygame.Vector2(0, +1)
                self.velocity += self.acceleration * self.simulator.dt

            elif t >= 55 and t < 57:
                self.acceleration = pygame.Vector2(0, +1)
                self.velocity += self.acceleration * self.simulator.dt
            elif t >= 57 and t < 59:
                self.acceleration = pygame.Vector2(0, -1)
                self.velocity += self.acceleration * self.simulator.dt

            elif t >= 75 and t < 77:
                self.acceleration = pygame.Vector2(0, -1)
                self.velocity += self.acceleration * self.simulator.dt
            elif t >= 77 and t < 79:
                self.acceleration = pygame.Vector2(0, +1)
                self.velocity += self.acceleration * self.simulator.dt

            elif t >= 105 and t < 107:
                self.acceleration = pygame.Vector2(0, +1)
                self.velocity += self.acceleration * self.simulator.dt
            elif t >= 107 and t < 109:
                self.acceleration = pygame.Vector2(0, -1)
                self.velocity += self.acceleration * self.simulator.dt

            elif t >= 39:
                self.acceleration = pygame.Vector2(0, 0)
                self.velocity = deepcopy(self.initial_velocity)
            else:
                self.acceleration = pygame.Vector2(0,0)
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
            self.car_img_rect = load_image_rect(CAR_IMG, colorkey=BLACK, alpha=True, scale=CAR_SCALE)
            prev_center = self.car_img_rect[0].get_rect(center = self.car_img_rect[0].get_rect().center).center
            rot_img = pygame.transform.rotate(self.car_img_rect[0], degrees(self.angle))
            rot_img = rot_img.convert_alpha()
            rot_rect = rot_img.get_rect(center = prev_center)
            self.car_img_rect = (rot_img, rot_rect)
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
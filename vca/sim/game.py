import os
import sys
import pygame
from copy import deepcopy

from pygame.locals import *
from settings import *
from car import Car
from block import Block
from drone_camera import DroneCamera
from game_utils import *

class Game:
    """Simulation Game
    """
    def __init__(self):
        # initialize screen
        pygame.init()
        self.screen_surface = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption(SCREEN_DISPLAY_TITLE)

        # instantiate a clock to keep time 
        self.clock = pygame.time.Clock()

        # initialize images for Sprites
        self.car_img = load_image(CAR_IMG, colorkey=BLACK, alpha=True, scale=CAR_SCALE)
        self.drone_img = load_image(DRONE_IMG, colorkey=BLACK, alpha=True, scale=DRONE_SCALE)

        # set screen saving mode to OFF
        self.save_screen = False

        self.eucildean_factor = 1.0


    def start_new(self):
        """helper function to perform tasks when we start a new game.
        """
        # initiate screen shot generator
        self.screen_shot = screen_saver(self.screen_surface, TEMP_FOLDER)

        # create a Group with all sprites 
        self.all_sprites = pygame.sprite.Group()

        # spawn car, instantiating the Car sprite would add itself to all_sprites
        self.car = Car(self, *CAR_INITIAL_POSITION, *CAR_INITIAL_VELOCITY, *CAR_ACCELERATION)

        # spawn block, instantiating the Block sprite would add itself to all_sprites
        self.blocks = []
        num_blocks = 8
        for i in range(num_blocks):
            self.blocks.append(Block(self))

        # create camera
        self.camera = DroneCamera(self)
        self.cam_accel_command = pygame.Vector2(0,0)
            

    def quit(self):
        """quits game, exits application
        """
        self.running = False
        pygame.quit()
        # sys.exit()


    def handle_events(self):
        """handles events in event queue while running game loop
        """
        for event in pygame.event.get():
            # check for closing window or ESC button press
            if event.type == pygame.QUIT or \
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit()
                break

            # check for s keypress (toggle screen saving on/off)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.save_screen = not self.save_screen

            # read acceleration commands
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.cam_accel_command.x = -1
            if keys[pygame.K_RIGHT]:
                self.cam_accel_command.x = 1
            if keys[pygame.K_UP]:
                self.cam_accel_command.y = -1
            if keys[pygame.K_DOWN]:
                self.cam_accel_command.y = 1

            self.eucildean_factor = 0.7071 if not (self.cam_accel_command.x == 0 or self.cam_accel_command.y == 0) else 1.0
            pygame.event.pump()



    def update(self):
        """Helper function to update game objects. 
            This method will be run in the game loop every frame
        """
        # update Group to update all sprites in it
        self.all_sprites.update()
        self.camera.move(deepcopy(self.eucildean_factor * self.cam_accel_command))
        self.cam_accel_command = pygame.Vector2(0, 0)

    def draw(self):
        """helper function to draw/render each frame
        """
        # fill background 
        self.screen_surface.fill(SCREEN_BG_COLOR)
        pygame.display.set_caption(f'car position {self.car.position} | cam position {self.camera.position}')
        # draw the sprites
        for sprite in self.all_sprites:
            self.camera.compensate_camera_motion(sprite)
        self.all_sprites.draw(self.screen_surface)

        # flip drawing board
        pygame.display.flip()


    def run(self):
        """the game loop
        """
        self.running = True
        while self.running:
            # run loop at desired FPS by introducing a wait of 1/FPS seconds
            # also save time elapsed dt in seconds 
            self.dt = self.clock.tick(FPS) / 1000.0

            # handle events
            self.handle_events()
            if not self.running:
                break

            # update game objects
            self.update()

            # draw stuffs
            self.draw()

            # save screen 
            if self.save_screen:
                next(self.screen_shot)





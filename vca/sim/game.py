import os
import sys
import pygame

from pygame.locals import *
from settings import *
from car import Car
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
        self.car_img = load_image(CAR_IMG, colorkey=BLACK, alpha=True)

        self.save_screen = False

    def start_new(self):
        """helper function to perform tasks when we start a new game.
        """
        # initiate screen shot generator
        self.screen_shot = screen_saver(self.screen_surface, TEMP_FOLDER)

        # create Groups
        self.all_sprites = pygame.sprite.Group()

        # spawn car
        self.car = Car(self, *CAR_INITIAL_POSITION, *CAR_INITIAL_VELOCITY, *CAR_ACCELERATION)

    def quit(self):
        """quits game, exits application
        """
        pygame.quit()
        sys.exit()

    def handle_events(self):
        """handles events in event queue while running game loop
        """
        for event in pygame.event.get():
            # check for closing window or ESC button press
            if event.type == pygame.QUIT or \
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit()

            # check for s keypress (toggle screen saving on/off)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.save_screen = not self.save_screen

    def update(self):
        """Helper function to update game objects. 
            This method will be run in the game loop every frame
        """
        # update Group to update all sprites in it
        self.all_sprites.update()

    def draw(self):
        """helper function to draw/render each frame
        """
        # fill background 
        self.screen_surface.fill(SCREEN_BG_COLOR)

        # draw the sprites
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

            # update game objects
            self.update()

            # draw stuffs
            self.draw()

            # save screen 
            if self.save_screen:
                next(self.screen_shot)





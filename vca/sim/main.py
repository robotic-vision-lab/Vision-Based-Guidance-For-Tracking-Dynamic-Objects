import pygame 

from pygame.locals import *

from game import Game

if __name__ == "__main__":
    # instantiate game 
    car_sim_game = Game()

    # start new game
    car_sim_game.start_new()

    # run
    car_sim_game.run()

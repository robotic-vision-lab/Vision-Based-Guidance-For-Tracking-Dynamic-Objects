import pygame
from random import randrange

from pygame.locals import *

"""This module defines a Block sprite using pygame.sprite.Sprite.
    The base class for visible game objects. 
    Derived classes will want to override the Sprite.update() and 
    assign a Sprite.image and Sprite.rect attributes. 
    The initializer can accept any number of Group instances to be added to.

    When subclassing the Sprite, be sure to call the base initializer 
    before adding the Sprite to Groups.

    note: Adopted from https://www.pygame.org/docs/ref/sprite.html#pygame.sprite.Sprite
"""
from settings import *

class Block(pygame.sprite.Sprite):
    """[summary]

    Args:
        pygame ([type]): [description]
    """

    # Constructor. Pass in the color of the block,
    # and its x and y position
    def __init__(self, game):
        self.groups = game.all_sprites
        
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface(BLOCK_SIZE)
        self.image.fill(BLOCK_COLOR)

        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()

        self.reset_kinematics()
        self.game = game


    def reset_kinematics(self):
        """resets the kinematics of block
        """
        # set vectors representing the position, velocity and acceleration
        # note the velocity we assign below will be interpreted as pixels/sec
        self.position = pygame.Vector2(randrange(WIDTH - self.rect.width), randrange(HEIGHT - self.rect.height))
        self.velocity = pygame.Vector2(0,0)#(randrange(-50, 50), randrange(-50, 50))
        self.acceleration = pygame.Vector2(0, 0)



    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # update velocity and position
        self.velocity += self.acceleration * self.game.dt
        self.position += self.velocity * self.game.dt + 0.5 * self.acceleration * self.game.dt**2

        # re-spawn in view
        if self.position.x > WIDTH or \
            self.position.x < 0 - self.rect.width or \
            self.position.y > HEIGHT or \
            self.position.y < 0 - self.rect.height:
            self.reset_kinematics()


    def update(self):
        """Overwrites Sprite.update()
            When we call update() on a group this methods gets called.
            Every next frame while running the game loop this will get called
        """
        # for example if we want the sprite to move 5 pixels to the right
        self.update_kinematics()
        self.rect.topleft = self.position

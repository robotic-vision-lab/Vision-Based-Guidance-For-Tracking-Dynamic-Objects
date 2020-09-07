import pygame

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

class Block(pygame.sprite.Sprite):

    # Constructor. Pass in the color of the block,
    # and its x and y position
    def __init__(self, color, width, height):
       # Call the parent class (Sprite) constructor
       pygame.sprite.Sprite.__init__(self)

       # Create an image of the block, and fill it with a color.
       # This could also be an image loaded from the disk.
       self.image = pygame.Surface([width, height])
       self.image.fill(color)

       # Fetch the rectangle object that has the dimensions of the image
       self.rect = self.image.get_rect()

       # set vectors representing the position, velocity and acceleration
       self.position = pygame.Vector2()
       self.velocity = pygame.Vector2()
       self.acceleration = pygame.Vector2()


    def update(self):
        """Overwrites Sprite.update()
            When we call update() on a group this methods gets called.
            Every next frame while running the game loop this will get called
        """
        # for example if we want the sprite to move 5 pixels to the right
        self.rect.x += 5

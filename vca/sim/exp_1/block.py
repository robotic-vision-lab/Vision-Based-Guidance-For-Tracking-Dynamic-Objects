from random import randint, randrange, uniform
import pygame

from settings import *

class Block(pygame.sprite.Sprite):
    """Defines a Block sprite.
    """

    def __init__(self, simulator):
        self.groups = [simulator.all_sprites, simulator.car_block_sprites]

        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        self.simulator = simulator
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.w = BLOCK_WIDTH / PIXEL_TO_METERS_FACTOR
        self.h = BLOCK_HEIGHT / PIXEL_TO_METERS_FACTOR
        self.image = pygame.Surface((int(self.w), int(self.h)))
        self.fill_image()

        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()

        # self.reset_kinematics()
        _x = randrange(-(WIDTH - self.rect.width), (WIDTH - self.rect.width))
        _y = randrange(-(HEIGHT - self.rect.height), (HEIGHT - self.rect.height))
        self.position = pygame.Vector2(_x, _y) * self.simulator.pxm_fac
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)

        # self.rect.center = self.position
        self.update_rect()

    def reset_kinematics(self):
        """resets the kinematics of block
        """
        # set vectors representing the position, velocity and acceleration
        # note the velocity we assign below will be interpreted as pixels/sec
        fov = self.simulator.get_camera_fov()
        drone_pos = self.simulator.get_drone_position()

        _x = uniform(drone_pos[0] - fov[0] / 2, drone_pos[0] + fov[0])
        _y = uniform(drone_pos[1] - fov[1] / 2, drone_pos[1] + fov[1])
        self.position = pygame.Vector2(_x, _y)
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)

    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # update velocity and position
        self.velocity += self.acceleration * self.simulator.dt
        self.position += self.velocity * self.simulator.dt #+ 0.5 * \
            #self.acceleration * self.simulator.dt**2  # pylint: disable=line-too-long

        # re-spawn in view
        if self.rect.centerx > WIDTH or \
                self.rect.centerx < 0 - self.rect.width or \
                self.rect.centery > HEIGHT or \
                self.rect.centery < 0 - self.rect.height:
            self.reset_kinematics()

    def update_rect(self):
        """Position information is in bottom-left reference frame.
        This method transforms it to top-left reference frame and update the sprite's rect.
        This is for rendering purposes only, to decide where to draw the sprite.
        """

        x, y = self.position.elementwise() * (1, -1) / self.simulator.pxm_fac
        self.rect.centerx = int(x)
        self.rect.centery = int(y) + HEIGHT
        self.rect.center += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)

    def update(self):
        """Overwrites Sprite.update()
            When we call update() on a group this methods gets called.
            Every next frame while running the game loop this will get called
        """
        # for example if we want the sprite to move 5 pixels to the right
        self.update_kinematics()
        # self.update_rect()
        # self.rect.center = self.position

    def fill_image(self):
        """Helper function fills block image
        """
        r, g, b = BLOCK_COLOR
        d = BLOCK_COLOR_DELTA
        r += randint(-d, d)
        g += randint(-d, d)
        b += randint(-d, d)
        self.image.fill((r, g, b))

    def load(self):
        """Helper function updates width and height of image and fills image.
        Also, updates rect.
        """
        self.w /= self.simulator.alt_change_fac
        self.h /= self.simulator.alt_change_fac

        if self.w >= 2 and self.h >= 2:
            self.image = pygame.Surface((int(self.w), int(self.h)))
            self.fill_image()

        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()

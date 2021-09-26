from random import randrange, randint, uniform
import pygame

from .settings import *

class Bar(pygame.sprite.Sprite):
    """Defines an Occlusion Bar sprite.
    """

    def __init__(self, simulator):
        self.groups = [simulator.all_sprites, simulator.bar_sprites]

        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.groups)

        self.simulator = simulator
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.w = BAR_WIDTH / PIXEL_TO_METERS_FACTOR
        self.h = BAR_HEIGHT / PIXEL_TO_METERS_FACTOR
        self.image = pygame.Surface((int(self.w), int(self.h)))
        self.fill_image()

        # fetch bar Rect object from bar image and update bar rect
        self.rect = self.image.get_rect()

        # self.reset_kinematics()
        _x = WIDTH + self.rect.centerx
        # _x = randrange(-(WIDTH - self.rect.width), (WIDTH - self.rect.width))
        _y = randrange(-(HEIGHT - self.rect.height), (HEIGHT - self.rect.height))

        self.position = pygame.Vector2(_x, _y) * self.simulator.pxm_fac
        self.velocity = pygame.Vector2(0.0, 0.0)
        self.acceleration = pygame.Vector2(0.0, 0.0)

        # self.rect.center = self.position
        self.update_rect()

    def reset_kinematics(self):
        """resets the kinematics of occlusion bar
        """
        # set vectors representing the position, velocity and acceleration
        # note the velocity we assign below will be interpreted as pixels/sec
        fov = self.simulator.get_camera_fov()
        drone_pos = self.simulator.get_drone_position()

        if self.simulator.time < 25:
            _x = -fov[0] #drone_pos[0] + fov[0] * uniform(0.5, 1.0)
        else:
            _x = drone_pos[0] + fov[0] * uniform(0.5, 1.0)
        _y = uniform(drone_pos[1] - fov[1] / 3, drone_pos[1] + fov[1]/4)
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
        if self.rect.centerx > WIDTH + self.rect.width/2 or \
                self.rect.centerx < 0 - self.rect.width/2 or \
                self.rect.centery > HEIGHT + - self.rect.height/2 or \
                self.rect.centery < 0 - self.rect.height/2:
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

        # use BAR_COLOR and BAR_DELTA to generate random color 
        r, g, b = BAR_COLOR
        d = BAR_COLOR_DELTA
        r += randint(-d[0], d[0])
        g += randint(-d[1], d[1])
        b += randint(-d[2], d[2])

        # fill bar with random color
        self.image.fill((r, g, b))

    def load(self):
        """Helper function updates image width and height and fills image.
        Also, updates rect.
        """

        # recompute width and height corresponding to altitude change
        self.w /= self.simulator.alt_change_fac
        self.h /= self.simulator.alt_change_fac

        # update the image width and height, only if >= 2
        if self.w >= 2 and self.h >= 2:
            self.image = pygame.Surface((int(self.w), int(self.h)))
            self.fill_image()

        # fetch bar Rect object from image, and update bar rect
        self.rect = self.image.get_rect()

import pygame

class Car(pygame.sprite.Sprite):
    """Defines a car sprite.
    """

    def __init__(self, simulator, x, y, vx=0.0, vy=0.0, ax=0.0, ay=0.0):
        # assign itself to the all_sprites group
        self.groups = [simulator.all_sprites, simulator.car_block_sprites]

        # call Sprite initializer with group info
        pygame.sprite.Sprite.__init__(self, self.groups)

        # assign Sprite.image and Sprite.rect attributes for this Sprite
        self.image, self.rect = simulator.car_img

        # set kinematics
        # note the velocity and acceleration we assign below
        # will be interpreted as pixels/sec
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(vx, vy)
        self.acceleration = pygame.Vector2(ax, ay)

        # hold onto the game/simulator reference
        self.simulator = simulator

        # set initial rect location to position
        self.update_rect()

    def update_kinematics(self):
        """helper function to update kinematics of object
        """
        # update velocity and position
        self.velocity += self.acceleration * self.simulator.dt
        self.position += self.velocity * self.simulator.dt #+ 0.5 * \
            # self.acceleration * self.simulator.dt**2  # pylint: disable=line-too-long

    def update_rect(self):
        """update car sprite's rect.
        """
        x, y = self.position.elementwise() * (1, -1) / self.simulator.pxm_fac
        self.rect.centerx = int(x)
        self.rect.centery = int(y) + HEIGHT
        self.rect.center += pygame.Vector2(SCREEN_CENTER).elementwise() * (1, -1)

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
        self.image, self.rect = self.simulator.car_img
        self.update_rect()
        # self.rect.center = self.position + SCREEN_CENTER

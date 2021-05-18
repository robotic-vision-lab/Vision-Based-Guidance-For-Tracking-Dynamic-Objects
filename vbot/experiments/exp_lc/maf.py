from queue import deque

import pygame
from pygame.locals import *

class MA:
    """Filters statefully using a moving average technique
    """

    def __init__(self, window_size=10):
        self.car_x = deque(maxlen=window_size)
        self.car_y = deque(maxlen=window_size)
        self.car_vx = deque(maxlen=window_size)
        self.car_vy = deque(maxlen=window_size)

        self.ready = False

        # self.old_pos = self.avg_pos()
        # self.old_vel = self.avg_vel()

    def done_waiting(self):
        """Indicates readiness of filter

        Returns:
            bool: Ready or not
        """
        return len(self.car_vx) > 5

    def init_filter(self, pos, vel):
        """Initializes filter. Meant to be run for the first time only.

        Args:
            pos (pygame.Vector2): Car position measurement
            vel (pygame.Vector2): Car velocity measurement
        """
        self.new_pos = pygame.Vector2(pos)
        self.new_vel = pygame.Vector2(vel)
        self.add_pos(pos)
        self.add_vel(vel)
        self.ready = True

    def add(self, pos, vel):
        """Add a measurement

        Args:
            pos (pygame.Vector2): Car position measurement
            vel (pygame.Vector2): Car velocity measurement
        """
        # remember the last new average before adding to deque
        self.old_pos = self.new_pos
        self.old_vel = self.new_vel

        # add to deque
        self.car_x.append(pos[0])
        self.car_y.append(pos[1])
        self.car_vx.append(vel[0])
        self.car_vy.append(vel[1])

        # compute new average
        self.new_pos = self.avg_pos()
        self.new_vel = self.avg_vel()

    def add_pos(self, pos):
        """Add position measurement

        Args:
            pos (pygame.Vector2): Car position measurement
        """
        # remember the last new average before adding to deque
        self.old_pos = self.new_pos

        # add to deque
        self.car_x.append(pos[0])
        self.car_y.append(pos[1])

        # compute new average
        self.new_pos = self.avg_pos()

    def add_vel(self, vel):
        """Add velocity measurement

        Args:
            vel (pygame.Vector2): Car velocity measurement
        """
        # remember the last new average before adding to deque
        self.old_vel = self.new_vel

        # add to deque
        self.car_vx.append(vel[0])
        self.car_vy.append(vel[1])

        # compute new average
        self.new_vel = self.avg_vel()

    def get_pos(self):
        """Fetch estimated position

        Returns:
            pygame.Vector2: Car estimate position
        """
        return self.new_pos

    def get_vel(self):
        """Get estimated velocity

        Returns:
            pygame.Vector2: Car estimated velocity
        """
        return self.new_vel

    def avg_pos(self):
        """Helper function to average position measurements

        Returns:
            pygame.Vector2: Averaged car position
        """
        x = sum(self.car_x) / len(self.car_x)
        y = sum(self.car_y) / len(self.car_y)
        return pygame.Vector2(x, y)

    def avg_vel(self):
        """Helper function to average velocity measurements

        Returns:
            pygame.Vector2: Averaged car velocity
        """
        vx = sum(self.car_vx) / len(self.car_vx)
        vy = sum(self.car_vy) / len(self.car_vy)
        return pygame.Vector2(vx, vy)


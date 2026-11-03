import math


class Brush:
    def __init__(self, radius=30, center_x=0, center_y=0):
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y


    def is_in_circle (self, x, y):
        return ((x - self.center_x) * (x - self.center_x)) + ((y - self.center_y) * (y - self.center_y)) <= self.radius * self.radius

    """
    Guassian falloff for how strongly point is affected by brush
    """
    def falloff(self, x, y) -> float:
        if not self.is_in_circle(x, y):
            return 0.0

        d2 = (x - self.center_x)**2 + (y - self.center_y)**2
        sigma = self.radius / 3

        return math.exp(-d2 / (2 * sigma**2))
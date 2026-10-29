class Brush:
    def __init__(self, radius=30, center_x=0, center_y=0):
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y


    def is_in_circle (self, x, y):
        ((x - self.center_x) * (x - self.center_x)) + ((y - self.center_y) * (y - self.center_y)) <= self.radius * self.radius

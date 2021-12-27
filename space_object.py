import math
import config
import game_engine as engine
import random

class SpaceObject:
    # STATIC attributes
    max_angle = 360
    min_angle = 0

    def __init__(self, x, y, width, height, angle, obj_type, id):       
        # ATTRIBUTES
        self.x = float(x)
        self.y = float(y)
        self.width = int(width)
        self.height = int(height)
        self.angle = int(angle)
        self.obj_type = str(obj_type)
        self.id = int(id)


    def check_xy(self):
        # Creating a function that will check if the object goes out of the map and "dragging" it back to the other side
        if self.x > self.width:
            self.x -= self.width
        if self.x < 0:
            self.x = self.width + self.x
        if self.y > self.height:
            self.y -= self.height
        if self.y < 0:
            self.y = self.height + self.y


    def turn_left(self):
        self.angle += config.angle_increment
        if self.angle >= SpaceObject.max_angle:
            self.angle -= SpaceObject.max_angle
        


    def turn_right(self):
        self.angle -= config.angle_increment
        if self.angle < SpaceObject.min_angle:
            self.angle = SpaceObject.max_angle + self.angle


    def get_xy(self):
        xy = tuple ([self.x,self.y])
        return xy


    def update_score(self,score):
        self.score = score


    def move_forward(self):
        # Doing some math problem according to the request
        dx = math.cos(math.radians(self.angle))
        dy = math.sin(math.radians(self.angle))
        self.x += config.speed[self.obj_type] * dx
        self.y -= config.speed[self.obj_type] * dy
        SpaceObject.check_xy(self)


    def collide_with(self, other):
        x_distance = abs(self.x - other.x)
        y_distance = abs(self.y - other.y)
        # Check their distance in case they are at opposite edges of the screen
        if x_distance > self.width/2:
            x_distance = self.width - x_distance
        if y_distance > self.height/2:
            y_distance = self.height - y_distance      
        d = math.sqrt(math.pow(x_distance , 2) + math.pow(y_distance , 2))
        if d <= config.radius[self.obj_type] + config.radius[other.obj_type]:
            return True
        else:
            return False


    def __repr__(self):
        x = "{} {},{},{},{}".format(self.obj_type , round(self.x , 1) , round(self.y , 1) , self.angle , self.id)
        return x

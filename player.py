# 03A751 on the leaderboard

import math
import config

class Player:
    def __init__(self):
        self.shot_asteroids = []

    def action(self, spaceship, asteroid_ls, bullet_ls, fuel, score):
        # sort asteroids based on steps_to_reach
        asteroid_ls.sort(key=lambda x: self.steps_to_reach(spaceship, x))

        # don't shoot the same asteroid twice
        while len(asteroid_ls) > 1 and asteroid_ls[0].id in self.shot_asteroids:
            asteroid_ls.pop(0)

        # fly to the closest asteroid
        target = asteroid_ls[0]
        dist, angle = self.get_dist_angle(spaceship, target)

        # shoot if asteroid is close enough, and turn if needed
        shoot = dist <= 5
        left_turn = angle > 0
        right_turn = angle < 0

        if shoot:
            self.shot_asteroids.append(target.id)

        return [True, left_turn, right_turn, shoot]

    # calculate distance and angle between spaceship and asteroid
    def get_dist_angle(self, spaceship, asteroid):
        dx = spaceship.x - asteroid.x 
        dy = spaceship.y - asteroid.y

        dist = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, -dx)) % 360
        angle = self.anglediff(spaceship.angle, angle)

        return (dist // config.speed["spaceship"], angle // config.angle_increment)

    # calculate angle difference (needed when turning)
    # https://stackoverflow.com/a/7869457
    def anglediff(self, source, target):
        angle = target - source
        angle = (angle + 180) % 360 - 180
        return angle

    # calculate steps needed to reach the asteroid
    def steps_to_reach(self, spaceship, asteroid):
        dist, angle = self.get_dist_angle(spaceship, asteroid)
        return dist + abs(angle)

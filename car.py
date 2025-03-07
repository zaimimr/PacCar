import pygame as py
import math
from utils import intersection_line, euchlidean_distance
import numpy as np
TIMER = 200
class Car:
    def __init__(self, pos=(90, 320), size=(15, 30), vel=(0, 0), acc=0.4, angle=0):
        # CONST
        self.car_img = py.transform.scale(py.image.load("car.png"), size)
        self.mask = py.mask.from_surface(self.car_img)
        self.color = "GREEN"
        self.size = py.math.Vector2(size)
        self.fric = 0.95
        self.turn_speed = 4.0
        self.ml_factor = 1
        # VAR
        self.pos = py.math.Vector2(pos)
        self.vel = py.math.Vector2(vel)
        self.max_vel = 15
        self.acc = acc
        self.angle = angle
        self.next_coin = 0
        self.sensors = {}
        self.generate_sensors()
        self.dead = False
        self.score = 0

        self.timer = TIMER

    def update(self):
        self.vel *= self.fric
        self.pos += self.vel

    def reset(self):
        self.pos = py.math.Vector2(100, 300)
        self.vel = py.math.Vector2(0, 0)
        self.angle = 0
        self.score = 0
        self.next_coin = 0
        self.dead = False
        self.generate_sensors()
        self.timer = TIMER

    def check_collision(self, mask):
        col = bool(self.mask.overlap(mask, (0 - self.pos.x, 0 - self.pos.y)))
        if col:
            self.color = "RED"
            self.dead = True
        else:
            self.color = "GREEN"
        return col

    def move(self, direction):
        self.vel.x += math.sin(math.radians(self.angle)) * self.acc * direction
        self.vel.y += math.cos(math.radians(self.angle)) * self.acc * direction
        self.vel.x = max(-self.max_vel, min(self.vel.x, self.max_vel))
        self.vel.y = max(-self.max_vel, min(self.vel.y, self.max_vel))

    def break_speed(self):
        self.vel *= 0.8

    def turn(self, direction):
        radius = 30

        # Calculate tangential velocity
        tangential_velocity = math.sqrt(self.vel.x**2 + self.vel.y**2)

        # Calculate angular velocity (in radians per update)
        angular_velocity = tangential_velocity / radius

        # Convert angular velocity to degrees per update and update angle
        self.angle += min(5, math.degrees(angular_velocity)) * direction

    def handle_action(self, action, track_mask, coins):
        # [UP, DOWN, LEFT, RIGHT]
        match action:
            case 0:
                self.move(-1)
            case 1:
                self.turn(1)
            case 2:
                self.turn(-1)
            case 3:
                self.break_speed()
            case 4:
                pass

        self.update()
        reward = 0
        game_over = False

        collided = self.check_collision(track_mask)
        if collided:
            reward = -self.ml_factor
            game_over = True
        # elif self.timer <= 0:
        #     reward = -self.ml_factor
        #     game_over = True
        else:
            reward = self.collect_coin(coins)
        self.score += reward
        self.timer -= 1
        return reward, game_over, self.score

    def draw(self, screen, best=False):
        if show_car := False:
            car_img = py.transform.rotate(self.car_img, self.angle)
            screen.blit(car_img, self.pos)
        if show_mask := True:
            if best:
                self.color = "YELLOW"
            mask_image = self.mask.to_surface(setcolor=self.color)
            old_rect = mask_image.get_rect(center=self.pos + self.size / 2)
            mask_image = py.transform.rotate(mask_image, self.angle)
            rect = mask_image.get_rect(center=old_rect.center)
            screen.blit(mask_image, rect)

    def draw_coin_vector(self, screen, coins):
        py.draw.line(
            screen, "BROWN", self.pos + self.size / 2, coins[self.next_coin], 1
        )

    def collect_coin(self, coins):
        if (
            euchlidean_distance(self.pos + self.size / 2, coins[self.next_coin])
            <= 30
        ):
            self.timer = TIMER
            self.next_coin = (self.next_coin + 1) % len(coins)
            return self.ml_factor
        return 0

    def generate_sensors(self, screen=None, lines=None):
        for i in range(0, 360, 30):
            size = 200
            start_point = self.pos + self.size / 2
            endpoint = (
                start_point[0] + math.sin(math.radians(i + self.angle)) * size,
                start_point[1] + math.cos(math.radians(i + self.angle)) * size,
            )
            closest_point = None
            closest_point_dist = None
            if lines:
                for line in lines:
                    point = intersection_line(start_point, endpoint, line[0], line[1])
                    if point:
                        if closest_point == None:
                            closest_point = point
                            closest_point_dist = euchlidean_distance(start_point, point)
                        elif (
                            euchlidean_distance(start_point, point) < closest_point_dist
                        ):
                            closest_point = point
            if screen:
                py.draw.line(screen, "RED", start_point, endpoint, 1)
                if closest_point:
                    py.draw.circle(screen, "BLUE", closest_point, 3)

            self.sensors[i] = closest_point

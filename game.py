import pygame
from car import Car
from utils import get_mask_outline, make_outline_loops, tranform_points_to_lines, euchlidean_distance
import numpy as np

FPS = 60
class Game:

    def __init__(self, local = False) -> None:
        pygame.init()
        self.local = local


        self.running = True
        self.screen = pygame.display.set_mode((1600, 900))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)

        self.track = pygame.transform.scale(pygame.image.load("track.png"), (1600, 900))
        self.track_mask = pygame.mask.from_surface(self.track)
        self.mask_image = self.track_mask.to_surface()

        print("GENERATING BOUDAIRIES")
        self.lines = [item for row in tranform_points_to_lines(make_outline_loops(get_mask_outline(self.track_mask))) for item in row]
        self.coins = [(101, 262), (100, 160), (150, 62), (205, 59), (304, 60), (380, 58), (482, 60), (569, 59), (672, 57), (773, 57), (868, 55), (994, 55), (1108, 54), (1222, 58), (1322, 55), (1435, 54), (1492, 121), (1500, 199), (1502, 281), (1502, 379), (1506, 460), (1505, 563), (1443, 640), (1360, 651), (1272, 657), (1240, 712), (1228, 774), (1213, 846), (1144, 846), (1060, 846), (981, 850), (899, 848), (815, 850), (777, 798), (773, 723), (754, 643), (819, 597), (919, 600), (1029, 589), (1112, 538), (1179, 465), (1197, 382), (1124, 338), (1041, 322), (948, 310), (855, 307), (758, 322), (654, 365), (586, 427), (582, 492), (485, 503), (376, 511), (237, 530), (98, 558), (97, 392), (97, 470)]


    def update(self, action, car):
        # create a random action, there can only be one 1 in the array
        # action = [0, 0, 0, 0]
        # action[random.randint(0, 3)] = 1
        return car.handle_action(action, self.track_mask, self.coins)

    def draw(self, car):
        self.screen.fill("gray")
        self.screen.blit(self.track, (0, 0))

        for i, c in enumerate(self.coins):
            pygame.draw.circle(self.screen, "yellow", c, 20)
            text = self.font.render(str(i), True, "black")
            text_rect = text.get_rect(center=c)
            self.screen.blit(text, text_rect)


    def get_state(self, car):
        state =  np.concatenate((
                # np.array([car.pos.x, car.pos.y]),
                np.array([car.vel.x/car.max_vel, car.vel.y/car.max_vel]),
                # np.array([car.acc]),
                # np.array([car.angle]),
                # np.array([car.dead]),
                # np.array([car.score]),
                np.array([(200-euchlidean_distance(self.coins[car.next_coin], car.pos+car.size/2)/200)]),
                np.array([(250-euchlidean_distance(v, (car.pos+car.size/2)))/250 if v else 250 for v in car.sensors.values()])
                ))
        return state

    def run(self, car):
        self.draw(car)
        if self.local:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                car.move(-1)
            if keys[pygame.K_RIGHT]:
                car.turn(-1)
            if keys[pygame.K_LEFT]:
                car.turn(1)
            car.update()

    def flip(self):
        pygame.display.flip()
        dt = self.clock.tick(FPS) / 1000 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        if not self.running:
            pygame.quit()
        
if __name__ == "__main__":
    game = Game(local=True)
    car = Car()
    while game.running:
        game.run(car)
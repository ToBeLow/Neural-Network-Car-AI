import numpy
import pygame
import math
import time
from utils import *
from random import *
import sys


# Grass background scaled x2.5
GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)

# Track image at 90% of original size (same for the masks and border)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)
TRACK_MASK = scale_image(pygame.image.load("imgs/track-mask.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK=pygame.mask.from_surface(TRACK_BORDER)

# Finish image
FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK=pygame.mask.from_surface(FINISH)

# Cars images at 55%
RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)
PURPLE_CAR = scale_image(pygame.image.load("imgs/purple-car.png"), 0.55)
car_width,car_height=PURPLE_CAR.get_size()

### Half car-width and half car-height
HALF_WIDTH=car_width/2
HALF_HEIGHT=car_height/2
CAR_SIZE=HALF_WIDTH,HALF_HEIGHT

# Get the size of the track image
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()

# The window has the size of the track image
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Basic pygame setup
pygame.display.set_caption("Racing Game!")
pygame.font.init()
MAIN_FONT = pygame.font.SysFont("comicsans", 44)

# Constants for NEAT training
FPS = 60
WHITE = (255, 255, 255, 255)
FINISH_POSITION=(130,250)
images=[(GRASS,(0,0)),(TRACK,(0,0)),(FINISH,FINISH_POSITION),(TRACK_BORDER,(0,0))]
RADAR_ANGLES = [-90, -45, -30, -15, 0, 15, 30, 45, 90]
MAX_RADAR_RANGE = 200

# Function to render the game state, used in both training and replay
def draw(win, images, player_car, computer_car, game_info):
    # Draw the background and track
    for img,pos in images:
        win.blit(img,pos)
    # Draw the level, time, and velocity text
    level_text=MAIN_FONT.render(f'Level {game_info.level}',1,(255,255,255))
    win.blit(level_text,(10,HEIGHT-level_text.get_height()-90))

    time_text=MAIN_FONT.render(f'Time {game_info.get_level_time()}',1,(255,255,255))
    win.blit(time_text,(10,HEIGHT-time_text.get_height()-50))

    velocity_text=MAIN_FONT.render(f'Vel {round(computer_car.vel,1)} px/s',1,(255,255,255))
    win.blit(velocity_text,(10,HEIGHT-velocity_text.get_height()-10))

    # Draw the cars on top of everything else
    player_car.draw(win)
    computer_car.draw(win)
    pygame.display.update()

# Original class from the Racing Car game
class AbstractCar:

    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x,self.y=self.START_POS
        self.acceleration=1

    def rotate(self, left=False, right=False):
        if left and right:
            pass
        elif left:
            self.angle += self.rotation_vel + choice(range(-1,1))
        elif right:
            self.angle -= self.rotation_vel +  choice(range(-1,1))
        self.angle = int(self.angle) % 360

    def move_forward(self):
            # FIX: If we just bounced (negative velocity), reset to 0
            # so we actually move away from the wall immediately.
            if self.vel < 0:
                self.vel = 0
            self.vel = min(self.vel + self.acceleration, self.max_vel)
            self.move()

    def move_backwards(self):
        # FIX: Use max() instead of min() so it doesn't instantly snap to -2
        self.vel = max(self.vel - self.acceleration, -self.max_vel//2)
        self.move()

    # primeiro calculamos o  ângulo em radianos
    # Calculamos o deslocamento em x e y através da trignometra
    # actualizamos x e y mas subtraindo devido aos pontos cardeais do pygame e da corrida de carros
    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal


    def collide(self, mask, x=0, y=0):
            # FIX: Rotate the mask to match the car's visual rotation
            rotated_image = pygame.transform.rotate(self.img, self.angle)
            car_mask = pygame.mask.from_surface(rotated_image)

            # Center the mask correctly so the hitbox stays aligned with the image
            rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
            offset = (int(rect.x - x), int(rect.y - y))
            return mask.overlap(car_mask, offset)

    def reset(self):
        self.x,self.y=self.START_POS
        self.angle=0
        self.vel=0

    def bounce(self):
        self.vel=-self.vel
        self.move()

    # x,y will be the center of the car
    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
        x, y = int(self.x + CAR_SIZE[0]), int(self.y + CAR_SIZE[1])
        dx, dy = CAR_SIZE[0], CAR_SIZE[1]
        alfa = (self.angle) * math.pi / 180
        mrot = numpy.array([[math.cos(alfa), -math.sin(alfa)], [math.sin(alfa), math.cos(alfa)]])
        pts = numpy.array([[-dx, -dy], [dx, -dy], [-dx, dy], [dx, dy]])
        npts = numpy.dot(pts, mrot)
        for i in npts:
            if 0<=i[0]+x<WIDTH and 0<=i[1]+y<HEIGHT:
                pygame.draw.circle(win, TRACK_MASK.get_at((int(i[0] + x), int(i[1] + y))), \
                                                          (int(i[0] + x), int(i[1] + y)),2, 2)

    def turn_left(self):
        self.rotate(left=True)
        self.move_forward()

    def turn_right(self):
        self.rotate(right=True)
        self.move_forward()

# Original class from the Racing Car game for the player-controlled car, unchanged
class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (180, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

# New class for the computer-controlled car with radar sensors and NEAT integration
class RadarCar(AbstractCar):
    IMG = PURPLE_CAR
    START_POS = (150, 200)

    def __init__(self, max_vel, rotation_vel):
        super().__init__(max_vel, rotation_vel)
        # Initialize radar distances and endpoints for visualization
        self.radars = [0.0] * len(RADAR_ANGLES)
        self.radar_endpoints = [(0, 0)] * len(RADAR_ANGLES)
        # So in frame 1 they don't point to 0,0
        self.update_radars()

    def cast_radar(self, angle_offset):
        # Top left + half width/height
        cx = self.x + CAR_SIZE[0]
        cy = self.y + CAR_SIZE[1]
        # Set ray direction (angle) in rad
        total_rad = math.radians(self.angle + angle_offset)
        # Get unit vector for the ray
        dx = -math.sin(total_rad)
        dy = -math.cos(total_rad)

        # Cast the ray every pixel until wall or max range
        dist = MAX_RADAR_RANGE
        for d in range(1, MAX_RADAR_RANGE + 1):
            px = int(cx + dx * d)
            py = int(cy + dy * d)
            # If it's outside of bounds, stop right before it (avoids a crash)
            if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
                dist = d - 1
                break
            # If the ray hits a wall, return the distance
            if TRACK_BORDER_MASK.get_at((px, py)):
                dist = d
                break

        # Get the distance and coordinates of the pixel in question
        end_x = int(cx + dx * dist)
        end_y = int(cy + dy * dist)
        # Draw function needs them separate, so we return them both
        return float(dist), end_x, end_y

    def update_radars(self, noise=True):
        # Cast the radar for every angle
        for i, angle_offset in enumerate(RADAR_ANGLES):
            dist, end_x, end_y = self.cast_radar(angle_offset)
            # Add 2 pixels of Gaussian noise to the distance reading
            if noise:
                dist = max(0.0, min(float(MAX_RADAR_RANGE), gauss(dist, 2.0)))
            self.radars[i] = dist
            self.radar_endpoints[i] = (end_x, end_y)

    def step(self, net=None, noise=True):
        self.update_radars(noise)
        if net is None:
            self.move_forward()
            return

        # Normalize inputs (radar distances) to [0, 1] for the network
        inputs = [r / MAX_RADAR_RANGE for r in self.radars]
        output = net.activate(inputs)

        # Output[0]: angular acceleration, tanh in [-1,1] scaled to rotation_vel bounds
        # Output[1]: speed change, tanh in [-1,1] scaled to acceleration bounds
        delta_angle = output[0] * self.rotation_vel
        delta_speed = output[1] * self.acceleration

        # Add actuator noise (section 3.3)
        if noise:
            delta_angle = gauss(delta_angle, 0.3)
            delta_speed = gauss(delta_speed, 0.05)

        # Clamp to car limits
        delta_angle = max(-self.rotation_vel, min(self.rotation_vel, delta_angle))
        delta_speed = max(-self.acceleration, min(self.acceleration, delta_speed))

        self.angle = int(self.angle + delta_angle) % 360
        self.vel = max(-self.max_vel // 2, min(self.max_vel, self.vel + delta_speed))
        self.move()

    def draw(self, win):
        super().draw(win)
        # Center of the car
        cx = int(self.x + CAR_SIZE[0])
        cy = int(self.y + CAR_SIZE[1])
            # Draw a circle on the pixel and a line to it
        for end_x, end_y in self.radar_endpoints:
            pygame.draw.line(win, (0, 255, 0), (cx, cy), (end_x, end_y), 1)
            pygame.draw.circle(win, (255, 0, 0), (end_x, end_y), 3)

# Original move function for the player car, unchanged
def move_player(player_car):
    keys=pygame.key.get_pressed()
    moved=False
    if keys[pygame.K_a]:
        player_car.rotate(left=True)
    elif keys[pygame.K_d]:
        player_car.rotate(right=True)
    elif keys[pygame.K_w]:
        moved=True
        player_car.move_forward()
    elif keys[pygame.K_s]:
        moved=True
        player_car.move_backwards()
    if not moved:
        player_car.reduce_speed()

# Original collision handling function, unchanged
def handle_collision(player_car, computer_car, game_info):
    if player_car.collide(TRACK_BORDER_MASK) != None:
        player_car.bounce()

    if computer_car.collide(TRACK_BORDER_MASK) != None:
        computer_car.bounce()

    computer_finish_poi_collide = computer_car.collide(FINISH_MASK, *FINISH_POSITION)
    if computer_finish_poi_collide != None:
        blit_text_center(WIN,MAIN_FONT,"YOU LOST!")
        pygame.display.update()
        pygame.time.wait(5000)
        game_info.reset()
        player_car.reset()
        computer_car.reset()
        return True

    player_finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION)
    if player_finish_poi_collide != None:
        if player_finish_poi_collide[1] == 0:
            player_car.bounce()
        else:
            player_car.reset()
            game_info.next_level()
            computer_car.reset()
            return True
    return False

# Unchanged GameInfo class
class GameInfo:
    LEVELS = 10

    def __init__(self, level=1):
        self.level = level
        self.started = False
        self.level_start_time = 0

    def next_level(self):
        self.level += 1
        self.started = False

    def reset(self):
        self.level = 1
        self.started = False
        self.level_start_time = 0

    def game_finished(self):
        return self.level > self.LEVELS

    def start_level(self):
        self.started = True
        self.level_start_time = time.time()

    def get_level_time(self):
        if not self.started:
            return 0
        return round(time.time() - self.level_start_time)

# New runnable for the Radar car, self-explanatory
if __name__ == "__main__":
    player_car = PlayerCar(4, 4)
    computer_car = RadarCar(4, 4)
    game_info = GameInfo()
    run = True
    clock = pygame.time.Clock()

    while run:
        clock.tick(FPS)
        draw(WIN, images, player_car, computer_car, game_info)

        while not game_info.started:
            blit_text_center(WIN, MAIN_FONT, f'Press any key to start level {game_info.level}!')
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run=False
                    pygame.quit()
                    sys.exit()
                if event.type==pygame.KEYDOWN:
                    game_info.start_level()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run=False
                break

        move_player(player_car)
        computer_car.step()

        if handle_collision(player_car, computer_car, game_info):
            draw(WIN, images, player_car, computer_car, game_info)

        if game_info.game_finished():
            blit_text_center(WIN, MAIN_FONT, "YOU WON!")
            pygame.time.wait(5000)
            game_info.reset()
            player_car.reset()
            computer_car.reset()

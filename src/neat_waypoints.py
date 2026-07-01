import math
import os
import pickle
import sys
import traceback
from random import gauss

import neat
import pygame

from utils import blit_rotate_center, blit_text_center, scale_image


GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.55)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Waypoints - Racing Car")

pygame.font.init()
MAIN_FONT = pygame.font.SysFont("comicsans", 34)
DASH_FONT = pygame.font.SysFont("consolas", 17)

FPS = 60
TRAINING_SPEED = 4
GENERATIONS = 50
FINISH_POSITION = (130, 250)
MAX_FRAMES = 1800
WAYPOINT_REACHED_DISTANCE = 35
MIN_WAYPOINTS = 2
CONTROL_SMOOTHING = 0.70
ANGLE_NOISE = 0.12
SPEED_NOISE = 0.02

CAR_WIDTH, CAR_HEIGHT = GREEN_CAR.get_size()
CAR_HALF_WIDTH = CAR_WIDTH / 2
CAR_HALF_HEIGHT = CAR_HEIGHT / 2
MAX_DISTANCE = math.hypot(WIDTH, HEIGHT)

generation = 0
best_fitness = 0
best_genome = None
training_waypoints = []
last_generation_status = ""

SRC_DIR    = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SRC_DIR, "models")
WINNER_WAYPOINTS_PATH = os.path.join(MODELS_DIR, "winner_waypoints.pkl")
WAYPOINTS_PATH        = os.path.join(MODELS_DIR, "waypoints.pkl")


def save_pickle(path, value):
    with open(path, "wb") as f:
        pickle.dump(value, f)


def save_waypoints(waypoints):
    save_pickle(WAYPOINTS_PATH, list(waypoints))
    print(f"Waypoints saved to {WAYPOINTS_PATH}")


def save_waypoint_winner(genome, label="best-so-far"):
    save_pickle(WINNER_WAYPOINTS_PATH, genome)
    print(f"Saved waypoint {label} to {WINNER_WAYPOINTS_PATH}")


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def normalize_angle(angle):
    return ((angle + 180) % 360 - 180) / 180.0


def angle_to_point(car_angle, car_x, car_y, point):
    dx = point[0] - (car_x + CAR_HALF_WIDTH)
    dy = point[1] - (car_y + CAR_HALF_HEIGHT)
    desired = math.degrees(math.atan2(dx, dy)) % 360
    return (car_angle - desired + 180) % 360 - 180


def segment_heading(a, b):
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360


class WaypointNeatCar:
    START_POS = (150, 200)
    IMG = GREEN_CAR

    def __init__(self, max_vel, rotation_vel, waypoints):
        self.img = self.IMG
        self.max_vel = max_vel
        self.rotation_vel = rotation_vel
        self.acceleration = 1
        self.waypoints = list(waypoints)
        self.reset()

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0
        self.current_idx = 0
        self.reached_count = 0
        self.distance_travelled = 0.0
        self.prev_pos = (self.x, self.y)
        self.closest_distance = self.distance_to_current_waypoint()
        self.prev_delta_angle = 0.0
        self.prev_delta_speed = 0.0

    def rotate(self, left=False, right=False):
        if left and right:
            pass
        elif left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel
        self.angle = int(self.angle) % 360

    def move_forward(self):
        if self.vel < 0:
            self.vel = 0
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backwards(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel // 2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def bounce(self):
        self.vel = -self.vel
        self.move()

    def turn_left(self):
        self.rotate(left=True)
        self.move_forward()

    def turn_right(self):
        self.rotate(right=True)
        self.move_forward()

    def collide(self, mask, x=0, y=0):
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        car_mask = pygame.mask.from_surface(rotated_image)
        rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        offset = (int(rect.x - x), int(rect.y - y))
        return mask.overlap(car_mask, offset)

    def distance_to_current_waypoint(self):
        if self.current_idx >= len(self.waypoints):
            return 0.0
        target = self.waypoints[self.current_idx]
        cx = self.x + CAR_HALF_WIDTH
        cy = self.y + CAR_HALF_HEIGHT
        return math.hypot(target[0] - cx, target[1] - cy)

    def current_target(self):
        if self.current_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.current_idx]

    def waypoint_features(self):
        target = self.current_target()
        if target is None:
            return [0.0, 0.0, 0.0, self.vel / self.max_vel, 1.0]

        angle_error = normalize_angle(angle_to_point(self.angle, self.x, self.y, target))
        distance = clamp(self.distance_to_current_waypoint() / MAX_DISTANCE, 0.0, 1.0)

        if len(self.waypoints) > 1:
            previous_idx = max(0, self.current_idx - 1)
            next_idx = min(len(self.waypoints) - 1, self.current_idx + 1)
            current_heading = segment_heading(self.waypoints[previous_idx], self.waypoints[self.current_idx])
            next_heading = segment_heading(self.waypoints[self.current_idx], self.waypoints[next_idx])
            next_curve = normalize_angle(next_heading - current_heading)
        else:
            next_curve = 0.0

        speed = clamp(self.vel / self.max_vel, -0.5, 1.0)
        progress = self.current_idx / max(1, len(self.waypoints) - 1)
        return [angle_error, distance, next_curve, speed, progress]

    def step(self, net):
        before = (self.x, self.y)
        outputs = net.activate(self.waypoint_features())

        delta_angle = clamp(outputs[0], -1.0, 1.0) * self.rotation_vel
        delta_speed = clamp(outputs[1], -1.0, 1.0) * self.acceleration

        delta_angle = gauss(delta_angle, ANGLE_NOISE)
        delta_speed = gauss(delta_speed, SPEED_NOISE)

        delta_angle = clamp(delta_angle, -self.rotation_vel, self.rotation_vel)
        delta_speed = clamp(delta_speed, -self.acceleration, self.acceleration)
        delta_angle = CONTROL_SMOOTHING * self.prev_delta_angle + (1 - CONTROL_SMOOTHING) * delta_angle
        delta_speed = CONTROL_SMOOTHING * self.prev_delta_speed + (1 - CONTROL_SMOOTHING) * delta_speed
        self.prev_delta_angle = delta_angle
        self.prev_delta_speed = delta_speed

        self.angle = int(self.angle + delta_angle) % 360
        self.vel = clamp(self.vel + delta_speed, -self.max_vel // 2, self.max_vel)
        self.move()

        dx = self.x - before[0]
        dy = self.y - before[1]
        self.distance_travelled += math.hypot(dx, dy)
        self.prev_pos = before

        target_distance = self.distance_to_current_waypoint()
        improved = max(0.0, self.closest_distance - target_distance)
        self.closest_distance = min(self.closest_distance, target_distance)

        reached = False
        if self.current_idx < len(self.waypoints) and target_distance < WAYPOINT_REACHED_DISTANCE:
            self.current_idx += 1
            self.reached_count += 1
            self.closest_distance = self.distance_to_current_waypoint()
            reached = True

        return improved, reached

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)


def draw_scene(cars=None, waypoints=None, generation_text=None, extra_lines=None):
    WIN.blit(GRASS, (0, 0))
    WIN.blit(TRACK, (0, 0))
    WIN.blit(FINISH, FINISH_POSITION)

    waypoints = waypoints or []
    for i, waypoint in enumerate(waypoints):
        color = (0, 255, 0) if i == 0 else (255, 255, 0)
        pygame.draw.circle(WIN, color, waypoint, 6)
        if i > 0:
            pygame.draw.line(WIN, (0, 180, 80), waypoints[i - 1], waypoint, 2)

    for car in cars or []:
        car.draw(WIN)

    lines = []
    if generation_text:
        lines.append(generation_text)
    if extra_lines:
        lines.extend(extra_lines)

    for i, line in enumerate(lines):
        text = DASH_FONT.render(line, True, (255, 255, 255))
        WIN.blit(text, (10, HEIGHT - 24 * (len(lines) - i)))

    pygame.display.update()


def wait_for_key_or_quit(message, extra_lines=None):
    clock = pygame.time.Clock()
    while True:
        clock.tick(FPS)
        draw_scene(
            waypoints=training_waypoints,
            generation_text=message,
            extra_lines=(extra_lines or []) + ["Press any key to close."],
        )
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                return


def collect_waypoints():
    waypoints = []
    clock = pygame.time.Clock()

    while True:
        clock.tick(FPS)
        draw_scene(
            waypoints=waypoints,
            extra_lines=[
                "Click to add waypoints. Backspace removes last.",
                "Press Space/Enter to start NEAT training.",
            ],
        )

        if len(waypoints) < MIN_WAYPOINTS:
            blit_text_center(WIN, MAIN_FONT, f"Add at least {MIN_WAYPOINTS} waypoints")
            pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                waypoints.append(pygame.mouse.get_pos())
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE and waypoints:
                    waypoints.pop()
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN) and len(waypoints) >= MIN_WAYPOINTS:
                    return waypoints


def eval_genomes(genomes, config):
    global generation, best_fitness, best_genome, last_generation_status
    generation += 1

    nets = []
    cars = []
    active_genomes = []

    for _, genome in genomes:
        genome.fitness = 0
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        cars.append(WaypointNeatCar(4, 5, training_waypoints))
        active_genomes.append(genome)

    clock = pygame.time.Clock()
    total = len(cars)
    stall_snapshots = [(car.x, car.y) for car in cars]
    gen_start_ms = pygame.time.get_ticks()
    end_reason = "frame limit reached"

    print(f"\n--- Generation {generation} started with {total} cars ---")

    for frame in range(MAX_FRAMES):
        clock.tick(FPS * TRAINING_SPEED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                current_best = max((genome for _, genome in genomes), key=lambda genome: genome.fitness or 0)
                save_waypoints(training_waypoints)
                save_waypoint_winner(best_genome or current_best, "best-before-quit")
                pygame.quit()
                sys.exit()

        done = set()
        eliminated_by_collision = 0
        eliminated_by_completion = 0
        eliminated_by_stall = 0

        for i, car in enumerate(cars):
            improved, reached = car.step(nets[i])
            active_genomes[i].fitness += car.vel * 0.2
            active_genomes[i].fitness += improved * 0.6

            if reached:
                active_genomes[i].fitness += 500 + car.current_idx * 100

            if car.collide(TRACK_BORDER_MASK) is not None:
                active_genomes[i].fitness -= 75
                eliminated_by_collision += 1
                done.add(i)
            elif car.current_idx >= len(car.waypoints):
                active_genomes[i].fitness += 2000 + (MAX_FRAMES - frame) * 2
                eliminated_by_completion += 1
                done.add(i)
            elif car.collide(FINISH_MASK, *FINISH_POSITION) is not None and car.reached_count >= len(car.waypoints):
                active_genomes[i].fitness += 1000
                eliminated_by_completion += 1
                done.add(i)

        if frame % 90 == 89:
            for i, (car, snap) in enumerate(zip(cars, stall_snapshots)):
                moved = math.hypot(car.x - snap[0], car.y - snap[1])
                if moved < 20:
                    active_genomes[i].fitness -= 40
                    eliminated_by_stall += 1
                    done.add(i)

        for i in sorted(done, reverse=True):
            cars.pop(i)
            nets.pop(i)
            active_genomes.pop(i)
            stall_snapshots.pop(i)

        if frame % 90 == 89:
            stall_snapshots = [(car.x, car.y) for car in cars]

        if not cars:
            reasons = []
            if eliminated_by_completion:
                reasons.append(f"completed: {eliminated_by_completion}")
            if eliminated_by_collision:
                reasons.append(f"hit border: {eliminated_by_collision}")
            if eliminated_by_stall:
                reasons.append(f"stalled: {eliminated_by_stall}")
            end_reason = "all cars eliminated"
            if reasons:
                end_reason += " (" + ", ".join(reasons) + ")"
            break

        current_best = max(g.fitness for _, g in genomes)
        best_fitness = max(best_fitness, current_best)
        draw_scene(
            cars=cars,
            waypoints=training_waypoints,
            generation_text=f"Gen: {generation}  Alive: {len(cars)}/{total}",
            extra_lines=[
                f"Best ever: {int(best_fitness)}",
                f"Gen best: {int(current_best)}",
                f"Time: {pygame.time.get_ticks() - gen_start_ms}ms",
                last_generation_status,
            ],
        )

    for _, genome in genomes:
        if genome.fitness is None or genome.fitness < 0:
            genome.fitness = 0

    current_best = max(genome.fitness for _, genome in genomes)
    best_fitness = max(best_fitness, current_best)
    generation_best = max((genome for _, genome in genomes), key=lambda genome: genome.fitness)
    if best_genome is None or generation_best.fitness >= best_fitness:
        best_genome = generation_best
        save_waypoint_winner(best_genome, f"generation-{generation}")

    last_generation_status = f"Gen {generation} ended: {end_reason}. Best: {int(current_best)}"
    print(last_generation_status)

    draw_scene(
        waypoints=training_waypoints,
        generation_text=f"Gen {generation}/{GENERATIONS} complete",
        extra_lines=[
            end_reason,
            f"Generation best: {int(current_best)}",
            f"Best ever: {int(best_fitness)}",
        ],
    )
    pygame.time.wait(350)


def run(config_path, waypoints):
    global training_waypoints
    training_waypoints = list(waypoints)
    save_waypoints(training_waypoints)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes, n=GENERATIONS)
    print("Best genome:", winner)

    save_waypoint_winner(winner, "final-winner")
    save_waypoints(training_waypoints)
    wait_for_key_or_quit(
        "Training finished",
        [
            f"Best fitness: {int(best_fitness)}",
            f"Winner saved: {WINNER_WAYPOINTS_PATH}",
            f"Waypoints saved: {WAYPOINTS_PATH}",
        ],
    )
    return winner


if __name__ == "__main__":
    try:
        selected_waypoints = collect_waypoints()
        config_file = os.path.join(os.path.dirname(__file__), "neat_waypoint_config.txt")
        run(config_file, selected_waypoints)
    except Exception as exc:
        traceback.print_exc()
        training_waypoints = training_waypoints or []
        wait_for_key_or_quit(
            "Training stopped because of an error",
            [
                type(exc).__name__,
                str(exc)[:120],
                "The full traceback was printed in the terminal.",
            ],
        )

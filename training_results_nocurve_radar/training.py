import neat
import pygame
import sys
import os
import math
import pickle
import time
import json
from game import (
    RadarCar,
    WIN, HEIGHT, MAIN_FONT, FPS,
    images, FINISH_MASK, FINISH_POSITION, TRACK_BORDER_MASK
)

generation    = 0
best_fitness  = 0
first_lap_gen = None  # Generation where the first car completed a valid lap

# 1 = normal (60FPS), 2 = double speed (120FPS), 0 = uncapped FPS
TRAINING_SPEED = 0
# Render every N frames (1 = every frame, 10 = every 10th, 0 = no render)
RENDER_EVERY = 0

MAX_FRAMES   = 1500  # Generation time limit, unrelated to the bonus formula
BASE_BONUS   = 1000  # Flat reward for any valid lap completion
TIME_BONUS   = 5000  # Max extra reward for very fast laps
DECAY_FRAMES = 400   # Steepness of the speed reward


def finish_bonus(frame):
    # Exponential decay on frame count, saving a second at 15s matters far more than at 20s
    return BASE_BONUS + TIME_BONUS * math.exp(-frame / DECAY_FRAMES)


# Self-explanatory drawing function
def draw_training(win, cars, ge, generation, total, elapsed_ms):
    for img, pos in images:
        win.blit(img, pos)

    for car in cars:
        car.draw(win)

    current_best = max(g.fitness for g in ge)
    last_score   = ge[0].fitness if len(cars) == 1 else None

    gen_text   = MAIN_FONT.render(f'Gen: {generation}',              1, (255, 255, 255))
    time_text  = MAIN_FONT.render(f'Time: {elapsed_ms}ms',           1, (255, 255, 255))
    alive_text = MAIN_FONT.render(f'Alive: {len(cars)}/{total}',     1, (255, 255, 255))
    best_text  = MAIN_FONT.render(f'Best ever: {int(best_fitness)}', 1, (255, 255, 255))
    cur_text   = MAIN_FONT.render(f'Gen best: {int(current_best)}',  1, (255, 255, 255))
    win.blit(gen_text,   (10, HEIGHT - gen_text.get_height()   - 170))
    win.blit(time_text,  (10, HEIGHT - time_text.get_height()  - 130))
    win.blit(best_text,  (10, HEIGHT - best_text.get_height()  - 90))
    win.blit(cur_text,   (10, HEIGHT - cur_text.get_height()   - 50))
    win.blit(alive_text, (10, HEIGHT - alive_text.get_height() - 10))

    if last_score is not None:
        last_text = MAIN_FONT.render(f'Last: {int(last_score)}', 1, (255, 255, 0))
        win.blit(last_text, (10, HEIGHT - last_text.get_height() - 210))

    pygame.display.update()


def eval_genomes(genomes, config):
    global generation, best_fitness, first_lap_gen
    generation += 1

    nets = []
    cars = []
    ge   = []

    # Create a car for each genome
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(RadarCar(4, 4))
        genome.fitness = 0
        ge.append(genome)

    clock          = pygame.time.Clock()
    total          = len(cars)
    distances      = [0.0] * total
    prev_pos       = [(car.x, car.y) for car in cars]  # previous position
    stall_snapshot = [(car.x, car.y) for car in cars]  # last checked car position
    gen_start_ms   = pygame.time.get_ticks()

    # Main loop for the generation
    for frame in range(MAX_FRAMES):
        clock.tick(FPS * TRAINING_SPEED)

        # Just so we can exit mid training
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Step each car, reward forward movement, track total distance traveled
        for i, car in enumerate(cars):
            car.step(nets[i])
            ge[i].fitness += car.vel
            dx = car.x - prev_pos[i][0]
            dy = car.y - prev_pos[i][1]
            distances[i] += math.sqrt(dx * dx + dy * dy)
            prev_pos[i] = (car.x, car.y)

        # Remove cars that hit the border or completed a lap
        done = set()
        for i, car in enumerate(cars):
            if car.collide(TRACK_BORDER_MASK) is not None:
                done.add(i)
            elif car.collide(FINISH_MASK, *FINISH_POSITION) is not None:
                if distances[i] >= 2000:
                    ge[i].fitness += finish_bonus(frame)
                    if first_lap_gen is None:
                        first_lap_gen = generation
                done.add(i)

        # Every 60 frames, kill cars that haven't moved 30px from their last snapshot
        if frame % 60 == 59:
            for i, (car, snap) in enumerate(zip(cars, stall_snapshot)):
                if math.sqrt((car.x - snap[0])**2 + (car.y - snap[1])**2) < 30:
                    done.add(i)

        for i in sorted(done, reverse=True):
            cars.pop(i)
            nets.pop(i)
            ge.pop(i)
            distances.pop(i)
            prev_pos.pop(i)
            stall_snapshot.pop(i)

        # Reset stall snapshot for the next window after removals
        if frame % 60 == 59:
            stall_snapshot = [(car.x, car.y) for car in cars]

        if not cars:
            break

        best_fitness = max(best_fitness, max(g.fitness for g in ge))
        if RENDER_EVERY > 0 and frame % RENDER_EVERY == 0:
            draw_training(WIN, cars, ge, generation, total, pygame.time.get_ticks() - gen_start_ms)

    # NEAT expects non-negative fitness values
    for _, genome in genomes:
        if genome.fitness is None or genome.fitness < 0:
            genome.fitness = 0


def run(config_path, label, pop_size, n_gens):
    global generation, best_fitness, first_lap_gen
    generation    = 0
    best_fitness  = 0
    first_lap_gen = None

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    config.pop_size = pop_size

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    src = os.path.dirname(__file__)

    train_start = time.time()
    winner = population.run(eval_genomes, n=n_gens)
    training_time_s = round(time.time() - train_start, 2)

    print(f"[{label}] Training done in {training_time_s}s — best fitness: {best_fitness}")

    # Save winner genome
    with open(os.path.join(src, f"{label}_winner.pkl"), "wb") as f:
        pickle.dump(winner, f)

    # Extract per-generation fitness stats
    fitness_max  = [c.fitness for c in stats.most_fit_genomes]
    fitness_mean = stats.get_fitness_mean()

    summary = {
        "label":           label,
        "pop_size":        pop_size,
        "n_gens":          n_gens,
        "training_time_s": training_time_s,
        "first_lap_gen":   first_lap_gen,
        "best_fitness":    round(best_fitness, 2),
        "fitness_max":     [round(f, 2) for f in fitness_max],
        "fitness_mean":    [round(f, 2) for f in fitness_mean],
        "replay_laps_ms":  []
    }

    # Throw everything into a json when done for easy analysis later
    with open(os.path.join(src, f"{label}_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{label}] Stats saved.")

# Run format for ease of batch execution
RUNS = [
    ("pop50_gen50_crn",   50,  50),
    ("pop50_gen100_crn",  50,  100),
    ("pop50_gen150_crn",  50,  150),
    ("pop100_gen50_crn",  100, 50),
    ("pop100_gen100_crn", 100, 100),
    ("pop100_gen150_crn", 100, 150),
    ("pop150_gen50_crn",  150, 50),
    ("pop150_gen100_crn", 150, 100),
    ("pop150_gen150_crn", 150, 150),
]

# Simple runnable, just gotta change label
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "neat_config.txt")
    for label, pop_size, n_gens in RUNS:
        print(f"\n=== Starting run: {label} ===")
        run(config_path, label, pop_size, n_gens)

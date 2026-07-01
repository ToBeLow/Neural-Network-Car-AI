import neat
import pygame
import pickle
import os
import sys
import json
from game import (
    RadarCar,
    WIN, HEIGHT, MAIN_FONT, FPS,
    images, FINISH_MASK, FINISH_POSITION, TRACK_BORDER_MASK,
    blit_text_center
)

# Draw function for the replays, much simpler than the training one
def draw_replay(win, car, lap, elapsed_ms):
    for img, pos in images:
        win.blit(img, pos)

    car.draw(win)

    lap_text  = MAIN_FONT.render(f'Lap: {lap}',           1, (255, 255, 255))
    time_text = MAIN_FONT.render(f'Time: {elapsed_ms}ms', 1, (255, 255, 255))
    vel_text  = MAIN_FONT.render(f'Vel: {car.vel:.1f}',   1, (255, 255, 255))
    win.blit(lap_text,  (10, HEIGHT - lap_text.get_height()  - 90))
    win.blit(time_text, (10, HEIGHT - time_text.get_height() - 50))
    win.blit(vel_text,  (10, HEIGHT - vel_text.get_height()  - 10))

    pygame.display.update()


def replay(config_path, label):
    src = os.path.dirname(__file__)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Get the best fit individual from the pkl file
    with open(os.path.join(src, f"{label}_winner.pkl"), "rb") as f:
        genome = pickle.load(f)

    # Create the network with that genome and the config
    # Feed forward because we don't need recurrency for a single car
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    car = RadarCar(4, 4)

    clock     = pygame.time.Clock()
    lap       = 1
    lap_start = pygame.time.get_ticks()
    run       = True

    # Simple run loop, identical to the training one but without extra logic
    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False

        car.step(net)
        elapsed = pygame.time.get_ticks() - lap_start
        draw_replay(WIN, car, lap, elapsed)

        if car.collide(TRACK_BORDER_MASK) is not None:
            blit_text_center(WIN, MAIN_FONT, "Crashed! Restarting...")
            pygame.display.update()
            pygame.time.wait(2000)
            car.reset()
            car.update_radars()
            lap_start = pygame.time.get_ticks()

        elif car.collide(FINISH_MASK, *FINISH_POSITION) is not None:
            blit_text_center(WIN, MAIN_FONT, f"Lap {lap} — {elapsed}ms")
            pygame.display.update()
            pygame.time.wait(2000)

            # Append lap time to the run's stats file
            stats_path = os.path.join(src, f"{label}_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r", encoding="utf-8-sig") as f:
                    stats = json.load(f)
                stats["replay_laps_ms"].append(elapsed)
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)

            lap += 1
            car.reset()
            car.update_radars()
            lap_start = pygame.time.get_ticks()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    # Change the label to replay a different run
    LABEL = "pop50_gen100_crn"

    src = os.path.dirname(__file__)
    replay(
        os.path.join(src, "neat_config.txt"),
        LABEL
    )

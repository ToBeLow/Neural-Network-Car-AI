import math
import neat
import os
import pickle
import sys

import pygame

from neat_waypoints import WaypointNeatCar
from game import (
    RadarCar,
    WIN, HEIGHT, FPS,
    images, FINISH_MASK, FINISH_POSITION, TRACK_BORDER_MASK,
)

DASH_FONT   = pygame.font.SysFont("consolas", 17)

src = os.path.dirname(__file__)


def load_net(pkl_path, config_path):
    with open(pkl_path, "rb") as f:
        genome = pickle.load(f)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    return neat.nn.FeedForwardNetwork.create(genome, config)


def draw(radar_car, waypoint_car, waypoints, frame, status=None):
    for img, pos in images:
        WIN.blit(img, pos)

    for i, wp in enumerate(waypoints):
        pygame.draw.circle(WIN, (0, 255, 0), wp, 6)
        if i > 0:
            pygame.draw.line(WIN, (0, 150, 70), waypoints[i - 1], wp, 2)

    radar_car.draw(WIN)
    waypoint_car.draw(WIN)

    lines = [
        f"Frame: {frame}",
        f"Radar vel: {radar_car.vel:.1f}",
        f"Waypoint vel: {waypoint_car.vel:.1f}  wp {waypoint_car.current_idx}/{len(waypoints)}",
    ]
    if status:
        lines.append(status)

    for i, line in enumerate(lines):
        text = DASH_FONT.render(line, True, (255, 255, 255))
        WIN.blit(text, (10, HEIGHT - 24 * (len(lines) - i)))

    pygame.display.update()



def run_duel():
    models = os.path.join(src, "models")
    with open(os.path.join(models, "waypoints.pkl"), "rb") as f:
        waypoints = pickle.load(f)

    radar_net    = load_net(os.path.join(models, "winner_radar.pkl"),    os.path.join(src, "neat_config.txt"))
    waypoint_net = load_net(os.path.join(models, "winner_waypoints.pkl"), os.path.join(src, "neat_waypoint_config.txt"))

    radar_car    = RadarCar(4, 4)
    waypoint_car = WaypointNeatCar(4, 5, waypoints)

    pygame.display.set_caption("Duel — Waypoints vs Radar")
    clock      = pygame.time.Clock()
    radar_dist = 0.0
    prev_pos   = (radar_car.x, radar_car.y)
    frame      = 0

    while True:
        frame += 1
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

        radar_car.step(radar_net)
        waypoint_car.step(waypoint_net)

        radar_dist += math.hypot(radar_car.x - prev_pos[0], radar_car.y - prev_pos[1])
        prev_pos = (radar_car.x, radar_car.y)

        radar_crashed    = radar_car.collide(TRACK_BORDER_MASK) is not None
        waypoint_crashed = waypoint_car.collide(TRACK_BORDER_MASK) is not None
        radar_finished   = radar_car.collide(FINISH_MASK, *FINISH_POSITION) is not None and radar_dist >= 2000
        waypoint_finished = waypoint_car.current_idx >= len(waypoints)

        winner = None
        if radar_finished and waypoint_finished:
            winner = "Tie — both finished!"
        elif radar_finished:
            winner = "Winner: Radar NN"
        elif waypoint_finished:
            winner = "Winner: Waypoint NN"
        elif radar_crashed and waypoint_crashed:
            winner = "Tie — both crashed"
        elif radar_crashed:
            winner = "Winner: Waypoint NN (radar crashed)"
        elif waypoint_crashed:
            winner = "Winner: Radar NN (waypoint crashed)"

        draw(radar_car, waypoint_car, waypoints, frame, winner)

        if winner:
            print(winner)
            draw(radar_car, waypoint_car, waypoints, frame, winner)
            pygame.time.wait(2000)
            radar_car.reset()
            radar_car.update_radars()
            waypoint_car.reset()
            radar_dist = 0.0
            prev_pos   = (radar_car.x, radar_car.y)


if __name__ == "__main__":
    run_duel()

import os
import json
import glob
import matplotlib.pyplot as plt


def load_all_stats(src):
    pattern = os.path.join(src, "*_stats.json")
    stats = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            stats.append(json.load(f))
    return stats


def plot_fitness(stats_list, src):
    fig, (ax_max, ax_mean) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    for s in stats_list:
        label = s["label"]
        gens  = list(range(1, len(s["fitness_max"]) + 1))
        ax_max.plot(gens,  s["fitness_max"],  label=label)
        ax_mean.plot(gens, s["fitness_mean"], label=label)

    ax_max.set_title("Best fitness per generation")
    ax_max.set_xlabel("Generation")
    ax_max.set_ylabel("Fitness")
    ax_max.legend(fontsize=7)
    ax_max.grid(True)

    ax_mean.set_title("Mean fitness per generation")
    ax_mean.set_xlabel("Generation")
    ax_mean.set_ylabel("Fitness")
    ax_mean.legend(fontsize=7)
    ax_mean.grid(True)

    plt.tight_layout()
    out = os.path.join(src, "fitness_curves.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


def print_summary(stats_list):
    print(f"\n{'Label':<22} {'Pop':>5} {'Gens':>6} {'Train(s)':>10} {'1st lap':>8} {'Best fit':>10} {'Replay laps':>12}")
    print("-" * 80)
    for s in stats_list:
        laps = s["replay_laps_ms"]
        lap_str = f"{min(laps)}ms" if laps else "—"
        print(
            f"{s['label']:<22} "
            f"{s['pop_size']:>5} "
            f"{s['n_gens']:>6} "
            f"{s['training_time_s']:>10} "
            f"{str(s['first_lap_gen']):>8} "
            f"{s['best_fitness']:>10} "
            f"{lap_str:>12}"
        )


if __name__ == "__main__":
    src        = os.path.dirname(__file__)
    stats_list = load_all_stats(src)

    if not stats_list:
        print("No stats files found. Run neat_radar.py first.")
    else:
        print_summary(stats_list)
        plot_fitness(stats_list, src)

#!/usr/bin/env python3

"""Train with episode-level weighted sampling without modifying ACT policy code."""

import argparse
import sys
from functools import partial

from weighted_episode_sampler import WeightedEpisodeSampler


def parse_weighted_sampler_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--weighted_sampler.first_n_episodes", dest="first_n_episodes", type=int, default=0)
    parser.add_argument("--weighted_sampler.first_n_weight", dest="first_n_weight", type=float, default=1.0)
    parser.add_argument("--weighted_sampler.later_weight", dest="later_weight", type=float, default=1.0)
    parser.add_argument("--weighted_sampler.length_weight_power", dest="length_weight_power", type=float, default=0.0)
    parser.add_argument("--weighted_sampler.min_length_multiplier", dest="min_length_multiplier", type=float, default=0.5)
    parser.add_argument("--weighted_sampler.max_length_multiplier", dest="max_length_multiplier", type=float, default=2.0)
    parser.add_argument("--weighted_sampler.num_samples_per_epoch", dest="num_samples_per_epoch", type=int, default=None)
    return parser.parse_known_args(argv)


def main() -> None:
    sampler_args, remaining_argv = parse_weighted_sampler_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining_argv]

    from lerobot.utils.import_utils import register_third_party_plugins
    import lerobot.scripts.lerobot_train as train_module

    weighted_sampler = partial(
        WeightedEpisodeSampler,
        first_n_episodes=sampler_args.first_n_episodes,
        first_n_weight=sampler_args.first_n_weight,
        later_weight=sampler_args.later_weight,
        length_weight_power=sampler_args.length_weight_power,
        min_length_multiplier=sampler_args.min_length_multiplier,
        max_length_multiplier=sampler_args.max_length_multiplier,
        num_samples_per_epoch=sampler_args.num_samples_per_epoch,
    )

    train_module.EpisodeAwareSampler = weighted_sampler
    register_third_party_plugins()
    train_module.train()


if __name__ == "__main__":
    main()

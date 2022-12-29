from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from .cupgame import CupGameBGT, Strategy
from .fillrates import gen_rates_beta, gen_rates_discrete_uniform, get_rates_partition
from .helper import export_data


def play_bgt(
    strategy: Strategy, rates: np.ndarray, rounds: int
) -> (Strategy, int | float, int, int, np.ndarray):
    """
    Play an instance of BGT and return the results

    :param strategy: strategy of the emptier
    :param rates: fill rates
    :param rounds: number of rounds
    :return: (strategy, backlog, cycle start, cycle length, fill rates)
    """
    cg = CupGameBGT(rates, rounds)
    cg.play(strategy)
    return strategy, cg.get_backlog(True), cg.cycle_start, cg.cycle_length, rates


def batch_cup_game(
    func, strategies: list[Strategy], fill_rates: npt.ArrayLike, rounds: int
):
    """
    Simulate the cup game for a given number of instances with the provided strategies.

    :param func: function to use for playing, specifies the variant of the game to play
    :param strategies: list of strategies to simulate the cup game for
    :param fill_rates: list of generated fill rates; shape (number of instances, cups)
    :param rounds: number of rounds to play initially
    :return:
    """
    # create args dict for starmap function, where each strategy acts as a key and the
    # associated value is a list of instances to play with that strategy
    args = dict()
    for strategy in strategies:
        args[strategy] = list()
    for rates in fill_rates:
        for strategy in strategies:
            args[strategy].append((strategy, rates, rounds))
    res = dict()
    # use multiprocessing to speed up computation, use every processor except for 1
    with Pool(cpu_count() - 1) as p:
        for strategy in strategies:
            res[strategy] = pd.DataFrame(
                p.starmap(func, args[strategy]),
                columns=["strategy", "backlog", "cycle-start", "cycle-length", "rates"],
            )
    return pd.concat(res)


def analyze_bgt_beta_distribution(
    alpha: float,
    beta: float,
    cups: int,
    size: int,
    max_rate: int,
    rounds: int,
    seed: Optional[int] = None,
):
    """
    Analyze BGT with fill rates drawn from a beta distribution.

    :param alpha: alpha parameter for beta distribution
    :param beta: beta parameter for beta distribution
    :param cups: number of cups
    :param size: number of instances
    :param max_rate: max fill rate
    :param rounds: number of rounds to play, before checking for a cycle
    :param seed: seed to use for the random number generator
    """
    ss = np.random.SeedSequence(seed)
    gen_rates = gen_rates_beta(alpha, beta, cups, size, max_rate, seed=ss)
    df = batch_cup_game(play_bgt, [Strategy.MAX, Strategy.DEADLINE], gen_rates, rounds)
    comment = f"alpha: {alpha}, beta: {beta}, seed: {ss.entropy}"
    export_data(df, f"bgt-beta-{cups}cups-{size}rows", comment)


def analyze_bgt_uniform_distribution(
    cups: int,
    size: int,
    max_rate: int,
    rounds: int,
    seed: Optional[int] = None,
):
    """
    Analyze BGT with fill rates drawn from a uniform distribution.

    :param cups: number of cups
    :param size: number of instances
    :param max_rate: max fill rate
    :param rounds: number of rounds to play, before checking for a cycle
    :param seed: seed to use for the random number generator
    """
    ss = np.random.SeedSequence(seed)
    gen_rates = gen_rates_discrete_uniform(cups, size, max_rate=max_rate, seed=ss)
    df = batch_cup_game(play_bgt, [Strategy.MAX, Strategy.DEADLINE], gen_rates, rounds)
    comment = f"seed: {ss.entropy}"
    export_data(df, f"uniform-{cups}cups-{size}rows", comment)


def analyze_bgt_partitions(h: int, rounds: int):
    """
    Analyze BGT with fill rates generated from all partitions of an integer `h`.

    :param h: integer to generate partitions for
    :param rounds: number of rounds to play, before checking for a cycle
    """
    gen_rates = get_rates_partition(h)
    df = batch_cup_game(
        play_bgt,
        [Strategy.MAX, Strategy.DEADLINE, Strategy.FASTEST_ONE],
        gen_rates,
        rounds,
    )
    export_data(df, f"partition-{h}", None)

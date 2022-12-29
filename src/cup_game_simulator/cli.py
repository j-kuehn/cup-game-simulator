#!/usr/bin/env python3

import argparse
import sys
from enum import Enum

import numpy as np

from cup_game_simulator.cupgame import CupGameBGT, CupGameFXD, CupGameSTD, Strategy
from cup_game_simulator.fillrates import (
    gen_rates_discrete_uniform as generate_fill_rates,
)


class Variant(Enum):
    STD = "std"
    FXD = "fxd"
    BGT = "bgt"

    # override for argparse compat
    def __str__(self):
        return self.value


def get_args():
    """
    Initialize an ArgumentParser and parse all arguments

    :return: populated namespace with argument objects
    """
    parser = argparse.ArgumentParser(
        description="Play the cup game. "
        "If no custom fill rates are provided, they will be randomly generated.",
        usage="%(prog)s {std,fxd,bgt} {RM,RF,RT,DD} [options]"
    )
    parser.add_argument(
        "variant",
        type=Variant,
        choices=Variant,
        help="variant of the game; "
        "standard (std), "
        "fixed-rate (fxd), or "
        "bamboo garden trimming (bgt)",
    )
    parser.add_argument(
        "strategy",
        type=Strategy,
        choices=Strategy,
        help="strategy of the emptier; "
        "Reduce-Max (RM), "
        "Reduce-Fastest(1) (RF),"
        "Reduce-Fastest(2) (RT), or "
        "Deadline-Driven Strategy (DD)",
    )
    parser.add_argument("-c", "--cups", type=int, help="number of cups")
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=100,
        help="number of rounds to play before checking for a cycle; "
        "only used for bgt and fxd",
    )
    parser.add_argument(
        "-f",
        "--fillrates",
        nargs="+",
        type=int,
        help="custom fill rates separated by spaces; "
        "for std, list of rates will be reshaped to array shape (rounds, cups)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.variant == Variant.STD and (not args.fillrates or not args.cups):
        print(
            "Can't play. "
            "For the standard cup game, generated fill rates are not allowed. "
            "Specify fill rates and number of cups."
        )
        sys.exit(1)
    if args.variant == Variant.STD and len(args.fillrates) % args.cups:
        print(
            "Can't play. " "Number of fill rates must be a multiple of number of cups."
        )
        sys.exit(1)
    if args.variant != Variant.STD and (
        (args.fillrates and args.cups) or (not args.fillrates and not args.cups)
    ):
        print(
            "Can't play. "
            "Either specify list of fill rates for custom rates, "
            "or number of cups for generated rates."
        )
        sys.exit(1)
    np.set_printoptions(suppress=True)

    # correctly set rates variable
    if args.variant == Variant.STD:
        rates = np.array(args.fillrates).reshape(
            (len(args.fillrates) // args.cups, args.cups)
        )
    else:
        if not args.fillrates:
            rates = np.squeeze(generate_fill_rates(args.cups, 1))
        else:
            rates = np.array(args.fillrates)

    # instantiate cup game instance
    if args.variant == Variant.STD:
        cg = CupGameSTD(rates)
    elif args.variant == Variant.FXD:
        cg = CupGameFXD(rates, args.rounds)
    else:  # Variant.BGT
        cg = CupGameBGT(rates, args.rounds)

    cg.play(args.strategy)

    print("Fill rates: ", end="")
    print(rates)

    print("Backlog (raw): ", end="")
    print(cg.get_backlog())

    print("Backlog (normalized): ", end="")
    print(cg.get_backlog(True))

    if args.variant in (Variant.FXD, Variant.BGT):
        print("Cycle Start: ", end="")
        print(cg.cycle_start)
        print("Cycle Length: ", end="")
        print(cg.cycle_length)

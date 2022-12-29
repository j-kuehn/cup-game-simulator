import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Final

import numpy as np
import pandas as pd

from .fillrates import DTYPE


class Strategy(Enum):
    """
    This enum models the available strategies for the cup game.
    """

    MAX = "RM"
    FASTEST_ONE = "RF"
    FASTEST_TWO = "RT"
    DEADLINE = "DD"

    # override for argparse compat
    def __str__(self):
        return self.value


class CupGame(ABC):
    """
    This abstract base class models an instance of a generic cup game.
    Inherit from this class to model a specific variant of the cup game.
    """

    @abstractmethod
    def __init__(self, rates: np.ndarray):
        """
        Initialize an instance of the cup game.

        :param rates: array of integer fill rates; shape (rounds, cups),
        """
        self._cups: Final[int] = rates.shape[1]
        # last axis of _result corresponds to: h, h', r
        self._result: np.ndarray = np.zeros(
            (rates.shape[0] + 1, self._cups, 3), dtype=DTYPE
        )
        self._result[:-1, :, 2] = rates
        self._total: Final[int] = sum(rates[0])  # corresponds to H
        self._played: int = 0  # counts the number of played rounds

    def _filler_step(self, t: int):
        """
        Fill cups for round t.

        :param t: round
        """
        self._result[t, :, 1] = self._result[t, :, 0] + self._result[t, :, 2]

    def _reduce_max(self, t: int) -> int:
        """
        Select cup for round t according to the Reduce-Max strategy.

        :param t: round
        :return: cup to remove water from
        """
        return np.argmax(self._result[t, :, 1])

    def _reduce_fastest(self, t: int, x: int) -> int:
        """
        Select cup for round t according to the Reduce-Fastest(x) strategy.

        :param t: round
        :param x: threshold above which cups are considered
        :return: cup to remove water from, -1 if no cup selected
        """
        i_fast = -1
        r_fast = 0
        for i in range(self._cups):
            if self._result[t, i, 1] >= x and self._result[t, i, 2] >= r_fast:
                i_fast = i
                r_fast = self._result[t, i, 2]
        return i_fast

    def _reduce_fastest_one(self, t: int) -> int:
        """
        Select cup for round t according to the Reduce-Fastest(1) strategy.

        :param t: round
        :return: cup to remove water from, -1 if no cup selected
        """
        return self._reduce_fastest(t, self._total)

    def _reduce_fastest_two(self, t: int) -> int:
        """
        Select cup for round t according to the Reduce-Fastest(2) strategy.

        :param t: round
        :return: cup to remove water from, -1 if no cup selected
        """
        return self._reduce_fastest(t, 2 * self._total)

    def _deadline_driven(self, t: int) -> int:
        """
        Select cup for round t according to the Deadline-Driven strategy.

        :param t: round
        :return: cup to remove water from, -1 if no cup selected
        """
        cur_index = -1
        min_rounds = math.inf
        for i in range(self._cups):
            if self._result[t, i, 1] >= self._total:  # fill height at least 1
                rem_rounds = (2 * self._total - self._result[t, i, 1]) // self._result[
                    t, i, 2
                ]
                if rem_rounds < min_rounds:
                    cur_index = i
                    min_rounds = rem_rounds
        return cur_index

    @abstractmethod
    def _emptier_step(self, t: int, cup):
        """
        Remove water from a cup in round t.

        :param t: round
        :param cup: cup to remove water from; if -1, don't remove water from any cup
        """

    def play(self, strategy: Strategy):
        """
        Play the cup game with the specified strategy.

        :param strategy: strategy of the emptier
        """
        for t in range(self._played, self._result.shape[0] - 1):
            self._filler_step(t)

            # select correct cup
            cup = -1
            if strategy == Strategy.MAX:
                cup = self._reduce_max(t)
            elif strategy == Strategy.FASTEST_ONE:
                cup = self._reduce_fastest_one(t)
            elif strategy == Strategy.FASTEST_TWO:
                cup = self._reduce_fastest_two(t)
            elif strategy == Strategy.DEADLINE:
                cup = self._deadline_driven(t)

            self._result[t + 1, :, 0] = self._result[t, :, 1]
            self._emptier_step(t, cup)
            self._played += 1

    def get_backlog(self, norm: bool = False) -> int | float:
        """
        Get the backlog after all played rounds.

        :return: backlog
        """
        if not self._played:
            raise UserWarning("The game needs to be played first.")
        if norm:
            return np.max(self._result[: self._played, :, 1]) / self._total
        return np.max(self._result[: self._played, :, 1])

    def get_int_heights(self, norm: bool = False) -> np.ndarray:
        """
        Get the intermediate fill heights of all cups for each round.

        :return: 2D array with intermediate fill heights; shape: (rounds,int_heights)
        """
        if not self._played:
            raise UserWarning("The game needs to be played first.")
        if norm:
            return self._result[: self._played, :, 1] / self._total
        return self._result[: self._played, :, 1]

    def get_max_int_heights(self, norm: bool = False) -> np.ndarray:
        """
        Get the intermediate fill heights of each round.

        :return: 1D array with backlogs
        """
        if not self._played:
            raise UserWarning("The game needs to be played first.")
        if norm:
            return np.max(self._result[: self._played, :, 1], axis=1) / self._total
        return np.max(self._result[: self._played, :, 1], axis=1)

    def get_heights(self, norm: bool = False) -> np.ndarray:
        """
        Get the fill heights of all cups before each round.

        :return: 2D array with fill heights; shape: (rounds,heights)
        """
        if not self._played:
            raise UserWarning("The game needs to be played first.")
        if norm:
            return self._result[: self._played, :, 0] / self._total
        return self._result[: self._played, :, 0]


class CupGameFixedRate(CupGame, ABC):
    """
    This abstract base class models variants of the cup game with fixed fill rates.
    It implements a cycle detection and plays for as many rounds as needed
    until reaching a cycle.
    """

    def __init__(self, rates: np.ndarray, rounds: int):
        """
        :param rates: array of fill rates, consisting of positive integers
        :param rounds: number of rounds to play before first checking for a cycle
        """
        super().__init__(
            np.broadcast_to(np.flip(np.sort(rates)), (rounds, rates.shape[0]))
        )
        self.cycle_start = -1
        self.cycle_length = -1

    def _reduce_fastest(self, t: int, x: int) -> int:
        """
        Select cup for round t according to the Reduce-Fastest(x) strategy.

        :param t: round
        :param x: threshold above which cups are considered
        :return: cup to remove water from, -1 if no cup selected
        """
        # fastest growth rate is first index with fill height >= x
        for i in range(self._cups):
            if self._result[t, i, 1] >= x:
                return i
        return -1  # no fill height >= x

    def _find_cycle(self):
        """
        Search for a cycle in the results and set the instance variables
        for cycle start and length if cycle is found.
        """
        res = pd.DataFrame(self._result[:, :, 0]).duplicated(keep="first")
        idx_first_dupe: int = res.idxmax()
        if not res[idx_first_dupe]:  # no cycle
            return
        idx_start_cycle = np.argmax(
            (self._result[:, :, 0] == self._result[idx_first_dupe, :, 0]).all(axis=1)
        )
        self.cycle_start = idx_start_cycle
        self.cycle_length = idx_first_dupe - idx_start_cycle

    def play(self, strategy: Strategy):
        """
        Play the cup game with the specified strategy.
        Search for a cycle periodically and, after one has been found, stop playing.

        :param strategy: strategy of the emptier
        """
        while self.cycle_start == -1:
            if self._played == self._result.shape[0] - 1:
                os = self._result.shape
                zs = (os[0] - 1, os[1], os[2])
                self._result = np.concatenate((self._result, np.zeros(zs, dtype=DTYPE)))
                self._result[os[0] - 1 : -1, :, 2] = self._result[0, :, 2]
            super().play(strategy)
            self._find_cycle()
        self._played = self.cycle_start + self.cycle_length + 1


class CupGameBGT(CupGameFixedRate):
    """
    This is a concrete class modeling the BGT variant.
    """

    def _emptier_step(self, t: int, cup):
        if cup == -1:
            return
        self._result[t + 1, cup, 0] = 0


class CupGameEBGT(CupGameFixedRate):
    """
    This is a concrete class modeling the epsilon-BGT variant.
    """

    def __init__(self, rates: np.ndarray, rounds: int, eps: int):
        """

        :param rates: array of fill rates; shape (rounds, cups),
        data type positive integers
        :param rounds: number of rounds to play, before first checking for a cycle
        :param eps: epsilon, selected cups are emptied up to this amount
        """
        super().__init__(rates, rounds)
        self.eps = eps

    def _emptier_step(self, t: int, cup):
        if cup == -1:
            return
        self._result[t + 1, cup, 0] = min(self.eps, self._result[t + 1, cup, 0])


class CupGameFXD(CupGameFixedRate, ABC):
    """
    This is a concrete class modeling the fixed-rate variant.
    """

    def _emptier_step(self, t: int, cup):
        if cup == -1:
            return
        self._result[t + 1, cup, 0] = max(0, self._result[t, cup, 1] - self._total)


class CupGameSTD(CupGame):
    """
    This is a concrete class modeling the standard cup game.
    """

    def __init__(self, rates: np.ndarray):
        if not np.all(rates.sum(axis=1) == sum(rates[0])):
            raise ValueError("sum of rates must be the same for every round")
        super().__init__(rates)

    def _emptier_step(self, t: int, cup):
        if cup == -1:
            return
        self._result[t + 1, cup, 0] = max(0, self._result[t, cup, 1] - self._total)

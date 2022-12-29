import numpy as np
import pytest

from cup_game_simulator.cupgame import DTYPE, CupGameBGT, Strategy


class TGame:
    def __init__(
        self,
        strategy,
        rates,
        heights,
        int_heights,
        backlogs,
        final_backlog,
        rounds_until_cycle,
        cycle_length,
    ):
        self.strategy = strategy
        self.rates = np.array(rates, dtype=DTYPE)
        self.heights = np.array(heights)
        self.int_heights = np.array(int_heights)
        self.rounds = len(int_heights)
        self.backlogs = np.array(backlogs)
        self.total_backlog = final_backlog
        self.rounds_until_cycle = rounds_until_cycle
        self.cycle_length = cycle_length


bgt_2cups_max = TGame(
    Strategy.MAX,
    [1, 1],
    [[0, 0], [0, 1], [1, 0], [0, 1]],
    [[1, 1], [1, 2], [2, 1]],
    [1, 2, 2],
    2,
    1,
    2,
)

bgt_3cups_max = TGame(
    Strategy.MAX,
    [5, 3, 2],
    [[0, 0, 0], [0, 3, 2], [5, 0, 4], [0, 3, 6], [5, 6, 0], [0, 9, 2], [5, 0, 4]],
    [[5, 3, 2], [5, 6, 4], [10, 3, 6], [5, 6, 8], [10, 9, 2], [5, 12, 4]],
    [5, 6, 10, 8, 10, 12],
    12,
    2,
    4,
)

bgt_2cups_fastest_one = TGame(
    Strategy.FASTEST_ONE,
    [1, 1],
    [[0, 0], [1, 1], [0, 2], [1, 0], [0, 1], [1, 0]],
    [[1, 1], [2, 2], [1, 3], [2, 1], [1, 2]],
    [1, 2, 3, 2, 2],
    3,
    3,
    2,
)

bgt_3cups_fastest_one = TGame(
    Strategy.FASTEST_ONE,
    [5, 3, 2],
    [
        [0, 0, 0],
        [5, 3, 2],
        [0, 6, 4],
        [5, 9, 6],
        [0, 12, 8],
        [5, 0, 10],
        [0, 3, 12],
        [5, 6, 0],
        [0, 9, 2],
        [5, 0, 4],
        [0, 3, 6],
        [5, 6, 8],
        [0, 9, 10],
        [5, 0, 12],
        [0, 3, 14],
        [5, 6, 0],
    ],
    [
        [5, 3, 2],
        [10, 6, 4],
        [5, 9, 6],
        [10, 12, 8],
        [5, 15, 10],
        [10, 3, 12],
        [5, 6, 14],
        [10, 9, 2],
        [5, 12, 4],
        [10, 3, 6],
        [5, 6, 8],
        [10, 9, 10],
        [5, 12, 12],
        [10, 3, 14],
        [5, 6, 16],
    ],
    [5, 10, 9, 12, 15, 12, 14, 10, 12, 10, 8, 10, 12, 14, 16],
    16,
    7,
    8,
)

bgt_2cups_fastest_two = TGame(
    Strategy.FASTEST_TWO,
    [1, 1],
    [[0, 0], [1, 1], [2, 2], [3, 3], [0, 4], [1, 0], [2, 1], [3, 2], [0, 3], [1, 0]],
    [[1, 1], [2, 2], [3, 3], [4, 4], [1, 5], [2, 1], [3, 2], [4, 3], [1, 4]],
    [1, 2, 3, 4, 5, 2, 3, 4, 4],
    5,
    5,
    4,
)

bgt_3cups_fastest_two = TGame(
    Strategy.FASTEST_TWO,
    [5, 3, 2],
    [
        [0, 0, 0],
        [5, 3, 2],
        [10, 6, 4],
        [15, 9, 6],
        [0, 12, 8],
        [5, 15, 10],
        [10, 18, 12],
        [15, 0, 14],
    ],
    [
        [5, 3, 2],
        [10, 6, 4],
        [15, 9, 6],
        [20, 12, 8],
        [5, 15, 10],
        [10, 18, 12],
        [15, 21, 14],
    ],
    [5, 10, 15, 20, 15, 18, 21],
    21,
    -1,
    -1,
)

bgt_2cups_deadline = TGame(
    Strategy.DEADLINE,
    [1, 1],
    [[0, 0], [1, 1], [0, 2], [1, 0], [0, 1], [1, 0]],
    [[1, 1], [2, 2], [1, 3], [2, 1], [1, 2]],
    [1, 2, 3, 2, 2],
    3,
    3,
    2,
)

bgt_3cups_deadline = TGame(
    Strategy.DEADLINE,
    [5, 4, 1],
    [
        [0, 0, 0],
        [5, 4, 1],
        [0, 8, 2],
        [5, 0, 3],
        [0, 4, 4],
        [5, 8, 5],
        [0, 12, 6],
        [5, 0, 7],
        [0, 4, 8],
        [5, 8, 9],
        [0, 12, 10],
        [5, 0, 11],
        [0, 4, 12],
        [5, 8, 0],
        [0, 12, 1],
        [5, 0, 2],
        [0, 4, 3],
        [5, 8, 4],
        [0, 12, 5],
        [5, 0, 6],
        [0, 4, 7],
        [5, 8, 8],
        [0, 12, 9],
        [5, 0, 10],
        [0, 4, 11],
        [5, 8, 0],
    ],
    [
        [5, 4, 1],
        [10, 8, 2],
        [5, 12, 3],
        [10, 4, 4],
        [5, 8, 5],
        [10, 12, 6],
        [5, 16, 7],
        [10, 4, 8],
        [5, 8, 9],
        [10, 12, 10],
        [5, 16, 11],
        [10, 4, 12],
        [5, 8, 13],
        [10, 12, 1],
        [5, 16, 2],
        [10, 4, 3],
        [5, 8, 4],
        [10, 12, 5],
        [5, 16, 6],
        [10, 4, 7],
        [5, 8, 8],
        [10, 12, 9],
        [5, 16, 10],
        [10, 4, 11],
        [5, 8, 12],
    ],
    [
        5,
        10,
        12,
        10,
        8,
        12,
        16,
        10,
        9,
        12,
        16,
        12,
        13,
        12,
        16,
        10,
        8,
        12,
        16,
        10,
        8,
        12,
        16,
        11,
        12,
    ],
    16,
    13,
    12,
)


@pytest.mark.parametrize(
    "tg",
    [
        bgt_2cups_max,
        bgt_3cups_max,
        bgt_2cups_fastest_one,
        bgt_3cups_fastest_one,
        bgt_2cups_fastest_two,
        # bgt_3cups_fastest_two,
        bgt_2cups_deadline,
        bgt_3cups_deadline,
    ],
)
def test_cupgame_bgt_max(tg):
    cg = CupGameBGT(tg.rates, tg.rounds)
    cg.play(tg.strategy)

    assert np.array_equal(cg.get_heights(), tg.heights)
    assert np.array_equal(cg.get_int_heights()[:-1], tg.int_heights)
    assert np.array_equal(cg.get_max_int_heights()[:-1], tg.backlogs)
    assert cg.get_backlog() == tg.total_backlog

    divisor = sum(tg.rates)
    assert np.array_equal(cg.get_heights(True), tg.heights / divisor)
    assert np.array_equal(cg.get_int_heights(True)[:-1], tg.int_heights / divisor)
    assert np.array_equal(cg.get_max_int_heights(True)[:-1], tg.backlogs / divisor)
    assert cg.get_backlog(True) == tg.total_backlog / divisor

    assert cg.cycle_start == tg.rounds_until_cycle
    assert cg.cycle_length == tg.cycle_length

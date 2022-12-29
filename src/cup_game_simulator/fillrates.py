import numpy as np

DTYPE = np.uint
MAX_INT = np.iinfo(DTYPE).max


def nth_harmonic(n: int) -> int:
    """
    Calculate the nth harmonic number.

    :param n: n
    :return: nth harmonic number
    """
    h = 0
    for i in range(n):
        h += 1 / (i + 1)
    return h


def gen_rates_discrete_uniform(
    cups: int, rows: int = 1, max_rate: int = 1000, seed=None
) -> np.ndarray:
    """
    Generate random integer fill rates within the interval [minimum,maximum].

    :param cups: number of cups
    :param rows: number of rounds to generate fill rates for; no effect for FXD and BGT
    :param max_rate: upper end of the range of generated numbers
    :param seed: seed for the random number generator;
        if 'None', fresh, unpredictable entropy is used
    :return: generated fill rates, shape (rows, cups)
    """
    if max_rate * cups * nth_harmonic(cups) > MAX_INT:
        raise OverflowError
    rng = np.random.default_rng(seed)
    return rng.integers(1, max_rate, size=(rows, cups), dtype=DTYPE, endpoint=True)


def gen_rates_beta(
    alpha: float, beta: float, cups: int, rows: int, max_rate: int, seed=None
) -> np.ndarray:
    """
    Generate fill rates based on the beta distribution.

    Since the beta distribution is continuous and the fill rates are discrete,
    the generated value from the beta distribution is multiplied by
    the maximum fill rate and that rate is then rounded up to the next integer.

    :param alpha: alpha parameter for beta distribution
    :param beta: beta parameter for beta distribution
    :param cups: number of cups
    :param rows: number of rows to generate
    :param max_rate: indicates the maximum fill rate that can be generated
    :param seed: seed for the random number generator;
        if 'None', fresh, unpredictable entropy is used
    :return: generated fill rates, shape (rows, cups)
    """
    if max_rate * cups * nth_harmonic(cups) > MAX_INT:
        raise OverflowError
    rng = np.random.default_rng(seed)
    return np.array(
        np.ceil(rng.beta(alpha, beta, size=(rows, cups)) * max_rate), dtype=np.uint
    )


def get_rates_partition(n: int) -> list:
    """
    Generate a list of all integer partitions for a specific integer n.

    source: https://jeromekelleher.net/generating-integer-partitions.html

    :param n: integer to generate partitions for
    :return: list
    """
    partitions = list()
    a = [0 for _ in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1  # noqa: E741
        while x <= y:
            a[k] = x
            a[l] = y
            partitions.append(np.flip(np.array(a[: k + 2])))
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        partitions.append(np.flip(np.array(a[: k + 1])))
    return partitions

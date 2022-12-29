# Cup Game Simulator

The cup game simulator was developed for the experimental analysis performed as part of my bachelor's thesis.
It requires Python 3.10 or greater.
The recommended tool for installing dependencies is Poetry [[1](https://python-poetry.org/docs/)].
Alternatively, a `requirements.txt` file is included for dependency management.

The simulator provides a command line interface for simulating a single instance of the game.
For help, run `poetry run cli.py -h`.

Generating instances and batch simulating is handled by `analysis.py`, which was used for producing the input of my BGT analysis [[2](https://github.com/j-kuehn/bgt-analysis)].

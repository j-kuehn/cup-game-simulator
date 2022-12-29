import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

DATA = Path(os.getenv("BASE")) / "data"


def export_data(df: pd.DataFrame, fn_hint: str, comment: Optional[str] = None):
    """
    Export a 'DataFrame' to a csv file.

    :param df: pandas 'DataFrame' to write to the file
    :param fn_hint: will be attached to the filename
    :param comment: will be added to the file as comment in the first line
    :return:
    """
    fn = f"{get_current_timestamp()}-{fn_hint}.csv"
    with open(DATA / fn, mode="w", encoding="utf-8") as f:
        if comment:
            f.write(f"# {comment}\n")
        df.to_csv(f, index=False)


def get_current_timestamp() -> str:
    """
    Get the current local timestamp.

    :return: timestamp in the format yymmdd-HHMMSS
    """
    return datetime.now().strftime("%y%m%d-%H%M%S")

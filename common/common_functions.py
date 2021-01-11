import numpy as np


def log_return(df, col_name):
    """
    Compute the log returns

    Args:
        df: <pd.DataFrame> containing data
        col_name: <str> name of column to compute returns on

    Returns:
        df: <pd.DataFrame> with return column appended
        ret_col_name: <str> resulting column name containing the returns.  Format is {col_name}_log_ret
    """
    ret_col_name = col_name + '_log_ret'

    df[ret_col_name] = np.log(df[col_name] / df[col_name].shift(1))

    return df, ret_col_name

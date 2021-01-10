import numpy as np


def log_return(df, col_name):
    ret_col_name = col_name + '_log_ret'

    df[ret_col_name] = np.log(df[col_name] / df[col_name].shift(1))

    return df, ret_col_name

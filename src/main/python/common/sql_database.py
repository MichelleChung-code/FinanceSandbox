import sqlite3 as sq3
from pathlib import Path
import os
import numpy as np
import pandas.io.sql as pds


def create_table(conn, table_name, cols_ls):
    """
    Create SQL lite DB table

    Args:
        conn: <sqlite3.Connection> connection object
        table_name: <str> name of table to create
        cols_ls: <str> list of columns and designated type in format:
        'col1name col1type, col2name col2type, col3name col3type'

    """
    query = 'CREATE TABLE {table_name} ({cols_ls})'.format(table_name=table_name, cols_ls=cols_ls)
    conn.execute(query)
    conn.commit()


def write_to_table(conn, data, table_name):
    """
    Writes data to table

    Args:
        conn: <sqlite3.Connection> connection object
        data: <ndarray> data to write to table
        table_name: <str> name of table to write to

    """
    for row in data:
        conn.execute(
            'INSERT INTO {table_name} VALUES({num_question_marks})'.format(table_name=table_name,
                                                                           num_question_marks=', '.join(
                                                                               ['?' for i, _ in enumerate(row)])),
            tuple(i for i in row))

    conn.commit()


def get_connection(db_path):
    """
    Creates the sqlite3 connection object

    Args:
        db_path: <str> path to store table

    Returns:
        <sqlite3.Connection> connection object
    """
    return sq3.connect(db_path)


if __name__ == '__main__':
    cols_ls = "Date date, Num1 real, Num2 real"
    table_name = 'test'

    db_path = os.path.join(str(Path(__file__).parents[1]), 'data', 'test.db')

    try:
        os.remove(db_path)
        print('DB Table already exists, existing table deleted.')
    except OSError:
        pass

    conn = get_connection(db_path)
    create_table(conn, table_name, cols_ls)

    data_to_insert = np.random.standard_normal((10000, 3))
    write_to_table(conn, data_to_insert, table_name)

    # read into numpy array
    res_np = np.array(conn.execute('SELECT * FROM {table_name}'.format(table_name=table_name)).fetchmany(5)).round(3)
    print(res_np)

    # read into pandas
    res_pd = pds.read_sql('SELECT * FROM {table_name}'.format(table_name=table_name), conn)
    print(res_pd.head())

    conn.close()

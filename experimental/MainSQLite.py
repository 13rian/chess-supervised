import logging
import time


from utils import utils

import sqlite3
from sqlite3 import Error




# The logger
utils.init_logger(logging.DEBUG, file_name="../log/chess_sl.log")
logger = logging.getLogger('Chess_SL')


db_file = "test.db"



def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn




def insert_position(conn, position):
    """
    Create a new task
    :param conn:
    :param position:
    :return:
    """

    sql = ''' INSERT INTO position(fen, moves, results)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, position)
    connection.commit()                 # commit takes a lot of time


def insert_or_update(conn, position, pos_update):
    sql_insert = """INSERT OR IGNORE INTO position(fen, moves, results)
            VALUES(?,?,?);"""

    sql_update = """UPDATE position SET 
            moves = moves || ?,
            results = results || ?
            WHERE fen = ?;"""

    cur = conn.cursor()
    cur.execute(sql_insert, position)
    cur.execute(sql_update, pos_update)
    connection.commit()                 # commit takes a lot of time



if __name__ == '__main__':
    # select the moves of the first position
    connection = create_connection("fen_positions-kb-light.db")
    c = connection.cursor()
    res = c.execute("SELECT * FROM position WHERE fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';")
    row = c.fetchone()
    row2 = c.fetchone()

    logger.debug(len(row[2].split("_")))


    # create a new db
    connection = create_connection(db_file)

    # create a table
    sql_create_table = """CREATE TABLE IF NOT EXISTS position (
                            id integer PRIMARY KEY,
                            fen text,
                            moves text,
                            results text
                        );"""

    # create the index of the fen
    sql_index = """CREATE UNIQUE INDEX IF NOT EXISTS idx_position_fen ON
                    position(fen);"""

    try:
        c = connection.cursor()
        c.execute(sql_create_table)
        c.execute(sql_index)
        connection.commit()

    except Error as e:
        print(e)


    # start = time.time()
    # for i in range(100):
    #     position = ('kknn/nnkkk {}'.format(i), '67', '1')
    #     ret = insert_position(connection, position)
    # connection.commit()
    # print(time.time() - start)


    position = ('kknn/nnkkk', '67', '1')
    pos_update = ('_68', '_-1', 'kknn/nnkkk')
    insert_or_update(connection, position, pos_update)


    connection.close()

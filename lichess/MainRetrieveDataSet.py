from pathlib import Path
import bz2
import os

import urllib.request
import chess.pgn

from lichess import lichess_data


# download the url files from lichess
# lichess_data.download_url_files()


# download all pgn files
# lichess_data.download_pgn_files()


# parse the pgn files
pgn_dir = lichess_data.pgn_folder + "/threeCheck"
path_list = os.listdir(pgn_dir)

elo_threshold = 2000
game_count = 0
for pgn_file_name in path_list:
    if pgn_file_name.endswith("bz2"):
        continue

    pgn_file_path = pgn_dir + "/" + pgn_file_name
    pgn_file = open(pgn_file_path)
    print("start to process file {}".format(pgn_file_name))

    # read out all games in the pgn file
    game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn
    while game is not None:
        board = chess.Board()               # create a new board

        min_elo = min(game.headers["WhiteElo"], game.headers["BlackElo"])
        if int(min_elo) >= elo_threshold:
            game_count += 1

        game = chess.pgn.read_game(pgn_file)  # read out the next game from the pgn


    pgn_file.close()

print("for elo threshold " + str(elo_threshold) + str(game_count) + " were found")



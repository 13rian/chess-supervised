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
elo_threshold = 2000
pgn_dir = lichess_data.pgn_folder + "/threeCheck"   # kingOfTheHill     threeCheck
game_count = lichess_data.count_games(elo_threshold, pgn_dir)

print("for elo threshold " + str(elo_threshold) + " " + str(game_count) + " were found")



from pathlib import Path
import bz2
import os

import urllib.request
import chess.pgn

# variants = [
#     "antichess",
#     "atomic",
#     "chess960",
#     "crazyhouse",
#     "horde",
#     "kingOfTheHill",
#     "racingKings",
#     "threeCheck",
# ]

variants = [
    "threeCheck",
    "kingOfTheHill",
    "crazyhouse"
]

url_files_folder = "url_files"
pgn_folder = "pgn"


def download_url_files():
    """
    download the url files from lichess
    :return:
    """

    Path(url_files_folder).mkdir(parents=True, exist_ok=True)

    for variant in variants:
        url = "https://database.lichess.org/" + variant + "/list.txt"
        path = url_files_folder + "/" + variant + ".txt"
        urllib.request.urlretrieve(url, path)


def download_pgn_files():
    """
    download all pgn files
    :return:
    """

    for variant in variants:
        url_file = url_files_folder + "/" + variant + ".txt"
        pgn_dir = pgn_folder + "/" + variant

        Path(pgn_dir).mkdir(parents=True, exist_ok=True)

        with open(url_file, "r") as fp:
            line = fp.readline()
            while line:
                url = line.strip()
                path = pgn_dir + "/" + url.split("/")[-1]
                print("start to download " + path)
                urllib.request.urlretrieve(url, path)

                # extract the file
                extracted_file = path[0:-4]
                with open(extracted_file, 'wb') as new_file, bz2.BZ2File(path, 'rb') as file:
                    for data in iter(lambda: file.read(100 * 1024), b''):
                        new_file.write(data)

                line = fp.readline()


def count_games(elo_threshold, pgn_dir):
    """
    counts the number of games where the weaker player has at least the minimal passed elo
    :param elo_threshold:   the minimal elo of the weaker player in a game
    :param pgn_dir:         the directory containing all the pgn files
    :return:                the total games played with the defined minimal elo_threshold
    """

    game_count = 0
    for pgn_file_name in os.listdir(pgn_dir):
        if not pgn_file_name.endswith(".pgn"):
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

    return game_count







import copy

import chess

from globals import CONST, Config
import board_representation


class GameBoard:
    """
    class that contains the complete games logic, this is the only place AlphaZero requires human knowledge
    (basic rules of the games to learn)
    """

    def clone(self):
        """
        returns a new board with the same state
        :return:
        """
        board = copy.deepcopy(self)
        return board


    def is_terminal(self):
        """
        returns true if the position is terminal and false if the games is still running
        :return:
        """
        pass


    def current_player(self):
        """
        returns the current player, needs to be CONST.WHITE or CONST.BLACK
        :return:
        """
        pass


    def symmetries(self, policy):
        """
        :param policy:      the policy of the original board
        returns a list of symmetric boards and policies or None if there are no symmetries or if symmetric positions
        should not be used for the training. the list should not include the original board. the boards and the
        policies need to have the same order in order to be able to map them to each other
        :return:    list of boards
                    list of policy
        """
        return None, None


    @staticmethod
    def symmetry_count():
        """
        returns the number of symmetries that are created by symmetric_boards(). the symmetry count is always 1 larger
        than the size of the list returned by symmetric_boards(). if there is no symmetry this count should be 1
        :return:
        """
        return 1

    
    def white_perspective(self):
        """
        returns the board from the white perspective. If it is white's move the normal board representation is returned.
        if it is black's move the white and the black pieces are swapped.
        :return:    the matrix representation of the board
                    the current player (CONST.WHITE or CONST.BLACK)
        """
        pass
    
    
    def state_id(self):
        """
        returns a unique state id of the current board
        :return:
        """
        pass


    def execute_action(self, action):
        """
        executes the passed action in the current state of the game
        :param action:    action to execute, this is an integer between 0 and the total number of all actions
        :return:
        """
        pass


    def legal_actions(self):
        """
        returns a list of all legal actions of the current game state
        :return:
        """
        pass


    def illegal_actions(self):
        """
        returns a list of all illegal actions, this list could be calculated from the legal actions but
        sometimes there are faster implementations available to find all illegal actions
        :return:
        """
        pass


    def reward(self):
        """
        returns the reward of the games
        :return:    -1 if black has won
                    0 if the games is drawn or the games is still running
                    1 if white has won
        """
        pass


    def training_reward(self):
        """
        returns the reward for training, this is normally the same method as self.reward()
        :return:
        """
        pass



class ChessBoard(GameBoard):
    def __init__(self):
        self.chess_board = chess.Board()


    def is_terminal(self):
        return self.chess_board.is_game_over()


    def current_player(self):
        if self.chess_board.turn == chess.WHITE:
            return CONST.WHITE
        else:
            return CONST.BLACK


    def white_perspective(self):
        player = CONST.WHITE if self.chess_board.turn == chess.WHITE else CONST.BLACK
        return board_representation.board_to_matrix(self.chess_board), player


    def state_id(self):
        return self.chess_board.fen()


    def execute_action(self, action):
        move = board_representation.index_to_move(action, self.chess_board.turn)
        self.chess_board.push(move)


    def legal_actions(self):
        legal_actions = []
        for move in self.chess_board.legal_moves:
            action = board_representation.move_to_index(move, self.chess_board.turn)
            legal_actions.append(action)

        return legal_actions


    def illegal_actions(self):
        legal_actions = self.legal_actions()
        illegal_actions = [a for a in range(CONST.POLICY_SIZE)]
        for legal_action in legal_actions:
            illegal_actions.remove(legal_action)

        return illegal_actions


    def reward(self):
        if not self.chess_board.is_game_over():
            return 0

        if self.chess_board.turn == chess.WHITE:
            return -1
        else:
            return 1


    def training_reward(self):
        return self.reward()

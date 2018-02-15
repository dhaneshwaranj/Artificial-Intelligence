#!/usr/bin/env python
from isolation import Board, game_as_text
from random import randint

# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.

# Submission Class 1
class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Evaluation function that outputs a score equal to how many 
        moves are open for AI player on the board.

        Args
            param1 (Board): The board and game state.
            param2 (bool): True if maximizing player is active.

        Returns:
            float: The current state's score. Number of your agent's moves.

        """

        # TODO: finish this function!
        #raise NotImplementedError
        return len(game.get_legal_moves())
        # I am just returning the number of my open moves. The logic for giving the final score based on how good the
        # board is for us is done in CustomEvalFn which doesn't just score the board based on open moves

# Submission Class 2
class CustomEvalFn:

    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state
        
        Custom evaluation function that acts however you think it should. This 
        is not required but highly encouraged if you want to build the best 
        AI possible.
        
        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.
            
        """

        best_val = len(game.get_legal_moves())
        if best_val == 0 and maximizing_player_turn is True:        # If we are playing after the last move, we lose.
            best_val = float("-inf")
            return best_val
        elif best_val == 0 and maximizing_player_turn is False:     # If opponent plays after the last move, we win.
            best_val = float("inf")
            return best_val
        # elif maximizing_player_turn is True:                        # If we cut-off the search.
        #     return -best_val
        else:
            return -best_val

            # TODO: finish this function!
        raise NotImplementedError


class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=4, eval_fn=CustomEvalFn()):
        """Initializes your player.
        
        if you find yourself with a superior eval function, update the default 
        value of `eval_fn` to `CustomEvalFn()`
        
        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.turn = 0
        self.player1 = 0

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent
        
        Args:
            game (Board): The board and game state.
            legal_moves (dict): Dictionary of legal moves and their outcomes
            time_left (function): Used to determine time left before timeout
            
        Returns:
            (tuple): best_move
        """

        if game.move_count == 0:
            self.player1 = 1
        self.turn += 1
        rand = 4

        if self.turn % rand == 0 and self.player1 == 0:
            best_move = legal_moves[randint(0, len(legal_moves) - 1)]
            new_game = game.copy()                                              # Copy board.
            new_game.__apply_move__(best_move)                                  # Apply move in board copy.
            util = self.eval_fn.score(new_game)
            if util == 0 or util == float("-inf"):
                best_move, util = self.idminimax(game, time_left, depth=self.search_depth+self.turn)

        elif self.turn % 3 == 0:
            best_move, util = self.idalphabeta(game, time_left, depth=self.search_depth+self.turn)

        else:
            best_move, util = self.idminimax(game, time_left, depth=self.search_depth+self.turn)
        # change minimax to alphabeta after completing alphabeta part of assignment
        return best_move

    def utility(self, game):
        """Can be updated if desired"""
        return self.eval_fn.score(game)

    def minimax(self, game, time_left, depth=4, maximizing_player=True):
        """Implementation of the minimax algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """

        children = game.get_legal_moves()
        if depth == 0 or len(children) == 0 or time_left() < 80:         # Check if we are at the end of our search.
            best_move = game.__last_queen_move__                         # Assign the best move.
            best_val = self.eval_fn.score(game, maximizing_player)       # Calculate the Utility.
            return best_move, best_val                                   # If we are cutting-off, send the node's score.

        elif maximizing_player is True:                                  # Max's turn.
            best_val = float("-inf")                                     # Init best_val to -inf.
            best_move = ()                                               # Init best_move to None.
            for child in children:                                       # Iterate for all Child Nodes.
                new_game = game.copy()                                   # Copy board.
                new_game.__apply_move__(child)                           # Apply move.
                temp_move, val = self.minimax(new_game, time_left,
                                              depth-1, False)            # Run minimax recursively.
                if val > best_val:                                       # Check if value is better than previous best.
                    best_val = val                                       # Update best_value.
                    best_move = child                                    # Update best_move.
            return best_move, best_val                                   # Return best_move and associated Utility.

        else:                                                            # Min's turn.
            best_val = float("inf")                                      # Init best_val to inf.
            best_move = ()                                               # Init best_move to None.
            for child in children:                                       # Iterate for all Child Nodes.
                new_game = game.copy()                                   # Copy board.
                new_game.__apply_move__(child)                           # Apply move.
                temp_move, val = self.minimax(new_game, time_left,
                                              depth-1, True)             # Run minimax recursively.
                if val < best_val:                                       # Check if value is worse than previous worst.
                    best_val = val                                       # Get best_value.
                    best_move = child                                    # Update child.
            return best_move, best_val                                   # Return best_move and associated Utility.

        # TODO: finish this function!
        raise NotImplementedError

    def alphabeta(self, game, time_left, depth=4, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implementation of the alphabeta algorithm
        
        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """

        children = game.get_legal_moves()
        if depth == 0 or len(children) == 0 or time_left() < 80:         # Check if we are at the end of our search.
            best_move = game.__last_queen_move__                         # Assign the best move.
            best_val = self.eval_fn.score(game, maximizing_player)       # Calculate the Utility.
            return best_move, best_val                                   # If we are cutting-off, send the node & score.

        elif maximizing_player is True:                                  # Max's turn.
            best_val = float("-inf")                                     # Init best_val to -inf.
            best_move = ()                                               # Init best_move to None.
            for child in children:                                       # Iterate for all Child Nodes.
                new_game = game.copy()                                   # Copy board.
                new_game.__apply_move__(child)                           # Apply move in board copy.
                temp_move, val = self.alphabeta(new_game, time_left,
                                                depth-1, alpha, beta,
                                                False)                   # Run alphabeta recursively.
                if val > best_val:                                       # Check if value is better than previous best
                    best_val = val                                       # Get best_value.
                    best_move = child                                    # Assign best move.
                    alpha = max(alpha, best_val)                         # Update alpha.
                    if beta <= alpha:                                    # Beta cut-off.
                        break
            return best_move, best_val                                   # Return best_move and associated Utility.

        else:                                                            # Min's turn.
            best_val = float("inf")                                      # Init best_val to inf.
            best_move = ()                                               # Init best_move to None.
            for child in children:                                       # Iterate for all Child Nodes.
                new_game = game.copy()                                   # Copy board.
                new_game.__apply_move__(child)                           # Apply move in board copy.
                temp_move, val = self.alphabeta(new_game, time_left,
                                                depth-1, alpha, beta,
                                                True)                    # Run alphabeta recursively.
                if val < best_val:                                       # Check if value is worse than previous worst.
                    best_val = val                                       # Get best_value.
                    best_move = child                                    # Assign best move.
                    beta = min(beta, best_val)                           # Update beta.
                    if beta <= alpha:                                    # Alpha cut-off.
                        break
            return best_move, best_val                                   # Return best_move and associated Utility.

        # TODO: finish this function!
        raise NotImplementedError
        return best_move, val

    def iterminimax(self, game, time_left, depth=4, maximizing_player=True):
        """Implementation of the iterative minimax algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int, bool): best_move, best_val, timeout
        """
        timeout = 0
        time = time_left()
        children = game.get_legal_moves()
        if depth == 0 or len(children) == 0 or time < 30:                # Check if we are at the end of our search.
            best_move = game.__last_queen_move__                         # Assign the best move.
            best_val = self.eval_fn.score(game, maximizing_player)       # Calculate the Utility.
            if time < 30:                                                # Check to see if we timing out.
                timeout = 1
            return best_move, best_val, timeout                          # If we are cutting-off, send the node's score.

        elif maximizing_player is True:                                  # Max's turn.
            best_val = float("-inf")                                     # Init best_val to -inf.
            best_move = ()                                               # Init best_move to None.
            for child in children:                                       # Iterate for all Child Nodes.
                new_game = game.copy()                                   # Copy board.
                new_game.__apply_move__(child)                           # Apply move.
                temp_move, val, timeout = \
                    self.iterminimax(new_game, time_left,
                                     depth-1, False)                     # Run iterative minimax recursively.
                if timeout == 1:
                    break                                                # Break if we timeout.
                if val > best_val:                                       # Check if value is better than previous best.
                    best_val = val                                       # Update best_value.
                    best_move = child                                    # Update best_move.
            return best_move, best_val, timeout                          # Return best_move and associated Utility.

        else:                                                            # Min's turn.
            best_val = float("inf")                                      # Init best_val to inf.
            best_move = ()                                               # Init best_move to None.
            for child in children:                                       # Iterate for all Child Nodes.
                new_game = game.copy()                                   # Copy board.
                new_game.__apply_move__(child)                           # Apply move.
                temp_move, val, timeout = \
                    self.iterminimax(new_game, time_left,
                                     depth-1, True)                      # Run iterative minimax recursively.
                if timeout == 1:
                    break                                                # Break if we timeout.
                if val < best_val:                                       # Check if value is worse than previous worst.
                    best_val = val                                       # Get best_value.
                    best_move = child                                    # Update child.
            return best_move, best_val, timeout                          # Return best_move and associated Utility.

        # TODO: finish this function!
        raise NotImplementedError

    def idminimax(self, game, time_left, depth=4, maximizing_player=True):
        """Function that calls the iterative minimax algorithm
        after incrementing the depth at each pass"""

        best_move = ()
        best_val = float("-inf")
        for i in range(0, depth+1, 1):                                   # Iterate from depth 0 to self.depth.
            temp_move, temp_val, timeout = \
                self.iterminimax(game, time_left, i, True)               # Call Iterative Minimax.
            if timeout == 0:                                             # If we don't timeout,
                best_move = temp_move                                    # return the best_move, and
                best_val = temp_val                                      # best_val
            else:                                                        # If we timeout, then return the previous
                break                                                    # best_move and best_val
        return best_move, best_val

    def iteralphabeta(self, game, time_left, depth=4, alpha=float("-inf"), beta=float("inf"),
                      maximizing_player=True):
        """Implementation of the iterative alphabeta algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        timeout = 0
        time = time_left()
        children = game.get_legal_moves()
        if depth == 0 or len(children) == 0 or time < 20:                # Check if we are at the end of our search.
            best_move = game.__last_queen_move__                         # Assign the best move.
            best_val = self.eval_fn.score(game, maximizing_player)       # Calculate the Utility.
            if time < 20:                                                # Check to see if we timing out.
                timeout = 1
            return best_move, best_val, timeout                          # If we are cutting-off, send the node's score.

        elif maximizing_player is True:                                  # Max's turn.
            best_val = float("-inf")                                     # Init best_val to -inf.
            best_move = ()                                               # Init best_move to None.
            for child in children:                                       # Iterate for all Child Nodes.
                new_game = game.copy()                                   # Copy board.
                new_game.__apply_move__(child)                           # Apply move in board copy.
                temp_move, val, timeout = \
                    self.iteralphabeta(new_game, time_left, depth-1,
                                       alpha, beta, False)               # Run Iterative alphabeta recursively.
                if timeout == 1:
                    break                                                # Break if we timeout.
                if val > best_val:                                       # Check if value is better than previous best
                    best_val = val                                       # Get best_value.
                    best_move = child                                    # Assign best move.
                    alpha = max(alpha, best_val)                         # Update alpha.
                    if beta <= alpha:                                    # Beta cut-off.
                        break
            return best_move, best_val, timeout                          # Return best_move and associated Utility.

        else:                                                            # Min's turn.
            best_val = float("inf")                                      # Init best_val to inf.
            best_move = ()                                               # Init best_move to None.
            for child in children:                                       # Iterate for all Child Nodes.
                new_game = game.copy()                                   # Copy board.
                new_game.__apply_move__(child)                           # Apply move in board copy.
                temp_move, val, timeout = \
                    self.iteralphabeta(new_game, time_left, depth-1,
                                       alpha, beta, True)                # Run Iterative alphabeta recursively.
                if timeout == 1:
                    break                                                # Break if we timeout.
                if val < best_val:                                       # Check if value is worse than previous worst.
                    best_val = val                                       # Get best_value.
                    best_move = child                                    # Assign best move.
                    beta = min(beta, best_val)                           # Update beta.
                    if beta <= alpha:                                    # Alpha cut-off.
                        break
            return best_move, best_val, timeout                          # Return best_move and associated Utility.


        # TODO: finish this function!
        raise NotImplementedError
        return best_move, val

    def idalphabeta(self, game, time_left, depth=4, maximizing_player=True):
        """Function that calls the iterative alphabeta algorithm
        after incrementing the depth at each pass"""
        best_move = ()
        best_val = float("-inf")
        for i in range(0, depth+1, 1):                                   # Iterate from depth 0 to self.depth.
            temp_move, temp_val, timeout = \
                self.iteralphabeta(game, time_left, i, True)             # Call Iterative alpha-beta.
            if timeout == 0:                                             # If we don't timeout,
                best_move = temp_move                                    # return the best_move, and
                best_val = temp_val                                      # best_val
            else:                                                        # If we timeout, then return the previous
                break                                                    # best_move and best_val
        return best_move, best_val

    # def iteralphabetanodereorder(self, game, time_left, depth=4, alpha=float("-inf"), beta=float("inf"),
    #                              maximizing_player=True, children=(), level=0):
    #     """Implementation of the iterative alphabeta algorithm
    #
    #     Args:
    #         game (Board): A board and game state.
    #         time_left (function): Used to determine time left before timeout
    #         depth: Used to track how deep you are in the search tree
    #         alpha (float): Alpha value for pruning
    #         beta (float): Beta value for pruning
    #         maximizing_player (bool): True if maximizing player is active.
    #         children : list of depth, util, (move)
    #         level : count of level
    #
    #     Returns:
    #         (tuple, int): best_move, best_val
    #     """
    #     timeout = 0
    #     time = time_left()
    #
    #     if maximizing_player is True:
    #         legal_moves = [x[2] for x in children if x[0] == level]
    #
    #     if len(children) == 0:
    #         legal_moves = game.get_legal_moves()
    #
    #     if depth == 0 or len(children) == 0 or time < 20:                # Check if we are at the end of our search.
    #         best_move = game.__last_queen_move__                         # Assign the best move.
    #         best_val = self.eval_fn.score(game, maximizing_player)       # Calculate the Utility.
    #         if time < 20:                                                # Check to see if we timing out.
    #             timeout = 1
    #         children.append([level, best_val, best_move])
    #         return best_move, best_val, timeout, children              # If we are cutting-off, send the node's score.
    #
    #     elif maximizing_player is True:                                  # Max's turn.
    #         best_val = float("-inf")                                     # Init best_val to -inf.
    #         best_move = ()                                               # Init best_move to None.
    #         for child in children:                                       # Iterate for all Child Nodes.
    #             new_game = game.copy()                                   # Copy board.
    #             new_game.__apply_move__(child)                           # Apply move in board copy.
    #             temp_move, val, timeout = \
    #                 self.iteralphabetanodereorder(new_game, time_left,
    #                                               depth-1, alpha, beta,
    #                                               False, level+1)        # Run Iterative alphabeta recursively.
    #             if timeout == 1:
    #                 break                                                # Break if we timeout.
    #             children.append([level, best_val, best_move])
    #             if val > best_val:                                       # Check if value is better than previous best
    #                 best_val = val                                       # Get best_value.
    #                 best_move = child                                    # Assign best move.
    #                 alpha = max(alpha, best_val)                         # Update alpha.
    #                 if beta <= alpha:                                    # Beta cut-off.
    #                     break
    #         return best_move, best_val, timeout, children                # Return best_move and associated Utility.
    #
    #     else:                                                            # Min's turn.
    #         best_val = float("inf")                                      # Init best_val to inf.
    #         best_move = ()                                               # Init best_move to None.
    #         for child in children:                                       # Iterate for all Child Nodes.
    #             new_game = game.copy()                                   # Copy board.
    #             new_game.__apply_move__(child)                           # Apply move in board copy.
    #             temp_move, val, timeout = \
    #                 self.iteralphabetanodereorder(new_game, time_left,
    #                                               depth-1, alpha, beta,
    #                                               True, level+1)         # Run Iterative alphabeta recursively.
    #             if timeout == 1:
    #                 break                                                # Break if we timeout.
    #             children.append([level, best_val, best_move])
    #             if val < best_val:                                      # Check if value is worse than previous worst.
    #                 best_val = val                                       # Get best_value.
    #                 best_move = child                                    # Assign best move.
    #                 beta = min(beta, best_val)                           # Update beta.
    #                 if beta <= alpha:                                    # Alpha cut-off.
    #                     break
    #         return best_move, best_val, timeout, children                # Return best_move and associated Utility.
    #
    #
    #     # TODO: finish this function!
    #     raise NotImplementedError
    #     return best_move, val
    #
    # def idalphabetanodereorder(self, game, time_left, depth=4, maximizing_player=True):
    #     """Function that calls the iterative alphabeta algorithm
    #     after incrementing the depth at each pass"""
    #     best_move = ()
    #     best_val = float("-inf")
    #     children = []
    #     for i in range(0, depth+1, 1):                                   # Iterate from depth 0 to self.depth.
    #         temp_move, temp_val, timeout, children = \
    #             self.iteralphabeta(game, time_left, i, True, children)   # Call Iterative alpha-beta.
    #         if timeout == 0:                                             # If we don't timeout,
    #             best_move = temp_move                                    # return the best_move, and
    #             best_val = temp_val                                      # best_val
    #         else:                                                        # If we timeout, then return the previous
    #             break                                                    # best_move and best_val
    #         maxchildren = filter(lambda x: x[0] % 2 == 1, children)
    #         maxchildren = sorted(maxchildren, key=lambda x: x[1], reverse=True)
    #         minchildren = filter(lambda x: x[0] % 2 == 0, children)
    #         minchildren = sorted(minchildren, key=lambda x: x[1])
    #         children = maxchildren + minchildren
    #     return best_move, best_val

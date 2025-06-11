# operation_module.py
# (版本：最终版 - 采用高效的二维空间BFS寻路)

from collections import deque
import copy

# 确保导入了Piece和TETROMINOES
from tetrominoes import Piece, TETROMINOES

def generate_move_sequence(game, target_placement_info):
    """
    [Final Optimized Version] Finds the shortest sequence of moves using an
    efficient BFS that searches only in the (x, rotation) space.
    """
    start_piece = game.current_piece
    _target_type, target_rot, target_x, target_y = target_placement_info

    # --- BFS for Rotation and Horizontal Movement ONLY ---
    # The state for our search is (x, rotation).
    start_state = (start_piece.x, start_piece.rotation)

    # queue stores tuples of: ((current_x, current_rot), path_so_far)
    queue = deque([(start_state, [])])
    visited = {start_state}

    path_to_target_config = None

    # Limit search depth to prevent extreme cases on very cluttered boards.
    max_bfs_steps = 2500
    steps_taken = 0

    while queue and steps_taken < max_bfs_steps:
        (current_x, current_rot), path = queue.popleft()
        steps_taken += 1

        # --- Goal Check: Have we reached the target column and rotation? ---
        if current_x == target_x and current_rot == target_rot:
            # We must also verify that from this state, a hard drop is possible and leads to the correct y.
            # This is a crucial validation step.
            sim_piece_at_goal = Piece(current_x, start_piece.y, start_piece.type, current_rot)
            if game._is_valid_position(sim_piece_at_goal.shape_coords, current_x, sim_piece_at_goal.y):
                y_after_drop = sim_piece_at_goal.y
                while game._is_valid_position(sim_piece_at_goal.shape_coords, current_x, y_after_drop + 1):
                    y_after_drop += 1

                if y_after_drop == target_y:
                    path_to_target_config = path
                    break # Path found and validated!

        # --- Expand to next possible states (left, right, rotate_cw, rotate_ccw) ---
        sim_piece = Piece(current_x, start_piece.y, start_piece.type, current_rot)

        # Direction: dx, dr (delta_x, delta_rotation)
        possible_moves = {
            'left':       (-1, 0),
            'right':      (1, 0),
            'rotate_cw':  (0, 1),
            'rotate_ccw': (0, -1)
        }

        for move_name, (dx, dr) in possible_moves.items():
            # Create a copy to simulate the next move
            next_piece = copy.deepcopy(sim_piece)

            if dr != 0: # It's a rotation
                if not next_piece.rotate(dr, game.board_width, game.board_height,
                                         lambda s,x,y: game._is_valid_position(s,x,y, game.grid)):
                    continue # Rotation failed
            else: # It's a translation
                next_piece.x += dx
                # For translation, we only need to check validity at the current height,
                # as the piece can potentially pass through narrow gaps before dropping.
                if not game._is_valid_position(next_piece.shape_coords, next_piece.x, next_piece.y, game.grid):
                    continue # Translation is blocked

            new_state = (next_piece.x, next_piece.rotation)
            if new_state not in visited:
                visited.add(new_state)
                new_path = path + [move_name]
                queue.append((new_state, new_path))

    # --- Assemble Final Path ---
    if path_to_target_config is not None:
        # A valid path for horizontal/rotation was found. Append hard_drop.
        return path_to_target_config + ['hard_drop']
    else:
        # If no path is found, the target is truly unreachable via a valid sequence.
        return None # Return None, the planner will discard this move.
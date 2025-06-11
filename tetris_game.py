# tetris_game.py
# (版本：真正完整的最终版 - 支持CNN, T-Spin, 启发式惩罚, 辅助头, 并包含所有渲染方法)

import pygame
import numpy as np
import random
import os
import torch
from collections import deque, defaultdict
from srs_data import KICK_DATA
from tetrominoes import TETROMINOES, COLORS, PIECE_TYPES, Piece, INITIAL_POSITIONS, GHOST_COLOR, EMPTY_COLOR, GRID_LINE_COLOR
import copy
from operation_module import generate_move_sequence

class TetrisGame:
    def __init__(self, config, render_mode=False):
        self.config = config
        self.board_width = config.get('board_width', 10)
        self.board_height = config.get('board_height', 20)
        self.block_size = config.get('block_size', 30)
        self.render_mode = render_mode

        # --- 加载所有需要的奖励和惩罚参数 ---
        self.tspin_reward = self.config.get('tspin_reward', 400)
        self.tspin_mini_reward = self.config.get('tspin_mini_reward', 100)
        self.tspin_single_reward = self.config.get('tspin_single_reward', 800)
        self.tspin_double_reward = self.config.get('tspin_double_reward', 1200)
        self.tspin_triple_reward = self.config.get('tspin_triple_reward', 1600)
        self.line_clear_reward = self.config.get('line_clear_reward', 100)
        self.tetris_reward_bonus = self.config.get('tetris_reward_bonus', 400)
        self.hole_penalty_factor = self.config.get('hole_penalty_factor', 0.0)
        self.height_penalty_factor = self.config.get('height_penalty_factor', 0.0)
        self.bumpiness_penalty_factor = self.config.get('bumpiness_penalty_factor', 0.0)
        self.piece_drop_reward = self.config.get('piece_drop_reward', 0.0)
        self.game_over_penalty = self.config.get('game_over_penalty', -500)

        if self.render_mode:
            self._init_pygame()
        self.reset()

    def reset(self):
        self.grid = self._create_grid()
        self.current_piece, self.next_piece, self.held_piece = None, None, None
        self.can_hold, self.game_over = True, False
        self.score, self.lines_cleared_total = 0, 0
        self.piece_bag = []
        self._fill_piece_bag()
        self._spawn_new_piece()

    def _init_pygame(self):
        if not pygame.get_init(): pygame.init()
        self.screen_width = self.board_width * self.block_size + self.config.get('info_panel_width', 200)
        self.screen_height = self.board_height * self.block_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("AI Tetris")
        try:
            self.font = pygame.font.SysFont('arial', 24)
            self.small_font = pygame.font.SysFont('arial', 18)
        except pygame.error:
            self.font = pygame.font.Font(None, 30)
            self.small_font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()

    def _create_grid(self):
        return [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]

    def _fill_piece_bag(self):
        if not self.piece_bag:
            self.piece_bag = random.sample(PIECE_TYPES, len(PIECE_TYPES))

    def _get_piece_from_bag(self):
        if not self.piece_bag: self._fill_piece_bag()
        return self.piece_bag.pop(0)

    def _spawn_new_piece(self):
        self.current_piece = self.next_piece or Piece(INITIAL_POSITIONS[self._get_piece_from_bag()][0], INITIAL_POSITIONS[self._get_piece_from_bag()][1], self._get_piece_from_bag())
        next_type = self._get_piece_from_bag()
        self.next_piece = Piece(INITIAL_POSITIONS[next_type][0], INITIAL_POSITIONS[next_type][1], next_type)
        if not self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, self.current_piece.y):
            self.game_over = True

    def _is_valid_position(self, shape_coords, piece_x, piece_y, grid=None):
        grid_to_check = grid if grid is not None else self.grid
        for r_offset, c_offset in shape_coords:
            r, c = piece_y + r_offset, piece_x + c_offset
            if not (0 <= c < self.board_width and 0 <= r < self.board_height and grid_to_check[r][c] == 0):
                return False
        return True

    def _place_piece_on_grid(self, piece, target_grid):
        new_grid = [row[:] for row in target_grid]
        color_val = PIECE_TYPES.index(piece.type) + 1
        for r_offset, c_offset in piece.shape_coords:
            r, c = piece.y + r_offset, piece.x + c_offset
            if 0 <= r < self.board_height and 0 <= c < self.board_width:
                new_grid[r][c] = color_val
        return new_grid

    def _lock_piece(self):
        reward_components = defaultdict(float)
        is_tspin = self._check_tspin_conditions(self.current_piece, self.grid)
        self.grid = self._place_piece_on_grid(self.current_piece, self.grid)
        lines_cleared = self._clear_lines()
        self.lines_cleared_total += lines_cleared
        self.can_hold = True
        height, holes, _, bumpiness = self._calculate_grid_metrics(self.grid)
        reward_components['hole_penalty'] = holes * self.hole_penalty_factor
        reward_components['height_penalty'] = height * self.height_penalty_factor
        reward_components['bumpiness_penalty'] = bumpiness * self.bumpiness_penalty_factor
        if is_tspin:
            reward_components['tspin_reward'] = self.tspin_reward
            if lines_cleared == 1: reward_components['tspin_single_reward'] = self.tspin_single_reward
            elif lines_cleared == 2: reward_components['tspin_double_reward'] = self.tspin_double_reward
            elif lines_cleared == 3: reward_components['tspin_triple_reward'] = self.tspin_triple_reward
            elif lines_cleared == 0: reward_components['tspin_mini_reward'] = self.tspin_mini_reward
        elif lines_cleared > 0:
            reward_components['line_clear_reward'] = lines_cleared * self.line_clear_reward
            if lines_cleared == 4: reward_components['tetris_reward_bonus'] = self.tetris_reward_bonus
        reward_components['piece_drop_reward'] = self.piece_drop_reward
        self._spawn_new_piece()
        is_terminal_step = self.game_over
        if is_terminal_step:
            reward_components['game_over_penalty'] = self.game_over_penalty
        reward_components['final_reward'] = sum(reward_components.values())
        return reward_components, is_terminal_step, lines_cleared

    def _clear_lines(self):
        """
        [Corrected Version] Clears completed lines from the grid and returns the count.
        This version builds a new grid to avoid index shifting errors.
        """
        # 1. Create a new grid, but only with the rows that are NOT full.
        new_grid = [row for row in self.grid if not all(cell != 0 for cell in row)]

        # 2. Calculate how many lines were cleared.
        lines_cleared = self.board_height - len(new_grid)

        if lines_cleared > 0:
            # 3. Create the required number of new empty rows.
            empty_rows = [[0 for _ in range(self.board_width)] for _ in range(lines_cleared)]

            # 4. Prepend the empty rows to the new grid to simulate blocks falling.
            self.grid = empty_rows + new_grid

        return lines_cleared

    def _calculate_grid_metrics(self, grid_state):
        heights = [next((self.board_height - r for r in range(self.board_height) if grid_state[r][c] != 0), 0) for c in range(self.board_width)]
        agg_height = sum(heights)
        holes = sum(1 for c in range(self.board_width) for r in range(self.board_height - heights[c] + 1, self.board_height) if grid_state[r][c] == 0)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(self.board_width - 1))
        return agg_height, holes, 0, bumpiness


    def get_all_possible_next_states_and_features(self, for_piece=None):
        """
        [最终整合版] Finds all possible final placements by:
        1. Exhaustively simulating all simple drops and SRS kicks.
        2. Verifying that a valid path exists to each potential placement using BFS.
        This ensures every considered move is 100% legal and reachable.
        """
        piece_to_eval = for_piece if for_piece is not None else self.current_piece
        if not piece_to_eval or self.game_over:
            return []

        # Stage 1: Find all THEORETICALLY possible placements (including all kicks)
        # Use a set to automatically handle duplicates of (x, y, rot)
        theoretically_possible_placements = set()

        # Iterate through all possible starting rotations
        for start_rot in range(len(TETROMINOES[piece_to_eval.type])):
            # Iterate through all possible starting columns
            for start_x in range(-2, self.board_width + 2):
                sim_piece = Piece(start_x, 0, piece_to_eval.type, start_rot)

                # Ensure the piece can spawn at the top without collision
                if not self._is_valid_position(sim_piece.shape_coords, sim_piece.x, 0):
                    if not self._is_valid_position(sim_piece.shape_coords, sim_piece.x, 1):
                        continue
                    else:
                        sim_piece.y = 1

                # --- Find placement via simple Hard Drop ---
                hard_drop_y = sim_piece.y
                while self._is_valid_position(sim_piece.shape_coords, sim_piece.x, hard_drop_y + 1):
                    hard_drop_y += 1
                theoretically_possible_placements.add((sim_piece.x, hard_drop_y, sim_piece.rotation))

                # --- Find placements via Rotation + Kicks ---
                # From the position right above the hard drop spot, try to rotate
                pre_drop_y = hard_drop_y

                for direction in [1, -1]: # Clockwise and Counter-clockwise
                    num_rotations = len(TETROMINOES[piece_to_eval.type])
                    if num_rotations <= 1: continue

                    target_rot = (start_rot + direction + num_rotations) % num_rotations
                    kick_tests = KICK_DATA[piece_to_eval.type].get((start_rot, target_rot), [])

                    for dx, dy in kick_tests:
                        final_x = sim_piece.x + dx
                        final_y = pre_drop_y - dy # Pygame Y-axis is inverted

                        target_shape_coords = TETROMINOES[piece_to_eval.type][target_rot]
                        if self._is_valid_position(target_shape_coords, final_x, final_y):
                            # Found a valid placement through a kick, now find its resting spot
                            final_resting_y = final_y
                            while self._is_valid_position(target_shape_coords, final_x, final_resting_y + 1):
                                final_resting_y += 1
                            theoretically_possible_placements.add((final_x, final_resting_y, target_rot))
                            # No 'break' here, to test all 5 kicks exhaustively

        # Stage 2: Filter theoretical placements by checking actual reachability
        final_results = []
        visited_hashes = set()

        for x, y, rot in theoretically_possible_placements:
            target_info = (piece_to_eval.type, rot, x, y)

            # Use the BFS pathfinder to see if a valid sequence of moves exists
            path = generate_move_sequence(self, target_info)

            # Only if a path is found, do we consider this a truly valid move
            if path is not None:
                # --- This logic now only runs for reachable placements ---
                temp_grid = self._place_piece_on_grid(Piece(x, y, piece_to_eval.type, rot), self.grid)
                grid_hash = tuple(map(tuple, temp_grid))
                if grid_hash in visited_hashes: continue
                visited_hashes.add(grid_hash)

                height, holes, _, _ = self._calculate_grid_metrics(temp_grid)
                _, completed_lines = self._simulate_line_clear(temp_grid)

                aux_labels = {'lines': completed_lines, 'holes': holes, 'height': height}
                final_coords = [(y + ro, x + co) for ro, co in TETROMINOES[piece_to_eval.type][rot]]
                cnn_input = self._convert_grid_to_cnn_input(temp_grid, final_coords)

                action_info = (piece_to_eval.type, rot, x, y)
                final_results.append((action_info, cnn_input, aux_labels))

        return final_results

    def _simulate_line_clear(self, grid):
        new_grid = [row[:] for row in grid]
        lines = [r for r, row in enumerate(new_grid) if all(cell for cell in row)]
        if not lines: return new_grid, 0
        for r_idx in sorted(lines, reverse=True):
            del new_grid[r_idx]
            new_grid.insert(0, [0] * self.board_width)
        return new_grid, len(lines)

    def apply_ai_chosen_action(self, action_info):
        if self.game_over: return defaultdict(float), True, 0

        _piece_type, target_rotation, target_x, target_y = action_info

        # --- 新增：最终安全校验 ---
        # Before teleporting, do a final validation check.
        # This prevents any bug in the planner from creating an illegal state.
        temp_piece_for_check = Piece(target_x, target_y, _piece_type, target_rotation)
        if not self._is_valid_position(temp_piece_for_check.shape_coords, temp_piece_for_check.x, temp_piece_for_check.y):
            print(f"FATAL ERROR: AI chose an illegal move {action_info}. Ending game.")
            self.game_over = True
            # Return a response consistent with game over
            reward_components = defaultdict(float)
            reward_components['game_over_penalty'] = self.game_over_penalty
            reward_components['final_reward'] = self.game_over_penalty
            return reward_components, True, 0

        # --- 原有逻辑 ---
        self.current_piece.type = _piece_type
        self.current_piece.rotation = target_rotation
        self.current_piece.shape_coords = TETROMINOES[self.current_piece.type][target_rotation]
        self.current_piece.x = target_x
        self.current_piece.y = target_y

        return self._lock_piece()

    def execute_atomic_action(self, action):
        """
        [Fully Robust Version] Executes a single atomic game action and provides
        debug feedback for any failed attempts.
        """
        if self.game_over or not self.current_piece:
            return

        action_executed = False

        if action == 'left':
            if self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x - 1, self.current_piece.y):
                self.current_piece.x -= 1
                action_executed = True
            else:
                print("Debug: Move left failed (blocked).")

        elif action == 'right':
            if self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x + 1, self.current_piece.y):
                self.current_piece.x += 1
                action_executed = True
            else:
                print("Debug: Move right failed (blocked).")

        elif action == 'rotate_cw':
            if self.current_piece.rotate(1, self.board_width, self.board_height, lambda s,x,y: self._is_valid_position(s,x,y)):
                action_executed = True
            else:
                print("Debug: Clockwise rotation failed (no valid kick found).")

        elif action == 'rotate_ccw':
            if self.current_piece.rotate(-1, self.board_width, self.board_height, lambda s,x,y: self._is_valid_position(s,x,y)):
                action_executed = True
            else:
                print("Debug: Counter-clockwise rotation failed (no valid kick found).")

        elif action == 'soft_drop':
            if self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, self.current_piece.y + 1):
                self.current_piece.y += 1
                action_executed = True
            else:
                print("Debug: Soft drop failed (blocked), locking piece...")
                self._lock_piece() # This is a terminal action for a piece, not a failure.
                action_executed = True # Locking is also a successful outcome of soft-dropping at the bottom.

        elif action == 'hard_drop':
            while self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, self.current_piece.y + 1):
                self.current_piece.y += 1
            self._lock_piece()
            action_executed = True

        elif action == 'hold':
            if self._hold_piece():
                action_executed = True
                print("Debug: Hold successful.")
            else:
                print("Debug: Hold failed (cannot hold now).")

        else:
            print(f"Warning: execute_atomic_action received unknown action '{action}'")


    def _hold_piece(self):
        """
        [Robust Version] Executes the Hold action.
        Returns True on success, False on failure.
        """
        if not self.can_hold:
            return False # Action failed

        if self.held_piece is None:
            self.held_piece = Piece(0, 0, self.current_piece.type, 0)
            self._spawn_new_piece()
        else:
            held_piece_type_before_swap = self.held_piece.type
            self.held_piece = Piece(0, 0, self.current_piece.type, 0)
            spawn_pos = INITIAL_POSITIONS[held_piece_type_before_swap]
            self.current_piece = Piece(spawn_pos[0], spawn_pos[1], held_piece_type_before_swap)

            if not self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, self.current_piece.y):
                self.game_over = True

        self.can_hold = False
        return True # Action successful

    def _check_tspin_conditions(self, piece, grid):
        if piece.type != 'T': return False
        x, y = piece.x, piece.y
        corners = [(y - 1, x - 1), (y - 1, x + 1), (y + 1, x - 1), (y + 1, x + 1)]
        occupied_corners = sum(1 for r, c in corners if not (0 <= c < self.board_width and 0 <= r < self.board_height) or grid[r][c] != 0)
        return occupied_corners >= 3

    def _convert_grid_to_cnn_input(self, grid, placement_coords):
        board_shape = (self.board_height, self.board_width)
        channels = np.zeros((self.config['cnn_input_channels'], *board_shape), dtype=np.float32)
        heights_per_col = [next((r for r in range(self.board_height) if grid[r][c] != 0), self.board_height) for c in range(self.board_width)]
        for r in range(self.board_height):
            for c in range(self.board_width):
                if grid[r][c] != 0:
                    channels[0, r, c] = 1.0 # Filled channel
                elif r > heights_per_col[c]:
                    channels[1, r, c] = 1.0 # Holes channel
                if self.board_height - r <= heights_per_col[c]:
                    channels[2, r, c] = 1.0 # Height channel
        for r, c in placement_coords:
            if 0 <= r < self.board_height and 0 <= c < self.board_width:
                channels[3, r, c] = 1.0 # Placement channel
        return torch.from_numpy(channels).unsqueeze(0)

    def get_ghost_piece_y(self):
        if not self.current_piece: return 0
        y = self.current_piece.y
        while self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, y + 1):
            y += 1
        return y

    def render(self):
        if not self.render_mode: return
        self.screen.fill((20, 20, 20))
        self.draw_board()
        self.draw_gridlines()
        if self.current_piece and not self.game_over:
            ghost_y = self.get_ghost_piece_y()
            ghost_piece = Piece(self.current_piece.x, ghost_y, self.current_piece.type, self.current_piece.rotation)
            self.draw_piece(ghost_piece, is_ghost=True)
        self.draw_piece(self.current_piece)
        self.draw_info_panel()
        if self.game_over:
            s = pygame.Surface((self.board_width * self.block_size, self.board_height * self.block_size), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            text_surf = self.font.render("GAME OVER", True, (255, 60, 60))
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.board_width*self.block_size/2, self.board_height*self.block_size/2)))
        pygame.display.flip()
        self.clock.tick(self.config.get('fps', 30))

    def draw_board(self):
        for r, row in enumerate(self.grid):
            for c, cell_val in enumerate(row):
                pygame.draw.rect(self.screen, COLORS[PIECE_TYPES[cell_val-1]] if cell_val != 0 else EMPTY_COLOR,
                                 (c*self.block_size, r*self.block_size, self.block_size, self.block_size), 0)

    def draw_gridlines(self):
        for x in range(0, self.board_width * self.block_size + 1, self.block_size):
            pygame.draw.line(self.screen, GRID_LINE_COLOR, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height + 1, self.block_size):
            pygame.draw.line(self.screen, GRID_LINE_COLOR, (0, y), (self.board_width * self.block_size, y))

    def draw_piece(self, piece, is_ghost=False):
        if not piece: return
        shape_coords = piece.shape_coords
        for r_offset, c_offset in shape_coords:
            x, y = (piece.x + c_offset) * self.block_size, (piece.y + r_offset) * self.block_size
            if is_ghost:
                pygame.draw.rect(self.screen, GHOST_COLOR, (x, y, self.block_size, self.block_size), 2)
            else:
                pygame.draw.rect(self.screen, piece.color, (x, y, self.block_size, self.block_size), 0)

    def draw_info_panel(self):
        panel_x = self.board_width * self.block_size + 20
        y = 20
        line_height = self.font.get_linesize() + 5
        self.screen.blit(self.font.render(f"Score: {self.score}", True, (255,255,255)), (panel_x, y)); y+=line_height
        self.screen.blit(self.font.render(f"Lines: {self.lines_cleared_total}", True, (255,255,255)), (panel_x, y)); y+=line_height*1.5
        self.screen.blit(self.font.render("Next:", True, (255,255,255)), (panel_x, y)); y+=line_height
        if self.next_piece: self.draw_piece(Piece(self.board_width + 2, (y / self.block_size) - self.next_piece.y, self.next_piece.type))
        y += 4 * self.block_size
        self.screen.blit(self.font.render("Hold:", True, (255,255,255)), (panel_x, y)); y+=line_height
        if self.held_piece: self.draw_piece(Piece(self.board_width + 2, (y / self.block_size) - self.held_piece.y, self.held_piece.type))

    def get_screenshot(self, path):
        if self.render_mode:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pygame.image.save(self.screen, path)
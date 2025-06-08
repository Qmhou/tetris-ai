# tetris_game.py
# (版本：已包含“预消行数”和“连击”特征及奖励)

import pygame
import numpy as np
import random
import os
import torch
# 假设 tetrominoes.py 包含:
# PIECE_TYPES, COLORS, TETROMINOES (形状), INITIAL_POSITIONS,
# GHOST_COLOR (例如 (100, 100, 100, 128)), EMPTY_COLOR, GRID_LINE_COLOR,
# 以及 Piece 类 (包含其旋转逻辑)
from tetrominoes import TETROMINOES, COLORS, PIECE_TYPES, Piece, INITIAL_POSITIONS, GHOST_COLOR, EMPTY_COLOR, GRID_LINE_COLOR

class TetrisGame:
    def __init__(self, config, render_mode=False):
        self.config = config
        self.board_width = config.get('board_width', 10)
        self.board_height = config.get('board_height', 20)
        self.block_size = config.get('block_size', 30)
        self.render_mode = render_mode

        # 初始化游戏状态
        self.grid = self._create_grid()
        self.current_piece = None
        self.next_piece = None
        self.held_piece = None
        self.can_hold = True
        self.score = 0
        self.lines_cleared_total = 0
        self.game_over = False
        self.current_level = 0
        self.combo_count = 0  # << 新增: 连击计数器
        self.max_combo_this_game = 0
        # 初始化方块袋
        self.piece_bag = []
        self._fill_piece_bag()
        self._spawn_new_piece()

        # << 新增: 加载井区占有惩罚因子 >>
        self.well_occupancy_penalty_factor = float(config.get('well_occupancy_penalty_factor', -2.0))
        
        # << 新增: 加载特征缩放因子 >>
        self.scaling_factors = config.get('feature_scaling_factors', {
            'max_height': 20.0, 'max_holes': 50.0, 'max_generalized_wells': 100.0,
            'max_bumpiness': 40.0, 'max_completed_lines': 4.0, 'max_combo': 15.0,
            'max_well_occupancy': 40.0
        }) # 如果config中没有，则使用默认值

        # 从配置加载奖励参数
        self.height_penalty_factor = float(config.get('height_penalty_factor', -0.1))
        self.hole_penalty_factor = float(config.get('hole_penalty_factor', -2.5))
        self.bumpiness_penalty_factor = float(config.get('bumpiness_penalty_factor', -0.8))
        self.lines_cleared_reward_factor = float(config.get('lines_cleared_reward_factor', 200))
        self.game_over_penalty_val = float(config.get('game_over_penalty', -120))
        self.piece_drop_reward = float(config.get('piece_drop_reward', -0.01))
        self.clear_line_scores = config.get('clear_line_scores', [0, 100, 200, 300, 400])
        # << 新增: 加载新的计分参数 >>
        self.combo_score_base = float(config.get('combo_score_base', 100))
        self.combo_multiplier_schedule = config.get('combo_multiplier_schedule', [0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        # ...
        # << 新增: 连击奖励相关参数 >>
        self.combo_base_reward = float(config.get('combo_base_reward', 50))
        # 连击加成表: 对应连击数 1, 2, 3, 4, 5, 6, 7, 8, 9, 10+
        self.combo_bonus_schedule = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

        # 初始化Pygame渲染资源 (仅在渲染模式下)
        if self.render_mode:
            if not pygame.get_init():
                pygame.init()
            
            self.screen_width = self.board_width * self.block_size + config.get('info_panel_width', 200)
            self.screen_height = self.board_height * self.block_size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("AI Tetris")
            try:
                self.font = pygame.font.SysFont('arial', 24)
                self.small_font = pygame.font.SysFont('arial', 18)
            except pygame.error:
                print("Arial 字体未找到，使用 Pygame 默认字体。")
                self.font = pygame.font.Font(None, 30)
                self.small_font = pygame.font.Font(None, 24)
            self.ocr_font_color = (220, 220, 220)
            self.clock = pygame.time.Clock()

    def reset(self):
        """重置游戏状态以便开始新的一局。"""
        self.grid = self._create_grid()
        self.score = 0
        self.lines_cleared_total = 0
        self.game_over = False
        self.current_level = 0
        self.can_hold = True
        self.held_piece = None
        self.combo_count = 0  # << 新增：重置连击计数
        self.max_combo_this_game = 0

        self.piece_bag.clear()
        self._fill_piece_bag()
        self._spawn_new_piece()

    def _create_grid(self, fill_value=0):
        return [[fill_value for _ in range(self.board_width)] for _ in range(self.board_height)]

    def _fill_piece_bag(self):
        """如果方块袋为空，则重新填充所有7种方块并打乱顺序。"""
        if not self.piece_bag:
            self.piece_bag = random.sample(PIECE_TYPES, len(PIECE_TYPES))

    def _get_piece_from_bag(self):
        if not self.piece_bag:
            self._fill_piece_bag()
        return self.piece_bag.pop(0)

    def _spawn_new_piece(self):
            """生成新的当前方块和下一个方块。"""
            if self.next_piece is None:
                # 仅在游戏第一块时执行
                piece_type1 = self._get_piece_from_bag()
                self.current_piece = Piece(INITIAL_POSITIONS[piece_type1][0], INITIAL_POSITIONS[piece_type1][1], piece_type1)
            else:
                # 正常流程：Next方块成为Current方块
                self.current_piece = self.next_piece

            # 从方块袋中为Next方块抽取一个新类型
            piece_type2 = self._get_piece_from_bag()
            self.next_piece = Piece(INITIAL_POSITIONS[piece_type2][0], INITIAL_POSITIONS[piece_type2][1], piece_type2)

            # 检查新生成的方块是否有效，否则游戏结束
            if not self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, self.current_piece.y):
                self.game_over = True

    def _is_valid_position(self, shape_coords, piece_x, piece_y, grid_to_check=None):
        """检查给定形状和位置是否有效（未出界且未与现有方块碰撞）。"""
        current_grid = grid_to_check if grid_to_check is not None else self.grid
        for r_offset, c_offset in shape_coords:
            r, c = piece_y + r_offset, piece_x + c_offset
            if not (0 <= c < self.board_width and 0 <= r < self.board_height and current_grid[r][c] == 0):
                return False
        return True

    def _place_piece_on_grid(self, piece_obj, target_grid):
        """在给定的网格副本上放置一个方块，并返回新网格。"""
        new_grid = [row[:] for row in target_grid]
        piece_color_val = PIECE_TYPES.index(piece_obj.type) + 1 
        for r_offset, c_offset in piece_obj.shape_coords:
            r, c = piece_obj.y + r_offset, piece_obj.x + c_offset
            if 0 <= r < self.board_height and 0 <= c < self.board_width:
                 new_grid[r][c] = piece_color_val
        return new_grid
    

    def _lock_piece(self):
        """
        [CNN+T-Spin Version] Locks the piece, detects T-Spins, and calculates
        a reward based on the new T-Spin-focused reward system.
        """
        
        # --- 步骤 1: T-Spin 检测 ---
        # Must be done BEFORE placing the piece, using its final position
        # We need to know if the last move was a rotation, this requires tracking state
        # For simplicity, we'll use the robust 3-corner geometric check
        is_tspin = self._check_tspin_conditions(self.current_piece, self.grid)

        # --- 步骤 2: 锁定方块到网格 ---
        self.grid = self._place_piece_on_grid(self.current_piece, self.grid)

        # --- 步骤 3: 消行和计分 ---
        lines_cleared = self._clear_lines()
        self.lines_cleared_total += lines_cleared
        
        # --- 步骤 4: 计算奖励 (新的奖励逻辑) ---
        reward = 0
        score_change = 0 # You'd update self.score based on a new scoring system

        if is_tspin:
            reward += self.config['tspin_reward']
            if lines_cleared == 0:
                reward += self.config['tspin_mini_reward']
            elif lines_cleared == 1:
                reward += self.config['tspin_single_reward']
            elif lines_cleared == 2:
                reward += self.config['tspin_double_reward']
            elif lines_cleared == 3:
                reward += self.config['tspin_triple_reward']
        else: # Regular line clear
            if lines_cleared > 0:
                reward += lines_cleared * self.config['line_clear_reward']
                if lines_cleared == 4:
                    reward += self.config['tetris_reward_bonus']
        
        # --- 步骤 5: 游戏结束判断和新方块 ---
        self._spawn_new_piece()
        is_terminal_step = self.game_over 
        if is_terminal_step:
            reward += self.config['game_over_penalty']
        
        # This structure is simplified. You would return a dictionary of reward components for logging.
        return {"final_reward": reward}, is_terminal_step, lines_cleared

    def _clear_lines(self):
        lines_to_clear_indices = [r_idx for r_idx, row in enumerate(self.grid) if all(cell != 0 for cell in row)]
        if not lines_to_clear_indices:
            return 0
        for r_idx in sorted(lines_to_clear_indices, reverse=True):
            del self.grid[r_idx]
            self.grid.insert(0, [0 for _ in range(self.board_width)])
        return len(lines_to_clear_indices)

    # tetris_game.py (class TetrisGame)

    def _calculate_grid_metrics(self, grid_state):
        # ... (获取所有列的高度，以及计算 y_max_board, num_covered_holes, num_generalized_wells 的代码保持不变) ...
        column_heights = [0] * self.board_width
        for c in range(self.board_width):
            for r in range(self.board_height):
                if grid_state[r][c] != 0:
                    column_heights[c] = self.board_height - r
                    break
        
        y_max_board = max(column_heights) if any(h > 0 for h in column_heights) else 0

        num_covered_holes = 0
        for c in range(self.board_width):
            col_has_block_above = False
            for r in range(self.board_height):
                if grid_state[r][c] != 0:
                    col_has_block_above = True
                elif col_has_block_above and grid_state[r][c] == 0:
                    num_covered_holes += 1
        
        total_cells_in_bounding_box = self.board_width * y_max_board
        num_filled_cells_in_bounding_box = 0
        if y_max_board > 0:
            for r_scan in range(self.board_height - y_max_board, self.board_height):
                for c_scan in range(self.board_width):
                    if grid_state[r_scan][c_scan] != 0:
                        num_filled_cells_in_bounding_box += 1
        
        empty_cells_in_bounding_box = total_cells_in_bounding_box - num_filled_cells_in_bounding_box
        num_generalized_wells = empty_cells_in_bounding_box - num_covered_holes

        # --- 快捷崎岖度计算 (排除最右侧两列) ---
        # 我们的棋盘有10列 (索引0到9)。
        # 我们只计算前8列（索引0到7）之间的崎岖度。
        # 这意味着我们只需要计算7个高度差 (0-1, 1-2, ..., 6-7)。
        # 所以循环的范围是 range(self.board_width - 3)
        
        bumpiness_value = 0
        # 原有代码: for i in range(self.board_width - 1):
        # 在此处替换为以下代码:
        for i in range(self.board_width - 3): # << 核心修改点：只计算前8列内部的崎岖度
            bumpiness_value += abs(column_heights[i] - column_heights[i+1])
        
        # 返回所有计算出的指标
        return y_max_board, num_covered_holes, num_generalized_wells, bumpiness_value

    def get_all_possible_next_states_and_features(self, for_piece=None):
        """
        [CNN+T-Spin+Hold适配版] Finds all possible final placements for a GIVEN piece,
        including T-Spins, and returns the resulting board state as a CNN-ready tensor.
        """
        # Determine which piece to evaluate based on the provided argument
        if for_piece is None:
            piece_to_eval = self.current_piece
        else:
            piece_to_eval = for_piece

        if not piece_to_eval or self.game_over:
            return []

        possible_placements = []
        visited_hashes = set()
        
        piece = piece_to_eval
        
        # Iterate through all rotations for the given piece
        for rot_idx in range(len(TETROMINOES[piece.type])):
            # Create a temporary piece for simulation to get its shape
            sim_piece = Piece(0, 0, piece.type, rot_idx)
            
            # Iterate through all possible columns (with some buffer for kicks)
            for c in range(-2, self.board_width + 2):
                sim_piece.x = c
                
                # Search for valid final Y positions, including tucked positions
                # This is an exhaustive search to find all possible placements
                for r in range(-2, self.board_height):
                    sim_piece.y = r
                    
                    # Check if the piece can be placed at this (x,y,rot)
                    if self._is_valid_position(sim_piece.shape_coords, sim_piece.x, sim_piece.y, self.grid):
                        
                        # Check if this is a "final" resting spot (cannot move down further)
                        if not self._is_valid_position(sim_piece.shape_coords, sim_piece.x, sim_piece.y + 1, self.grid):
                            
                            # Create the resulting grid state after placing the piece
                            temp_grid = self._place_piece_on_grid(sim_piece, self.grid)
                            
                            # Use a hash of the grid to avoid duplicate evaluations
                            grid_hash = tuple(map(tuple, temp_grid))
                            if grid_hash not in visited_hashes:
                                visited_hashes.add(grid_hash)
                                
                                # Get the absolute coordinates of the final piece for the CNN input
                                final_piece_coords = [(sim_piece.y + ro, sim_piece.x + co) for ro, co in sim_piece.shape_coords]
                                
                                # Convert the resulting grid to the multi-channel tensor for the CNN
                                cnn_input = self._convert_grid_to_cnn_input(temp_grid, final_piece_coords)
                                
                                # Store the action info and the resulting state tensor
                                action_info = (piece.type, rot_idx, c, r)
                                possible_placements.append((action_info, cnn_input))
        
        return possible_placements


    def _hold_piece(self):
        """
        [调试版] 执行Hold操作，并打印详细的执行流程日志。
        """
        print("\n[DEBUG] Hold动作被触发。")

        if not self.can_hold:
            print("[DEBUG] Hold失败：因为 self.can_hold 当前是 False。请先锁定一个方块再尝试。")
            return

        print("[DEBUG] Hold检查通过 (self.can_hold is True)，准备执行Hold操作。")
        original_current_piece_type = self.current_piece.type if self.current_piece else "None"
        print(f"[DEBUG] 当前操作的方块是: {original_current_piece_type}")

        if self.held_piece is None:
            # --- 情况1：Hold区为空 ---
            print("[DEBUG] 检测到Hold区为空。")
            self.held_piece = Piece(0, 0, original_current_piece_type, 0) 
            print(f"[DEBUG] 状态更新：self.held_piece 已被设置为 '{self.held_piece.type}'。")
            
            print("[DEBUG] 正在调用 _spawn_new_piece() 以获取下一个方块...")
            self._spawn_new_piece()
            new_current_piece_type = self.current_piece.type if self.current_piece else "None"
            print(f"[DEBUG] 状态更新：新的 self.current_piece 已变为 '{new_current_piece_type}'。")
        else:
            # --- 情况2：Hold区有方块，进行交换 ---
            held_piece_type_before_swap = self.held_piece.type
            print(f"[DEBUG] 检测到Hold区不为空，持有的是'{held_piece_type_before_swap}'。准备交换。")
            
            # 1. 将当前方块的类型放入Hold区
            self.held_piece = Piece(0, 0, original_current_piece_type, 0)
            print(f"[DEBUG] 状态更新：self.held_piece 已被设置为 '{self.held_piece.type}'。")

            # 2. 从之前暂存的类型，创建新的当前方块
            self.current_piece = Piece(
                INITIAL_POSITIONS[held_piece_type_before_swap][0],
                INITIAL_POSITIONS[held_piece_type_before_swap][1],
                held_piece_type_before_swap
            )
            print(f"[DEBUG] 状态更新：新的 self.current_piece 已变为 '{self.current_piece.type}'。")
            
            if not self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, self.current_piece.y):
                self.game_over = True
                print("[DEBUG] 警告：交换后新方块位置不合法，游戏结束。")
        
        # 将开关设为False
        self.can_hold = False
        print("[DEBUG] 操作完成，self.can_hold 已被设置为 False。")

    def _simulate_line_clear(self, grid_to_modify):
        grid_copy = [row[:] for row in grid_to_modify]
        lines_cleared = 0
        r_idx = self.board_height - 1
        while r_idx >= 0:
            if all(cell != 0 for cell in grid_copy[r_idx]):
                lines_cleared += 1
                del grid_copy[r_idx]
                grid_copy.insert(0, [0 for _ in range(self.board_width)])
            else:
                r_idx -= 1
        return grid_copy, lines_cleared

    def apply_ai_chosen_action(self, chosen_action_info):
        """执行AI选择的最佳落子方案。"""
        if self.game_over or not self.current_piece:
            dummy_components = {key: 0.0 for key in ["score_change", "height_penalty", "hole_penalty", "bumpiness_penalty", "lines_reward", "combo_reward", "game_over_penalty", "piece_drop_reward", "final_reward"]}
            return dummy_components, self.game_over, 0

        _piece_type, target_rotation, target_x, target_y = chosen_action_info
        
        if self.current_piece.type != _piece_type:
            print(f"警告: AI决策 ({_piece_type})与当前方块({self.current_piece.type})不符。")
        
        self.current_piece.rotation = target_rotation
        self.current_piece.shape_coords = TETROMINOES[self.current_piece.type][target_rotation]
        self.current_piece.x = target_x
        self.current_piece.y = target_y 

        return self._lock_piece()

    def get_ghost_piece_y(self):
        """计算并返回幽灵方块的Y坐标。"""
        if not self.current_piece or self.game_over:
            return self.current_piece.y if self.current_piece else 0
        
        ghost_y = self.current_piece.y
        while self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, ghost_y + 1, self.grid):
            ghost_y += 1
        return ghost_y
        
    # --- 渲染相关方法 ---
    def render(self, episode_num_for_eval=None, reward_params_for_eval=None):
        if not self.render_mode or not hasattr(self, 'screen') or not self.screen: return
        
        self.screen.fill((20, 20, 20)) 
        self.draw_board() 
        self.draw_gridlines()

        if self.current_piece and not self.game_over:
            ghost_y = self.get_ghost_piece_y()
            ghost_piece_instance = Piece(self.current_piece.x, ghost_y, self.current_piece.type, self.current_piece.rotation)
            self.draw_piece(ghost_piece_instance, is_ghost=True)

        if self.current_piece:
            self.draw_piece(self.current_piece)
        
        self.draw_info_panel()

        if episode_num_for_eval is not None and reward_params_for_eval is not None:
            self._draw_eval_info_on_screen(episode_num_for_eval, reward_params_for_eval)
        
        if self.game_over:
            game_over_surface = self.font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_surface.get_rect(center=((self.board_width * self.block_size) / 2, (self.board_height * self.block_size) / 2))
            self.screen.blit(game_over_surface, text_rect)

        pygame.display.flip()
        if hasattr(self, 'clock') and self.clock:
            self.clock.tick(self.config.get('fps', 30))

    def draw_gridlines(self):
        if not self.render_mode: return
        game_area_height = self.board_height * self.block_size
        for x in range(0, self.board_width * self.block_size + 1, self.block_size):
            pygame.draw.line(self.screen, GRID_LINE_COLOR, (x, 0), (x, game_area_height))
        for y in range(0, game_area_height + 1, self.block_size):
            pygame.draw.line(self.screen, GRID_LINE_COLOR, (0, y), (self.board_width * self.block_size, y))

    def draw_board(self):
        if not self.render_mode: return
        for r_idx, row in enumerate(self.grid):
            for c_idx, cell_val in enumerate(row):
                color = EMPTY_COLOR
                if cell_val != 0:
                    try: color = COLORS[PIECE_TYPES[cell_val-1]]
                    except IndexError: color = (255,255,255) 
                pygame.draw.rect(self.screen, color, (c_idx * self.block_size, r_idx * self.block_size, self.block_size, self.block_size), 0)
    
    def draw_piece(self, piece_obj, offset_x_grid=0, offset_y_grid=0, is_ghost=False):
        if not self.render_mode or not piece_obj: return
        
        actual_color = piece_obj.color
        
        if is_ghost:
            ghost_block_surface = pygame.Surface((self.block_size, self.block_size), pygame.SRCALPHA)
            try:
                ghost_fill_color = (*GHOST_COLOR[:3], GHOST_COLOR[3] if len(GHOST_COLOR) > 3 else 128)
                ghost_block_surface.fill(ghost_fill_color)
            except (NameError, AttributeError, TypeError, IndexError):
                ghost_block_surface.fill((100, 100, 100, 128))
            ghost_border_color = (200, 200, 200)

        for r_offset, c_offset in piece_obj.shape_coords:
            block_screen_x = (piece_obj.x + c_offset + offset_x_grid) * self.block_size
            block_screen_y = (piece_obj.y + r_offset + offset_y_grid) * self.block_size
            block_rect = pygame.Rect(block_screen_x, block_screen_y, self.block_size, self.block_size)
            if is_ghost:
                self.screen.blit(ghost_block_surface, block_rect.topleft)
                pygame.draw.rect(self.screen, ghost_border_color, block_rect, 1)
            else:
                pygame.draw.rect(self.screen, actual_color, block_rect, 0)


    def draw_info_panel(self):
        if not self.render_mode or not hasattr(self, 'screen') or not self.screen: return
        
        panel_x_start = self.board_width * self.block_size + 20
        current_y = 20
        line_spacing = self.font.get_linesize() + 5

        # 绘制得分和总消行数
        score_text_surface = self.font.render(f"Score: {self.score}", True, self.ocr_font_color)
        self.screen.blit(score_text_surface, (panel_x_start, current_y))
        current_y += line_spacing

        lines_text_surface = self.font.render(f"Lines: {self.lines_cleared_total}", True, self.ocr_font_color)
        self.screen.blit(lines_text_surface, (panel_x_start, current_y))
        current_y += line_spacing

        # --- 新增：绘制连击信息 ---
        combo_color = (255, 200, 0) if self.combo_count > 0 else self.ocr_font_color # 连击时用醒目颜色
        
        combo_text_surface = self.font.render(f"Combo: {self.combo_count}", True, combo_color)
        self.screen.blit(combo_text_surface, (panel_x_start, current_y))
        current_y += line_spacing

        max_combo_text_surface = self.font.render(f"Max Combo: {self.max_combo_this_game}", True, self.ocr_font_color)
        self.screen.blit(max_combo_text_surface, (panel_x_start, current_y))
        current_y += line_spacing * 1.5 # 在连击信息和Next预览之间留出更多空隙
        # --- 新增结束 ---

        # 绘制 Next Piece
        next_label_surf = self.font.render("Next:", True, self.ocr_font_color)
        self.screen.blit(next_label_surf, (panel_x_start, current_y))
        if self.next_piece:
            preview_box_origin_x_grid = (panel_x_start // self.block_size)
            preview_box_origin_y_grid = ((current_y + next_label_surf.get_height() + 5) // self.block_size)
            temp_next_piece = Piece(x=2, y=1, piece_type=self.next_piece.type, rotation_state=self.next_piece.rotation)
            self.draw_piece(temp_next_piece, offset_x_grid=preview_box_origin_x_grid, offset_y_grid=preview_box_origin_y_grid)
        current_y += next_label_surf.get_height() + 5 + (3 * self.block_size) # 为Next Piece预览区留出空间

        # 绘制 Hold Piece
        hold_label_surf = self.font.render("Hold:", True, self.ocr_font_color)
        self.screen.blit(hold_label_surf, (panel_x_start, current_y))
        if self.held_piece:
            preview_box_origin_x_grid = (panel_x_start // self.block_size)
            preview_box_origin_y_grid = ((current_y + hold_label_surf.get_height() + 5) // self.block_size)
            temp_hold_piece = Piece(x=2, y=1, piece_type=self.held_piece.type, rotation_state=self.held_piece.rotation)
            self.draw_piece(temp_hold_piece, offset_x_grid=preview_box_origin_x_grid, offset_y_grid=preview_box_origin_y_grid)


    def _draw_eval_info_on_screen(self, episode_num, reward_params):
        """在屏幕上绘制评估信息。"""
        panel_x = self.board_width * self.block_size + 20
        info_y_start = 320 # 可根据实际布局调整
        line_height_ocr = self.small_font.get_linesize() + 2
        
        txt_surf = self.small_font.render(f"Eval Episode: {episode_num}", True, self.ocr_font_color)
        self.screen.blit(txt_surf, (panel_x, info_y_start))
        info_y_start += line_height_ocr
        
        txt_surf = self.small_font.render("Parameters:", True, self.ocr_font_color)
        self.screen.blit(txt_surf, (panel_x, info_y_start))
        info_y_start += line_height_ocr
        
        param_display_names = {
            "height_penalty_factor": "H PFac", "hole_penalty_factor": "Hole PFac",
            "bumpiness_penalty_factor": "Bump PFac", "lines_cleared_reward_factor": "Lines RFac",
            "game_over_penalty": "GameOver P", "piece_drop_reward": "Drop R",
            "combo_base_reward": "Combo RBase"
        }
        for key, value in reward_params.items():
            display_name = param_display_names.get(key, key)
            value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
            if value_str.endswith(".00"): value_str = str(int(value))
            
            text_to_render = f"  {display_name}: {value_str}"
            txt_surf = self.small_font.render(text_to_render, True, self.ocr_font_color)
            self.screen.blit(txt_surf, (panel_x, info_y_start))
            info_y_start += line_height_ocr
            if info_y_start > self.screen_height - self.small_font.get_linesize(): break
    
    def get_screenshot(self, path):
        if self.render_mode and hasattr(self, 'screen') and self.screen:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            try:
                pygame.image.save(self.screen, path)
            except pygame.error as e:
                print(f"保存截图至 {path} 时发生错误: {e}")


    def execute_atomic_action(self, action_string):
        """
        [健壮版] 执行一个原子的、单步的动作，并处理未知输入。
        """
        if self.game_over or not self.current_piece:
            return

        if action_string == 'left':
            if self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x - 1, self.current_piece.y):
                self.current_piece.x -= 1
        elif action_string == 'right':
            if self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x + 1, self.current_piece.y):
                self.current_piece.x += 1
        elif action_string == 'rotate_cw':
            if hasattr(self.current_piece, 'rotate'):
                self.current_piece.rotate(1, self.board_width, self.board_height, lambda s,x,y: self._is_valid_position(s,x,y,self.grid))
        elif action_string == 'rotate_ccw':
            if hasattr(self.current_piece, 'rotate'):
                self.current_piece.rotate(-1, self.board_width, self.board_height, lambda s,x,y: self._is_valid_position(s,x,y,self.grid))
        elif action_string == 'soft_drop':
            if self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, self.current_piece.y + 1):
                self.current_piece.y += 1
            else:
                self._lock_piece()
        elif action_string == 'hard_drop':
            while self._is_valid_position(self.current_piece.shape_coords, self.current_piece.x, self.current_piece.y + 1):
                self.current_piece.y += 1
            self._lock_piece()
        elif action_string == 'hold':
            self._hold_piece()
        else:
            # << 新增：处理所有未知的操作字符串
            print(f"警告: execute_atomic_action 接收到未知操作 '{action_string}'")

    def _check_tspin_conditions(self, piece, grid):
        """
        Checks if a T-piece placement meets the geometric conditions for a T-Spin.
        This uses the 3-corner rule, which is a standard method.
        """
        if piece.type != 'T':
            return False

        # The four corners relative to the piece's pivot (x, y)
        # These are the corners of the 3x3 bounding box of the T-piece
        corners = [
            (piece.y - 1, piece.x - 1), (piece.y - 1, piece.x + 1),
            (piece.y + 1, piece.x - 1), (piece.y + 1, piece.x + 1)
        ]
        
        occupied_corners = 0
        for r, c in corners:
            # Check if the corner is out of bounds or occupied by another block
            if not (0 <= c < self.board_width and 0 <= r < self.board_height) or grid[r][c] != 0:
                occupied_corners += 1
        
        # If 3 or more corners are occupied, it's a T-Spin setup
        return occupied_corners >= 3

    def _convert_grid_to_cnn_input(self, grid, current_piece_coords):
        """
        Converts the game grid into a multi-channel tensor for the CNN.
        """
        # Initialize channels
        # Using numpy for faster manipulation
        board_shape = (self.board_height, self.board_width)
        filled_channel = np.zeros(board_shape, dtype=np.float32)
        hole_channel = np.zeros(board_shape, dtype=np.float32)
        height_channel = np.zeros(board_shape, dtype=np.float32)
        placement_channel = np.zeros(board_shape, dtype=np.float32)

        # Populate filled_channel and find holes
        for c in range(self.board_width):
            is_sky = True
            for r in range(self.board_height):
                if grid[r][c] != 0:
                    filled_channel[r, c] = 1.0
                    is_sky = False
                elif not is_sky: # An empty cell below a filled cell
                    hole_channel[r, c] = 1.0

        # Populate height_channel
        for c in range(self.board_width):
            for r in range(self.board_height):
                if filled_channel[r, c] == 1.0:
                    height_channel[r:, c] = 1.0 # Fill from the highest block downwards
                    break
        
        # Populate placement_channel
        for r_offset, c_offset in current_piece_coords:
            r, c = r_offset, c_offset
            if 0 <= r < self.board_height and 0 <= c < self.board_width:
                placement_channel[r, c] = 1.0

        # Stack channels and convert to PyTorch tensor
        # Note: No cropping here, can be done in network if desired
        cnn_input = np.stack([filled_channel, hole_channel, height_channel, placement_channel], axis=0)
        return torch.from_numpy(cnn_input).unsqueeze(0) # Add batch dimension

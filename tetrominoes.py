import pygame

# Define colors for tetrominoes
COLORS = {
    'I': (0, 255, 255),   # Cyan
    'O': (255, 255, 0),   # Yellow
    'T': (128, 0, 128),   # Purple
    'S': (0, 255, 0),     # Green
    'Z': (255, 0, 0),     # Red
    'J': (0, 0, 255),     # Blue
    'L': (255, 165, 0)    # Orange
}
GHOST_COLOR = (100, 100, 100, 100) # Semi-transparent gray for ghost piece
EMPTY_COLOR = (50, 50, 50) # Dark gray for empty grid cells
GRID_LINE_COLOR = (100, 100, 100)

# Define shapes of tetrominoes
# Each shape is a list of (row, col) offsets from a pivot point.
# The pivot is usually one of the blocks or the center of rotation.
# Standard SRS shapes and their rotation states (0, 1, 2, 3)
TETROMINOES = {
    'S': [
        [(0, 0), (0, 1), (1, -1), (1, 0)],  # 状态 0 / 2
        [(0, 0), (1, 0), (1, 1), (2, 1)]   # 状态 1 / 3
    ],
    'Z': [
        [(0, -1), (0, 0), (1, 0), (1, 1)], # 状态 0 / 2
        [(0, 1), (1, 0), (1, 1), (2, 0)]   # 状态 1 / 3
    ],
    'I': [
        [(1, 0), (1, 1), (1, 2), (1, 3)],  # 状态 0 / 2
        [(0, 2), (1, 2), (2, 2), (3, 2)]   # 状态 1 / 3
    ],
    'O': [
        [(0, 1), (0, 2), (1, 1), (1, 2)]   # O型方块只有一个状态
    ],
    'T': [
        [(0, 1), (1, 0), (1, 1), (1, 2)],  # 状态 0
        [(0, 1), (1, 1), (1, 2), (2, 1)],  # 状态 1
        [(1, 0), (1, 1), (1, 2), (2, 1)],  # 状态 2
        [(0, 1), (1, 0), (1, 1), (2, 1)]   # 状态 3
    ],
    'J': [
        [(0, 0), (1, 0), (1, 1), (1, 2)],  # 状态 0
        [(0, 1), (0, 2), (1, 1), (2, 1)],  # 状态 1
        [(1, 0), (1, 1), (1, 2), (2, 2)],  # 状态 2
        [(0, 1), (1, 1), (2, 0), (2, 1)]   # 状态 3
    ],
    'L': [
        [(0, 2), (1, 0), (1, 1), (1, 2)],  # 状态 0
        [(0, 1), (1, 1), (2, 1), (2, 2)],  # 状态 1
        [(1, 0), (1, 1), (1, 2), (2, 0)],  # 状态 2
        [(0, 0), (0, 1), (1, 1), (2, 1)]   # 状态 3
    ]
}

# Define initial spawn positions (col, row)
# Typically pieces spawn with their pivot point around (width/2, 0) or (width/2, 1)
# These positions need to be adjusted based on the pivot definition in TETROMINOES
# Assuming pivot is near top of piece or its reference point for these coords.
# (Col, Row) - Row 0 is top, Col is usually center.
INITIAL_POSITIONS = {
    'I': (5, 1), # Adjusted for its shape, can be higher if board has buffer rows
    'O': (5, 0),
    'T': (5, 1),
    'S': (5, 1),
    'Z': (5, 1),
    'J': (5, 1),
    'L': (5, 1)
}

PIECE_TYPES = list(TETROMINOES.keys())

class Piece:
    def __init__(self, x, y, piece_type, rotation_state=0):
        self.x = x
        self.y = y
        self.type = piece_type
        self.rotation = rotation_state
        self.color = COLORS[self.type]
        self.shape_coords = TETROMINOES[self.type][self.rotation]

    def get_absolute_block_coords(self):
        """Returns a list of (row, col) for each block of the piece on the board."""
        return [(self.y + dy, self.x + dx) for dy, dx in self.shape_coords]

# tetrominoes.py (修改 Piece.rotate 方法)
    def rotate(self, direction, board_width, board_height, is_valid_position_func):
        """
        使用SRS踢墙数据来旋转方块。
        direction: 1 表示顺时针, -1 表示逆时针。
        is_valid_position_func: 一个回调函数，用于检查新位置是否有效。
        """
        from srs_data import get_kick_offsets # 确保在函数内导入以避免循环依赖

        if self.type == 'O': # O型方块不旋转
            return True

        num_rotations = len(TETROMINOES[self.type])
        original_rotation = self.rotation
        
        # 计算目标旋转状态 (处理-1的情况)
        target_rotation = (original_rotation + direction + num_rotations) % num_rotations

        # 获取需要测试的踢墙偏移量列表
        kick_test_set = get_kick_offsets(self.type, original_rotation, target_rotation)
        
        # 获取旋转后的新形状坐标
        new_shape_coords = TETROMINOES[self.type][target_rotation]

        # 依次尝试每个踢墙偏移量
        for dx_kick, dy_kick in kick_test_set:
            # 计算应用偏移后的新位置
            # Pygame中Y轴向下为正，而SRS标准通常向上为正，所以y_kick需要反转符号
            potential_x = self.x + dx_kick
            potential_y = self.y - dy_kick # << 关键点：注意这里的减号！

            # 使用游戏主逻辑提供的函数检查新位置是否有效
            if is_valid_position_func(new_shape_coords, potential_x, potential_y):
                # 如果有效，更新方块状态并成功返回
                self.x = potential_x
                self.y = potential_y
                self.rotation = target_rotation
                self.shape_coords = new_shape_coords
                return True
        
        # 如果所有踢墙尝试都失败了，则旋转失败
        return False

if __name__ == '__main__':
    # Example:
    # Dummy validation function for testing piece rotation
    def dummy_is_valid(shape_coords, x, y):
        # In a real game, this checks against board boundaries and other pieces
        for dy_offset, dx_offset in shape_coords:
            if not (0 <= x + dx_offset < 10 and 0 <= y + dy_offset < 20):
                return False
        return True

    test_piece = Piece(5, 1, 'T')
    print(f"Initial T: pos=({test_piece.x},{test_piece.y}), rot={test_piece.rotation}, coords={test_piece.get_absolute_block_coords()}")
    test_piece.rotate(1, 10, 20, dummy_is_valid) # Clockwise
    print(f"Rotated T: pos=({test_piece.x},{test_piece.y}), rot={test_piece.rotation}, coords={test_piece.get_absolute_block_coords()}")
    test_piece.rotate(1, 10, 20, dummy_is_valid)
    print(f"Rotated T: pos=({test_piece.x},{test_piece.y}), rot={test_piece.rotation}, coords={test_piece.get_absolute_block_coords()}")

    test_i_piece = Piece(INITIAL_POSITIONS['I'][0], INITIAL_POSITIONS['I'][1], 'I')
    print(f"Initial I: pos=({test_i_piece.x},{test_i_piece.y}), rot={test_i_piece.rotation}, coords={test_i_piece.get_absolute_block_coords()}")
    test_i_piece.rotate(1, 10, 20, dummy_is_valid)
    print(f"Rotated I: pos=({test_i_piece.x},{test_i_piece.y}), rot={test_i_piece.rotation}, coords={test_i_piece.get_absolute_block_coords()}")
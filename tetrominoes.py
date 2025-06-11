import pygame
import copy
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
        [Robust Version] Attempts to rotate the piece using SRS kick data.
        It works on a temporary copy to prevent corrupting the piece's state on failure.
        """
        from srs_data import get_kick_offsets # Local import to avoid circular dependency

        # Create a temporary clone of the piece to perform tests on.
        # This prevents modifying the original piece if all rotation attempts fail.
        temp_piece = copy.deepcopy(self)
        
        original_rotation = temp_piece.rotation
        num_rotations = len(TETROMINOES[temp_piece.type])
        
        if num_rotations <= 1:
            return False # Cannot rotate O-pieces

        target_rotation = (original_rotation + direction + num_rotations) % num_rotations
        temp_piece.rotation = target_rotation
        temp_piece.shape_coords = TETROMINOES[temp_piece.type][target_rotation]

        kick_test_set = get_kick_offsets(self.type, original_rotation, target_rotation)
        
        for dx_kick, dy_kick in kick_test_set:
            # Apply kick to the temporary piece's original coordinates
            potential_x = self.x + dx_kick
            potential_y = self.y - dy_kick # Pygame Y-axis is inverted
            
            # Check if the new position is valid
            if is_valid_position_func(temp_piece.shape_coords, potential_x, potential_y):
                # If a valid rotation is found, update the REAL piece's state all at once.
                self.x = potential_x
                self.y = potential_y
                self.rotation = target_rotation
                self.shape_coords = temp_piece.shape_coords
                return True # Rotation successful
        
        # If all kick tests fail, do nothing to the original piece and return failure.
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
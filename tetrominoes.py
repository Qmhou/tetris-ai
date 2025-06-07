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
    'I': [
        [(0, -2), (0, -1), (0, 0), (0, 1)],  # State 0 (Horizontal, pivot on 3rd block from left)
        [(-2, 0), (-1, 0), (0, 0), (1, 0)],  # State 1 (Vertical, pivot on 3rd block from top)
        [(0, -1), (0, 0), (0, 1), (0, 2)],  # State 2 (Horizontal, pivot on 2nd block from left)
        [(-1, 0), (0, 0), (1, 0), (2, 0)]   # State 3 (Vertical, pivot on 2nd block from top)
    ],
    'O': [ # O-piece rotation doesn't change its shape visually in SRS for a 2x2 block representation
        [(0, 0), (0, 1), (1, 0), (1, 1)], # Pivot usually center of 2x2
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    ],
    'T': [
        [(0, -1), (0, 0), (0, 1), (-1, 0)], # State 0 (T pointing up, pivot on center bottom)
        [(0, 0), (-1, 0), (1, 0), (0, 1)],  # State 1 (T pointing right)
        [(0, -1), (0, 0), (0, 1), (1, 0)],  # State 2 (T pointing down)
        [(0, 0), (-1, 0), (1, 0), (0, -1)]  # State 3 (T pointing left)
    ],
    'S': [
        [(0, 0), (0, 1), (-1, -1), (-1, 0)], # State 0
        [(-1, 0), (0, 0), (0, 1), (1, 1)],   # State 1
        [(0, 0), (0, -1), (1, 1), (1, 0)],   # State 2 (Same as state 0 if pivot is (0,0) for S)
        # Redefining to match SRS common practice, using states 0 and 1, where 2 is 0 and 3 is 1
        [(0, 0), (0, 1), (-1, -1), (-1, 0)], # State 2 (same as 0)
        [(-1, 0), (0, 0), (0, 1), (1, 1)],   # State 3 (same as 1)
    ],
    'Z': [
        [(0, -1), (0, 0), (-1, 0), (-1, 1)],# State 0
        [(-1, 1), (0, 1), (0, 0), (1, 0)],  # State 1
        [(0, 1), (0, 0), (1, 0), (1, -1)],  # State 2 (Same as state 0 if pivot is (0,0) for Z)
        # Redefining to match SRS common practice
        [(0, -1), (0, 0), (-1, 0), (-1, 1)],# State 2 (same as 0)
        [(-1, 1), (0, 1), (0, 0), (1, 0)],  # State 3 (same as 1)
    ],
    'J': [
        [(0, -1), (0, 0), (0, 1), (-1, -1)],# State 0
        [(-1, 0), (0, 0), (1, 0), (-1, 1)], # State 1
        [(0, -1), (0, 0), (0, 1), (1, 1)],  # State 2
        [(-1, 0), (0, 0), (1, 0), (1, -1)]  # State 3
    ],
    'L': [
        [(0, -1), (0, 0), (0, 1), (-1, 1)], # State 0
        [(-1, 0), (0, 0), (1, 0), (1, 1)],  # State 1
        [(0, -1), (0, 0), (0, 1), (1, -1)], # State 2
        [(-1, 0), (0, 0), (1, 0), (-1, -1)] # State 3
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

    def rotate(self, direction, board_width, board_height, is_valid_position_func):
        """
        Rotates the piece using SRS kick data.
        `is_valid_position_func` is a callback to the TetrisGame to check board collisions.
        """
        from srs_data import get_kick_offsets # Avoid circular import at module level

        original_rotation = self.rotation
        original_x, original_y = self.x, self.y

        if self.type == 'O': # O-piece does not "kick" or change shape typically
            return True # Or handle its fixed rotation if defined differently

        num_rotations = len(TETROMINOES[self.type])
        if num_rotations <= 1: # For pieces like 'O' if only one shape is defined
             return True

        target_rotation = (self.rotation + direction) % num_rotations
        if target_rotation < 0: # Handle negative direction for clockwise
             target_rotation += num_rotations


        kick_test_set = get_kick_offsets(self.type, original_rotation, target_rotation)

        new_shape_coords = TETROMINOES[self.type][target_rotation]

        for dx_kick, dy_kick in kick_test_set:
            potential_x = self.x + dx_kick
            potential_y = self.y - dy_kick # Kicks are often defined as (x, -y) relative to TGM display

            if is_valid_position_func(new_shape_coords, potential_x, potential_y):
                self.x = potential_x
                self.y = potential_y
                self.rotation = target_rotation
                self.shape_coords = new_shape_coords
                return True
        
        # If no kick works, revert to original state (though this shouldn't happen often with SRS)
        self.x = original_x
        self.y = original_y
        return False # Rotation failed

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
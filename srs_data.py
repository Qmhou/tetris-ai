# Super Rotation System (SRS) Kick Data
# Based on https://tetris.wiki/Super_Rotation_System
# (0,0) is the base rotation point.
# These are offsets to try if the primary rotation fails.
# Format: {piece_type: { (from_rotation_state, to_rotation_state): [(dx1, dy1), (dx2, dy2), ...], ... }}

# For J, L, S, T, Z pieces (Commonly referred to as JLSTZ kicks)
# srs_data.py

# J, L, S, T, Z 型方块的踢墙数据
JLSTZ_KICKS = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],  # 0->1
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],    # 1->0
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],    # 1->2
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],# 2->1
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],   # 2->3
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)], # 3->2
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)], # 3->0
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],   # 0->3
}

# I 型方块的踢墙数据
I_KICKS = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],  # 0->1
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],    # 1->0
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],  # 1->2
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],    # 2->1
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],  # 2->3
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],    # 3->2
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],  # 3->0
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],    # 0->3
}

# O 型方块不进行踢墙
O_KICKS = {}

KICK_DATA = {
    'I': I_KICKS, 'J': JLSTZ_KICKS, 'L': JLSTZ_KICKS, 
    'S': JLSTZ_KICKS, 'T': JLSTZ_KICKS, 'Z': JLSTZ_KICKS, 'O': O_KICKS
}

def get_kick_offsets(piece_type, from_rotation, to_rotation):
    """获取指定旋转的踢墙偏移量列表。"""
    if piece_type in KICK_DATA:
        rotation_kicks = KICK_DATA[piece_type]
        if (from_rotation, to_rotation) in rotation_kicks:
            return rotation_kicks[(from_rotation, to_rotation)]
    return [(0,0)] # 默认返回无偏移

if __name__ == '__main__':
    # Example usage:
    print("T piece, 0 -> 1 kicks:", get_kick_offsets('T', 0, 1))
    print("I piece, 1 -> 2 kicks:", get_kick_offsets('I', 1, 2))
    print("O piece, 0 -> 1 kicks:", get_kick_offsets('O', 0, 1))
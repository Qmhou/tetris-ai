# srs_data.py

# J, L, S, T, Z piece kick data
JLSTZ_KICKS = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}

# I piece kick data
I_KICKS = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}

KICK_DATA = {
    'I': I_KICKS, 'J': JLSTZ_KICKS, 'L': JLSTZ_KICKS, 
    'S': JLSTZ_KICKS, 'T': JLSTZ_KICKS, 'Z': JLSTZ_KICKS, 'O': {}
}

def get_kick_offsets(piece_type, from_rotation, to_rotation):
    """
    A helper function to retrieve the list of kick offsets for a specific rotation.
    """
    # Look up the kick data table for the given piece type
    rotation_kicks = KICK_DATA.get(piece_type)
    if rotation_kicks:
        # Return the list of offsets for the specific rotation transition
        return rotation_kicks.get((from_rotation, to_rotation), [(0, 0)])
    
    # If piece_type has no kick data (like 'O'), return a default non-kick
    return [(0, 0)]
# operation_module.py

from tetrominoes import TETROMINOES

def generate_move_sequence(current_piece, target_placement_info):
    """
    根据当前方块状态和AI决策的目标落点，生成一个简单、直观的操作指令序列。
    这个序列主要用于在AI游戏模式下进行可视化回放。

    Args:
        current_piece (Piece): 游戏中的当前活动方块实例。
        target_placement_info (tuple): AI决策的最佳落点信息 
                                     (piece_type, target_rotation, target_x, target_y)。

    Returns:
        list: 一个包含操作指令字符串的列表，例如 ['rotate_cw', 'left', 'hard_drop']。
    """
    _, target_rotation, target_x, _ = target_placement_info
    
    op_sequence = []
    
    # 步骤 1: 计算并添加旋转指令
    current_rot = current_piece.rotation
    num_rotations = len(TETROMINOES[current_piece.type])
    
    if current_rot != target_rotation:
        # 计算顺时针和逆时针两种路径的次数
        rot_diff_cw = (target_rotation - current_rot + num_rotations) % num_rotations
        rot_diff_ccw = (current_rot - target_rotation + num_rotations) % num_rotations

        # 选择次数较少的那条路径
        if rot_diff_cw <= rot_diff_ccw:
            op_sequence.extend(['rotate_cw'] * rot_diff_cw)
        else:
            op_sequence.extend(['rotate_ccw'] * rot_diff_ccw)

    # 步骤 2: 计算并添加水平移动指令
    # 注意：这里我们基于方块旋转前的x坐标进行计算。
    # 实际执行时，由于是逐帧移动并有碰撞检测，即使旋转后有“踢墙”位移，
    # 后续的水平移动指令也会在新的位置上生效，最终引导方块到正确的列。
    x_diff = target_x - current_piece.x
    if x_diff > 0:
        op_sequence.extend(['right'] * x_diff)
    elif x_diff < 0:
        op_sequence.extend(['left'] * abs(x_diff))
        
    # 步骤 3: 添加最终的硬降指令
    op_sequence.append('hard_drop')
    
    return op_sequence
import os
import re

def create_project_from_text(
    structure_text: str,
    base_output_path: str = ".",
    create_container_folder_for_first_line: bool = False
):
    """
    根据文本描述的项目树结构创建文件夹和文件。

    Args:
        structure_text (str): 多行字符串，描述项目树结构。
        base_output_path (str): 生成项目的基路径。
        create_container_folder_for_first_line (bool): 
            如果为 True (默认)，则使用文本中的第一行 (例如 "tetris-ai/") 
            在 base_output_path 下创建一个同名容器目录。
            如果为 False，则忽略第一行作为容器目录，直接在 base_output_path 
            下创建后续的结构。
    """
    lines = structure_text.strip().split('\n')
    if not lines:
        print("项目结构文本为空，未执行任何操作。")
        return

    effective_working_root = ""
    lines_for_processing_children = []

    if create_container_folder_for_first_line:
        root_name_from_text = lines[0].strip().rstrip('/')
        if not root_name_from_text:
            print("错误：当需要为第一行创建目录时，未能从第一行解析出有效名称。")
            return
        effective_working_root = os.path.join(base_output_path, root_name_from_text)
        
        # 子项从第二行开始 (如果第二行是 '│'，则从第三行开始)
        if len(lines) > 1 and lines[1].strip() == '│':
            lines_for_processing_children = lines[2:]
        elif len(lines) > 1:
            lines_for_processing_children = lines[1:]
        else: # 只有一行 (根目录行)，没有子项
            lines_for_processing_children = []
    else: # 不为第一行创建容器目录
        effective_working_root = base_output_path
        first_line_skipped_info = lines[0].strip() if lines else "（空文本）"
        print(f"信息：将不为文本中的第一项 '{first_line_skipped_info}' 创建顶层容器目录。")
        print(f"内容将直接在 '{os.path.abspath(effective_working_root)}' 下创建。")
        
        # 即使不创建第一行的目录，也需要跳过第一行（及其可能的 '│' 行）来找到真正的子项
        if len(lines) > 1 and lines[1].strip() == '│':
            lines_for_processing_children = lines[2:]
        elif len(lines) > 1:
            lines_for_processing_children = lines[1:]
        else: # 只有一行被跳过，没有子项
            lines_for_processing_children = []

    # 创建有效的工作根目录 (可能是 base_output_path 本身或其子目录)
    try:
        os.makedirs(effective_working_root, exist_ok=True)
        print(f"项目结构将在以下根路径创建/确认: {os.path.abspath(effective_working_root)}")
    except OSError as e:
        print(f"创建有效工作根路径 '{effective_working_root}' 失败: {e}")
        return

    # `current_paths_at_level` 存储每个层级当前的父目录路径
    current_paths_at_level = {-1: effective_working_root}
    INDENT_STEP = 4 # 每个层级缩进的字符数，基于 "│   "

    # 遍历处理子项列表
    for line_number_in_children_list, line_content in enumerate(lines_for_processing_children):
        line_content_stripped = line_content.rstrip()

        if not line_content_stripped.strip() or line_content_stripped.strip() == '│':
            continue

        match = re.match(r"^([│\s]*)(└─\s*|├─\s*)(.*)", line_content_stripped)
        if not match:
            print(f"警告 (处理子项列表中的第 {line_number_in_children_list + 1} 行): 无法解析行结构: '{line_content_stripped}'")
            continue

        # indent_prefix = match.group(1) # 例如 "│   "
        connector = match.group(2)      # 例如 "├─ "
        name_and_comment = match.group(3)

        item_name_full = name_and_comment.split('#')[0].strip()
        if not item_name_full:
            # 此处可以添加警告，但如果行完全是注释或空白，上面已跳过
            continue

        is_directory = item_name_full.endswith('/')
        item_name = item_name_full.rstrip('/')

        column_of_connector = line_content_stripped.find(connector)
        current_item_level = column_of_connector // INDENT_STEP

        parent_directory_path = current_paths_at_level.get(current_item_level - 1)
        if parent_directory_path is None:
            print(f"错误 (处理子项列表中的第 {line_number_in_children_list + 1} 行 '{item_name}'): 找不到层级 {current_item_level - 1} 的父目录路径。当前已知路径: {current_paths_at_level}")
            continue
        
        current_item_path = os.path.join(parent_directory_path, item_name)

        try:
            if is_directory:
                os.makedirs(current_item_path, exist_ok=True)
                print(f"已创建目录: {current_item_path}")
                current_paths_at_level[current_item_level] = current_item_path
                keys_to_delete = [lvl for lvl in current_paths_at_level if lvl > current_item_level]
                for lvl_del in keys_to_delete:
                    del current_paths_at_level[lvl_del]
            else:
                os.makedirs(os.path.dirname(current_item_path), exist_ok=True)
                with open(current_item_path, 'w', encoding='utf-8') as f:
                    pass 
                print(f"已创建文件: {current_item_path}")
        except OSError as e:
            print(f"错误: 创建 '{current_item_path}' 时发生错误: {e}")
            
    print("\n项目结构创建完成。")

# --- 如何使用该脚本 ---
project_tree_text = """
AI-Tetris/
├─ weights/                  # 存放训练好的模型权重
│   └── (例如 dqn_tetris_episode_1000.pth)
├─ screenshots/              # 存放训练过程中评估模式的游戏截图
│   └── (例如 eval_episode_1000_final.png)
├─ logs/                     # 存放训练日志
│   └── (例如 training_log_YYYYMMDD_HHMMSS.csv)
├─ config.yaml               # 配置文件 (游戏参数, AI参数, 训练参数)
├─ main.py                   # 主程序入口，模式选择
├─ train.py                  # AI训练脚本
├─ tetris_game.py            # Tetris游戏核心逻辑模块
├─ dqn_agent.py              # DQN Agent及神经网络模块
├─ tetrominoes.py            # 定义方块形状、颜色及Piece类
├─ srs_data.py               # SRS旋转的踢墙数据
├─ operation_module.py       # (概念上)操作模块，实际逻辑可能整合在tetris_game.py
├─ README.md                 
└─ requirements.txt          # (建议添加)项目依赖
"""

if __name__ == "__main__":
    # 场景1: 创建 'tetris-ai/' 顶层目录 (原始行为)
    # print("--- 场景 1: 创建顶层 'tetris-ai' 目录 ---")
    # create_project_from_text(project_tree_text, 
    #                          base_output_path="output_scenario1", 
    #                          create_container_folder_for_first_line=True)
    # print("\n")

    # 场景2: 不创建 'tetris-ai/' 顶层目录，直接在 base_output_path ("output_scenario2") 下创建内容
    print("--- 场景 2: 不创建顶层 'tetris-ai' 目录 ---")
    # 确保 output_scenario2 目录存在，如果不想在当前目录下直接铺开的话
    # os.makedirs("output_scenario2", exist_ok=True) # 如果 base_output_path 是一个新建的子目录
    
    # 如果你希望直接在当前目录下创建 tetris/, agent/ 等，设置 base_output_path="."
    create_project_from_text(project_tree_text, 
                             base_output_path=".",  # 例如，直接在当前运行脚本的目录下创建
                             create_container_folder_for_first_line=False)
import pygame
import random
import math
import time

# --- 配置 ---
# 颜色定义
BACKGROUND_COLOR = (20, 20, 30)
NODE_COLOR = (100, 180, 255)
NODE_BORDER_COLOR = (200, 220, 255)
ACTIVE_NODE_COLOR = (255, 255, 0) # 激活时节点的颜色
CONNECTION_COLOR = (60, 90, 120)
ACTIVE_CONNECTION_COLOR = (255, 165, 0) # 激活时连接的颜色
TEXT_COLOR = (220, 220, 220)
INPUT_TEXT_COLOR = (150, 255, 150)
OUTPUT_TEXT_COLOR = (255, 150, 150)

# 屏幕和网络布局
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700
NODE_RADIUS = 20
LAYER_SPACING = 250 # 层与层之间的水平距离

# 神经网络结构
INPUT_NODES = 7
HIDDEN_LAYER_1_NODES = 8
HIDDEN_LAYER_2_NODES = 4
OUTPUT_NODES = 1
layers = [INPUT_NODES, HIDDEN_LAYER_1_NODES, HIDDEN_LAYER_2_NODES, OUTPUT_NODES]

# 输入特征标签 (来自我们的项目)
input_labels = [
    "Height", "Holes", "Gen Wells",
    "Bumpiness", "Clear Lines", "Combo",
    "Well Occupy"
]


def calculate_layer_positions(screen_width, screen_height, num_layers, nodes_in_layers):
    """计算网络中所有节点的位置"""
    positions = []
    total_width = (num_layers - 1) * LAYER_SPACING
    start_x = (screen_width - total_width) / 2
    
    for i, num_nodes in enumerate(nodes_in_layers):
        layer_x = start_x + i * LAYER_SPACING
        layer_positions = []
        
        total_height = (num_nodes - 1) * (NODE_RADIUS * 3)
        start_y = (screen_height - total_height) / 2
        
        for j in range(num_nodes):
            node_y = start_y + j * (NODE_RADIUS * 3)
            layer_positions.append((layer_x, node_y))
        positions.append(layer_positions)
    return positions


def draw_network(screen, font, small_font, node_positions, active_nodes=None, active_connections=None):
    """绘制整个神经网络"""
    if active_nodes is None:
        active_nodes = set()
    if active_connections is None:
        active_connections = set()

    # 绘制连接线
    for i in range(len(node_positions) - 1):
        for j, start_pos in enumerate(node_positions[i]):
            for k, end_pos in enumerate(node_positions[i+1]):
                is_active = (i, j, k) in active_connections
                color = ACTIVE_CONNECTION_COLOR if is_active else CONNECTION_COLOR
                pygame.draw.line(screen, color, start_pos, end_pos, 2 if is_active else 1)
    
    # 绘制节点
    for i, layer in enumerate(node_positions):
        for j, pos in enumerate(layer):
            is_active = (i, j) in active_nodes
            node_c = ACTIVE_NODE_COLOR if is_active else NODE_COLOR
            border_c = NODE_BORDER_COLOR
            
            pygame.draw.circle(screen, node_c, pos, NODE_RADIUS)
            pygame.draw.circle(screen, border_c, pos, NODE_RADIUS, 2)
            
            # 绘制标签
            if i == 0: # 输入层
                label_text = input_labels[j]
                text_surf = small_font.render(label_text, True, INPUT_TEXT_COLOR)
                screen.blit(text_surf, (pos[0] - text_surf.get_width() - 25, pos[1] - text_surf.get_height() / 2))
            elif i == len(node_positions) - 1: # 输出层
                 label_text = "Q-Value"
                 text_surf = small_font.render(label_text, True, OUTPUT_TEXT_COLOR)
                 screen.blit(text_surf, (pos[0] + 30, pos[1] - text_surf.get_height() / 2))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Neural Network - Forward Pass Visualization")
    font = pygame.font.SysFont('arial', 24)
    small_font = pygame.font.SysFont('arial', 16)
    clock = pygame.time.Clock()

    node_positions = calculate_layer_positions(SCREEN_WIDTH, SCREEN_HEIGHT, len(layers), layers)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # --- 开始动画 ---
                    active_nodes = set()
                    active_connections = set()
                    
                    # 激活输入层
                    for j in range(layers[0]):
                        active_nodes.add((0, j))
                    
                    # 逐层激活
                    for i in range(len(layers) - 1):
                        screen.fill(BACKGROUND_COLOR)
                        draw_network(screen, font, small_font, node_positions, active_nodes, active_connections)
                        title_surf = font.render("Forward Pass: Calculating Q-Value...", True, TEXT_COLOR)
                        screen.blit(title_surf, (20, 20))
                        pygame.display.flip()
                        time.sleep(0.5)

                        # 激活连接
                        for j in range(layers[i]):
                            for k in range(layers[i+1]):
                                active_connections.add((i, j, k))
                        
                        screen.fill(BACKGROUND_COLOR)
                        draw_network(screen, font, small_font, node_positions, active_nodes, active_connections)
                        title_surf = font.render("Forward Pass: Calculating Q-Value...", True, TEXT_COLOR)
                        screen.blit(title_surf, (20, 20))
                        pygame.display.flip()
                        time.sleep(0.5)
                        
                        # 激活下一层节点
                        for k in range(layers[i+1]):
                            active_nodes.add((i+1, k))
                    
                    # 显示最终结果
                    screen.fill(BACKGROUND_COLOR)
                    draw_network(screen, font, small_font, node_positions, active_nodes, active_connections)
                    final_q_val = random.uniform(50, 500) # 模拟一个Q值
                    q_text = f"Predicted Q-Value: {final_q_val:.2f}"
                    q_surf = font.render(q_text, True, ACTIVE_NODE_COLOR)
                    q_pos = node_positions[-1][0]
                    screen.blit(q_surf, (q_pos[0] - q_surf.get_width()/2, q_pos[1] + 40))
                    
                    info_surf = font.render("Done! Press SPACE to run again.", True, TEXT_COLOR)
                    screen.blit(info_surf, (20, 650))
                    pygame.display.flip()

        screen.fill(BACKGROUND_COLOR)
        draw_network(screen, font, small_font, node_positions)
        
        # 初始提示
        title_surf = font.render("Forward Pass Visualization", True, TEXT_COLOR)
        screen.blit(title_surf, (SCREEN_WIDTH/2 - title_surf.get_width()/2, 20))
        prompt_surf = font.render("Press SPACE to start the animation", True, TEXT_COLOR)
        screen.blit(prompt_surf, (SCREEN_WIDTH/2 - prompt_surf.get_width()/2, 60))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()
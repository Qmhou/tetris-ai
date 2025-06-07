import pygame
import random
import math
import time

# --- 配置 ---
# 颜色定义
BACKGROUND_COLOR = (30, 20, 20) # 略带红色的背景
NODE_COLOR = (100, 180, 255)
NODE_BORDER_COLOR = (200, 220, 255)
ACTIVE_NODE_COLOR = (255, 255, 0)
CONNECTION_COLOR = (60, 90, 120)
ERROR_CONNECTION_COLOR = (255, 80, 80) # 误差反向传播的颜色
UPDATED_CONNECTION_COLOR = (150, 255, 150) # 权重更新后的颜色
TEXT_COLOR = (220, 220, 220)
LOSS_TEXT_COLOR = (255, 100, 100)

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

def draw_network(screen, font, node_positions, error_connections=None, updated_connections=None):
    """绘制整个神经网络"""
    if error_connections is None:
        error_connections = set()
    if updated_connections is None:
        updated_connections = set()
        
    # 绘制连接线
    for i in range(len(node_positions) - 1):
        for j, start_pos in enumerate(node_positions[i]):
            for k, end_pos in enumerate(node_positions[i+1]):
                conn_id = (i, j, k)
                color = CONNECTION_COLOR
                width = 1
                if conn_id in error_connections:
                    color = ERROR_CONNECTION_COLOR
                    width = 3
                elif conn_id in updated_connections:
                    color = UPDATED_CONNECTION_COLOR
                    width = 2
                pygame.draw.line(screen, color, start_pos, end_pos, width)
    
    # 绘制节点
    for i, layer in enumerate(node_positions):
        for j, pos in enumerate(layer):
            pygame.draw.circle(screen, NODE_COLOR, pos, NODE_RADIUS)
            pygame.draw.circle(screen, NODE_BORDER_COLOR, pos, NODE_RADIUS, 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Neural Network - Backward Pass Visualization")
    font = pygame.font.SysFont('arial', 24)
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
                    predicted_q = random.uniform(100, 200)
                    target_q = predicted_q + random.uniform(50, 150) # 目标值比预测值高
                    loss = (target_q - predicted_q) ** 2

                    error_connections = set()
                    updated_connections = set()
                    
                    # 从后往前逐层传播
                    for i in range(len(layers) - 2, -1, -1):
                        # 绘制当前状态，显示Loss
                        screen.fill(BACKGROUND_COLOR)
                        draw_network(screen, font, node_positions, error_connections, updated_connections)
                        title_surf = font.render(f"Backward Pass: Propagating Error (Loss)...", True, TEXT_COLOR)
                        screen.blit(title_surf, (20, 20))
                        
                        pred_surf = font.render(f"Predicted Q: {predicted_q:.2f}", True, TEXT_COLOR)
                        screen.blit(pred_surf, (SCREEN_WIDTH - 250, 40))
                        target_surf = font.render(f"Target Q: {target_q:.2f}", True, TEXT_COLOR)
                        screen.blit(target_surf, (SCREEN_WIDTH - 250, 70))
                        loss_surf = font.render(f"Loss: {loss:.2f}", True, LOSS_TEXT_COLOR)
                        screen.blit(loss_surf, (SCREEN_WIDTH - 250, 100))
                        
                        pygame.display.flip()
                        time.sleep(0.7)

                        # 激活当前层的误差连接
                        for j in range(layers[i]):
                            for k in range(layers[i+1]):
                                error_connections.add((i, j, k))
                        
                        # 重绘以显示误差传播
                        screen.fill(BACKGROUND_COLOR)
                        draw_network(screen, font, node_positions, error_connections, updated_connections)
                        title_surf = font.render(f"Backward Pass: Calculating Gradients...", True, TEXT_COLOR)
                        screen.blit(title_surf, (20, 20))
                        # ... (再次绘制右侧的Q值和Loss信息)
                        pred_surf = font.render(f"Predicted Q: {predicted_q:.2f}", True, TEXT_COLOR)
                        screen.blit(pred_surf, (SCREEN_WIDTH - 250, 40))
                        target_surf = font.render(f"Target Q: {target_q:.2f}", True, TEXT_COLOR)
                        screen.blit(target_surf, (SCREEN_WIDTH - 250, 70))
                        loss_surf = font.render(f"Loss: {loss:.2f}", True, LOSS_TEXT_COLOR)
                        screen.blit(loss_surf, (SCREEN_WIDTH - 250, 100))
                        pygame.display.flip()
                        time.sleep(0.7)

                        # 标记为已更新
                        for conn_id in error_connections:
                            updated_connections.add(conn_id)
                        error_connections.clear()

                    # 显示最终结果
                    screen.fill(BACKGROUND_COLOR)
                    draw_network(screen, font, node_positions, error_connections, updated_connections)
                    final_text = "All weights have been updated! The AI has learned from its mistake."
                    final_surf = font.render(final_text, True, UPDATED_CONNECTION_COLOR)
                    screen.blit(final_surf, (SCREEN_WIDTH/2 - final_surf.get_width()/2, SCREEN_HEIGHT - 50))
                    info_surf = font.render("Press SPACE to run again.", True, TEXT_COLOR)
                    screen.blit(info_surf, (20, 650))
                    pygame.display.flip()

        screen.fill(BACKGROUND_COLOR)
        draw_network(screen, font, node_positions)
        
        # 初始提示
        title_surf = font.render("Backward Pass Visualization", True, TEXT_COLOR)
        screen.blit(title_surf, (SCREEN_WIDTH/2 - title_surf.get_width()/2, 20))
        prompt_surf = font.render("Press SPACE to start the animation", True, TEXT_COLOR)
        screen.blit(prompt_surf, (SCREEN_WIDTH/2 - prompt_surf.get_width()/2, 60))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()
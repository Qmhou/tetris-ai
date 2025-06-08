import pygame
import random
import numpy as np

# --- 配置 ---
# 颜色定义
BACKGROUND_COLOR = (20, 20, 30)
NODE_COLOR = (100, 180, 255)
NODE_BORDER_COLOR = (200, 220, 255)
CURRENT_NODE_COLOR = (255, 255, 0) # 当前计算的节点
INCOMING_CONNECTION_COLOR = (255, 165, 0) # 指向当前节点的连接
TEXT_COLOR = (220, 220, 220)
VALUE_TEXT_COLOR = (200, 255, 200)
FORMULA_TEXT_COLOR = (255, 200, 150)

# 屏幕和网络布局
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 750
NODE_RADIUS = 22
LAYER_SPACING = 300

# 神经网络结构
INPUT_NODES = 7
HIDDEN_LAYER_1_NODES = 8
HIDDEN_LAYER_2_NODES = 4
OUTPUT_NODES = 1
layers_config = [INPUT_NODES, HIDDEN_LAYER_1_NODES, HIDDEN_LAYER_2_NODES, OUTPUT_NODES]

# 输入特征标签
input_labels = ["Height", "Holes", "Gen Wells", "Bumpiness", "Clear Lines", "Combo", "Well Occupy"]

class NeuralNetwork:
    def __init__(self, layer_config):
        self.layer_config = layer_config
        # 使用Numpy随机初始化权重和偏置
        self.weights = [np.random.randn(y, x) * 0.1 for x, y in zip(layer_config[:-1], layer_config[1:])]
        self.biases = [np.random.randn(y, 1) * 0.1 for y in layer_config[1:]]
        self.node_values = [np.zeros((n, 1)) for n in layer_config]

    def relu(self, x):
        return np.maximum(0, x)

    def forward_step(self, layer_index, node_index):
        """计算指定层中单个节点的值"""
        if layer_index == 0:
            return # 输入层的值是预设的
        
        # z = (weights * inputs) + bias
        z = np.dot(self.weights[layer_index-1][node_index], self.node_values[layer_index-1]) + self.biases[layer_index-1][node_index]
        
        # 应用激活函数 (输出层除外)
        if layer_index < len(self.layer_config) - 1:
            activation = self.relu(z)
        else:
            activation = z # 输出层是线性激活
            
        self.node_values[layer_index][node_index] = activation
        return float(z), float(activation)

def get_node_positions(screen_width, screen_height, layer_config):
    """计算所有节点的位置"""
    positions = []
    total_width = (len(layer_config) - 1) * LAYER_SPACING
    start_x = (screen_width - total_width) / 2
    for i, num_nodes in enumerate(layer_config):
        layer_x = start_x + i * LAYER_SPACING
        layer_positions = []
        total_height = (num_nodes - 1) * (NODE_RADIUS * 3.5)
        start_y = (screen_height - total_height) / 2
        for j in range(num_nodes):
            node_y = start_y + j * (NODE_RADIUS * 3.5)
            layer_positions.append((layer_x, node_y))
        positions.append(layer_positions)
    return positions

def draw_network(screen, fonts, net, positions, current_node=None, formula_text=""):
    """绘制神经网络及其状态"""
    # 绘制连接线
    for i in range(len(positions) - 1):
        for j, start_pos in enumerate(positions[i]):
            for k, end_pos in enumerate(positions[i+1]):
                is_incoming = current_node and current_node == (i + 1, k)
                color = INCOMING_CONNECTION_COLOR if is_incoming else (60, 90, 120)
                width = 3 if is_incoming else 1
                pygame.draw.line(screen, color, start_pos, end_pos, width)

    # 绘制节点和值
    for i, layer_pos in enumerate(positions):
        for j, pos in enumerate(layer_pos):
            is_current = current_node and current_node == (i, j)
            color = CURRENT_NODE_COLOR if is_current else NODE_COLOR
            pygame.draw.circle(screen, color, pos, NODE_RADIUS)
            pygame.draw.circle(screen, NODE_BORDER_COLOR, pos, NODE_RADIUS, 2)

            # 显示节点值
            value = net.node_values[i][j][0]
            if abs(value) > 0.001 or i == 0: # 只有非零值和输入值才显示
                val_text = f"{value:.2f}"
                val_surf = fonts['small'].render(val_text, True, VALUE_TEXT_COLOR)
                screen.blit(val_surf, (pos[0] - val_surf.get_width()/2, pos[1] - val_surf.get_height()/2))
            
            # 绘制标签
            if i == 0:
                label_surf = fonts['small'].render(input_labels[j], True, TEXT_COLOR)
                screen.blit(label_surf, (pos[0] - label_surf.get_width() - 30, pos[1] - label_surf.get_height()/2))
            elif i == len(positions) - 1:
                label_surf = fonts['small'].render("Q-Value", True, TEXT_COLOR)
                screen.blit(label_surf, (pos[0] + 35, pos[1] - label_surf.get_height()/2))
    
    # 显示计算公式
    if formula_text:
        formula_surf = fonts['formula'].render(formula_text, True, FORMULA_TEXT_COLOR)
        screen.blit(formula_surf, (20, SCREEN_HEIGHT - 50))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Forward Pass - Step-by-Step")
    fonts = {
        'default': pygame.font.SysFont('consolas', 22),
        'small': pygame.font.SysFont('consolas', 14),
        'formula': pygame.font.SysFont('consolas', 18, bold=True)
    }
    
    net = NeuralNetwork(layers_config)
    node_positions = get_node_positions(SCREEN_WIDTH, SCREEN_HEIGHT, layers_config)

    # --- 状态变量 ---
    # 0:待开始, 1:输入已加载, 2...n:计算步骤, -1:已完成
    animation_step = 0
    current_layer_idx, current_node_idx = 1, 0
    formula_text = ""

    def reset_animation():
        """重置动画状态和网络值"""
        nonlocal animation_step, current_layer_idx, current_node_idx, formula_text, net
        animation_step = 0
        current_layer_idx, current_node_idx = 1, 0
        formula_text = ""
        # 重新初始化网络，得到新的随机权重
        net = NeuralNetwork(layers_config)
    
    reset_animation()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if animation_step == -1: # 如果已完成，则重置
                        reset_animation()
                        
                    animation_step += 1
                    
                    if animation_step == 1:
                        # 加载示例输入数据
                        # [Height, Holes, Gen Wells, Bumpiness, Clear, Combo, Well Occupy]
                        sample_input = np.array([[5.0, 2.0, 10.0, 4.0, 1.0, 2.0, 0.0]]).T / 20.0 # 简单缩放
                        net.node_values[0] = sample_input
                        formula_text = "Step 1: Input features loaded. Press SPACE to calculate first hidden node."
                    elif animation_step > 1:
                        # 计算当前节点
                        z_val, act_val = net.forward_step(current_layer_idx, current_node_idx)
                        
                        # 准备公式文本
                        activation_func_name = "ReLU" if current_layer_idx < len(layers_config) - 1 else "Linear"
                        formula_text = f"L{current_layer_idx}N{current_node_idx}: Sum(weights*inputs)+bias = {z_val:.2f} -> {activation_func_name} -> Activation = {act_val:.2f}"
                        
                        # 移动到下一个节点
                        current_node_idx += 1
                        if current_node_idx >= layers_config[current_layer_idx]:
                            current_node_idx = 0
                            current_layer_idx += 1
                            if current_layer_idx >= len(layers_config):
                                animation_step = -1 # 动画完成
                                formula_text = f"Forward Pass Complete! Final Q-Value: {act_val:.2f}. Press SPACE to restart."
                
                if event.key == pygame.K_r:
                    reset_animation()

        screen.fill(BACKGROUND_COLOR)
        
        # 绘制
        current_node_to_draw = (current_layer_idx, current_node_idx) if 1 < animation_step != -1 else None
        draw_network(screen, fonts, net, node_positions, current_node_to_draw, formula_text)
        
        # 绘制提示
        if animation_step == 0:
            prompt_surf = fonts['default'].render("Press SPACE to load input and start.", True, TEXT_COLOR)
            screen.blit(prompt_surf, (20, 20))
        
        prompt_surf = fonts['default'].render("Press [SPACE] to advance, [R] to reset.", True, TEXT_COLOR)
        screen.blit(prompt_surf, (SCREEN_WIDTH - prompt_surf.get_width() - 20, 20))
            
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
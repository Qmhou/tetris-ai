import pygame
import random
import numpy as np

# --- 配置 ---
BACKGROUND_COLOR = (30, 20, 20)
NODE_COLOR = (100, 180, 255)
NODE_BORDER_COLOR = (200, 220, 255)
ERROR_CONNECTION_COLOR = (255, 80, 80)
ERROR_NODE_COLOR = (255, 120, 120) # 误差传播时节点的颜色
UPDATED_CONNECTION_COLOR = (150, 255, 150)
TEXT_COLOR = (220, 220, 220)
FORMULA_TEXT_COLOR = (255, 200, 150)
LOSS_TEXT_COLOR = (255, 100, 100)
ERROR_TEXT_COLOR = (255, 170, 170)
LEARNING_RATE = 0.1

SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
NODE_RADIUS = 22
LAYER_SPACING = 300

layers_config = [7, 8, 4, 1]

class NeuralNetwork:
    def __init__(self, layer_config):
        self.layer_config = layer_config
        self.weights = [np.random.randn(y, x) * 0.5 for x, y in zip(layer_config[:-1], layer_config[1:])]
        self.node_values = [np.random.rand(n, 1) * 0.5 for n in layer_config]
        self.node_errors = [np.zeros((n, 1)) for n in layer_config]

    def reset_for_new_pass(self):
        """仅重置误差，保留权重用于多次演示"""
        self.node_errors = [np.zeros((n, 1)) for n in self.layer_config]

    def accumulate_error_contribution(self, layer_idx, from_node_idx, to_node_idx):
        """(演示)从下一层单个节点(to_node)累加误差到前一层节点(from_node)"""
        # 下一层节点的误差
        next_layer_error = self.node_errors[layer_idx + 1][to_node_idx]
        # 连接的权重
        weight_val = self.weights[layer_idx][to_node_idx, from_node_idx]
        
        # 贡献的误差 = 下一层误差 * 权重
        error_contribution = next_layer_error * weight_val
        self.node_errors[layer_idx][from_node_idx] += error_contribution
        
        return float(error_contribution), float(self.node_errors[layer_idx][from_node_idx])

    def update_weight(self, layer_idx, from_node_idx, to_node_idx):
        """(演示)利用前向激活值和后向误差更新单个权重"""
        forward_activation = self.node_values[layer_idx][from_node_idx]
        backward_error = self.node_errors[layer_idx + 1][to_node_idx]
        
        # 简化梯度计算: 梯度 ≈ 后层误差 * 前层激活值
        gradient = backward_error * forward_activation
        
        old_weight = self.weights[layer_idx][to_node_idx, from_node_idx]
        change = LEARNING_RATE * gradient
        new_weight = old_weight - change
        self.weights[layer_idx][to_node_idx, from_node_idx] = new_weight
        
        return float(gradient), float(old_weight), float(new_weight)

def get_node_positions(screen_width, screen_height, layer_config):
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

def draw_network(screen, fonts, net, positions, updated_connections, highlight_info=None, formula_text=""):
    # ... (与之前版本类似，但高亮逻辑更复杂)
    h_conn = highlight_info.get('conn') if highlight_info else None
    h_node = highlight_info.get('node') if highlight_info else None

    # 绘制连接线
    for i in range(len(positions) - 1):
        for j in range(len(positions[i])):
            for k in range(len(positions[i+1])):
                color, width = (60, 90, 120), 1
                if (i, j, k) in updated_connections: color, width = UPDATED_CONNECTION_COLOR, 3
                if h_conn and h_conn == (i, j, k): color, width = ERROR_CONNECTION_COLOR, 4
                pygame.draw.line(screen, color, positions[i][j], positions[i+1][k], width)

    # 绘制节点
    for i, layer_pos in enumerate(positions):
        for j, pos in enumerate(layer_pos):
            color = NODE_COLOR
            if h_node and h_node == (i,j): color = ERROR_NODE_COLOR
            pygame.draw.circle(screen, color, pos, NODE_RADIUS)
            pygame.draw.circle(screen, NODE_BORDER_COLOR, pos, NODE_RADIUS, 2)
            
            error_value = net.node_errors[i][j][0]
            if abs(error_value) > 1e-6:
                error_text = f"e={error_value:.2f}"
                error_surf = fonts['small'].render(error_text, True, ERROR_TEXT_COLOR)
                screen.blit(error_surf, (pos[0] - error_surf.get_width()/2, pos[1] - error_surf.get_height()/2))
    
    if formula_text:
        formula_surf = fonts['formula'].render(formula_text, True, FORMULA_TEXT_COLOR)
        screen.blit(formula_surf, (20, SCREEN_HEIGHT - 70))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Backward Pass v3 - Accumulation Demo")
    fonts = {
        'default': pygame.font.SysFont('consolas', 22),
        'formula': pygame.font.SysFont('consolas', 16, bold=True),
        'small': pygame.font.SysFont('consolas', 14, bold=True)
    }

    # --- 状态变量 ---
    animation_step = 0 # 0:待开始, 1:Loss已计算, 2:开始循环
    mode = "PROPAGATE" # 'PROPAGATE' or 'UPDATE'
    l_idx = len(layers_config) - 2
    from_n_idx, to_n_idx = 0, 0
    
    formula_text, predicted_q, target_q, loss = "", 0, 0, 0
    updated_connections, highlight_info = set(), {}
    net = NeuralNetwork(layers_config)

    def reset_animation():
        nonlocal animation_step, mode, l_idx, from_n_idx, to_n_idx, formula_text
        nonlocal predicted_q, target_q, loss, updated_connections, highlight_info, net
        animation_step = 0
        mode = "PROPAGATE"
        l_idx = len(layers_config) - 2
        from_n_idx, to_n_idx = 0, 0
        formula_text = ""
        updated_connections, highlight_info = set(), {}
        net = NeuralNetwork(layers_config)
        predicted_q = net.node_values[-1][0][0] * 500
        target_q = predicted_q + random.uniform(80, 150)
        loss = 0.5 * (target_q - predicted_q)**2

    reset_animation()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: reset_animation()
                if event.key == pygame.K_SPACE:
                    if animation_step == -1: reset_animation(); continue
                    
                    if animation_step == 0:
                        animation_step = 1
                        error = predicted_q - target_q
                        net.node_errors[-1][0] = error
                        formula_text = f"Step 1: Calculate output error (Pred-Targ) = {error:.2f}"
                        highlight_info = {'node': (len(layers_config)-1, 0)}
                    
                    elif animation_step >= 1:
                        if mode == "PROPAGATE":
                            # 累加误差
                            contrib, total_err = net.accumulate_error_contribution(l_idx, from_n_idx, to_n_idx)
                            formula_text = f"Propagate from L{l_idx+1}N{to_n_idx} to L{l_idx}N{from_n_idx}: TotalError({total_err:.2f}) += Contrib({contrib:.2f})"
                            highlight_info = {'conn': (l_idx, from_n_idx, to_n_idx), 'node': (l_idx, from_n_idx)}
                            mode = "UPDATE"
                        
                        elif mode == "UPDATE":
                            # 更新权重
                            grad, old_w, new_w = net.update_weight(l_idx, from_n_idx, to_n_idx)
                            formula_text = f"Update W L{l_idx}({from_n_idx}->{to_n_idx}): New({new_w:.3f}) = Old({old_w:.3f}) - LR*Grad({grad:.3f})"
                            updated_connections.add((l_idx, from_n_idx, to_n_idx))
                            highlight_info = {'conn': (l_idx, from_n_idx, to_n_idx)}

                            # 移到下一个连接进行传播
                            to_n_idx += 1
                            if to_n_idx >= layers_config[l_idx+1]:
                                to_n_idx = 0
                                # 当一个节点的所有下游误差都传播回来后，才移动到下一个节点
                                from_n_idx += 1
                                if from_n_idx >= layers_config[l_idx]:
                                    from_n_idx = 0
                                    # 当一层所有节点误差都计算完后，才移动到前一层
                                    l_idx -= 1
                                    if l_idx < 0:
                                        animation_step = -1
                                        formula_text = "Backward Pass Complete! Press [R] to restart."
                                        highlight_info = {}
                            mode = "PROPAGATE"
        
        screen.fill(BACKGROUND_COLOR)
        node_positions = get_node_positions(SCREEN_WIDTH, SCREEN_HEIGHT, layers_config)
        draw_network(screen, fonts, net, node_positions, updated_connections, highlight_info, formula_text)
        
        # 绘制信息面板
        info_y = 60
        pred_surf = fonts['default'].render(f"Predicted Q: {predicted_q:.2f}", True, TEXT_COLOR)
        screen.blit(pred_surf, (SCREEN_WIDTH - 280, info_y))
        target_surf = fonts['default'].render(f"Target Q: {target_q:.2f}", True, TEXT_COLOR)
        screen.blit(target_surf, (SCREEN_WIDTH - 280, info_y + 30))
        loss_surf = fonts['default'].render(f"Loss: {loss:.2f}", True, LOSS_TEXT_COLOR)
        screen.blit(loss_surf, (SCREEN_WIDTH - 280, info_y + 60))
        
        if animation_step == 0:
            prompt_surf = fonts['default'].render("Press [SPACE] to start.", True, TEXT_COLOR)
            screen.blit(prompt_surf, (20, 20))
        
        prompt_surf = fonts['default'].render("Press [SPACE] to advance, [R] to reset.", True, TEXT_COLOR)
        screen.blit(prompt_surf, (SCREEN_WIDTH - prompt_surf.get_width() - 20, 20))
        
        pygame.display.flip()
        
    pygame.quit()

if __name__ == '__main__':
    main()
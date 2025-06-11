# record_manual_play.py
# A standalone, turn-based tool for recording expert demonstrations for RL training.

import pygame
import yaml
import os
import pickle
import argparse
import copy
from collections import defaultdict

# We import the classes from our existing project files without modifying them.
from tetris_game import TetrisGame
from dqn_agent import DQNAgent

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def record_expert_play(config, agent, demonstration_name):
    """
    Initializes and runs the main loop for the annotation tool.
    """
    game = TetrisGame(config, render_mode=True)
    game.reset()

    # This list will hold all experiences for the CURRENT demonstration session
    recorded_experiences = []
    session_counter = 0

    # This list acts as an undo stack for the CURRENT piece's moves
    current_piece_move_history = []

    output_dir = "expert_data"
    os.makedirs(output_dir, exist_ok=True)

    # --- Print instructions for the user ---
    print("\n--- Expert Experience Annotation Tool ---")
    print(f"Current Demonstration Type: '{demonstration_name}'")
    print("This is a TURN-BASED tool. Pieces will NOT fall automatically.")
    print("\nControls:")
    print("  - Arrows, Z, X, C:  Move/Rotate/Hold piece")
    print("  - Spacebar:         LOCK the piece and RECORD the experience")
    print("  - Backspace:        UNDO the last move of the CURRENT piece")
    print("  - S:                SAVE the current demonstration and start a new one")
    print("  - R:                RESET/discard the current demonstration attempt")
    print("  - ESC/Close Window: Exit the program")
    print("-" * 20)

    running = True

    # --- Main Loop ---
    while running:
        game.render() # Update the display

        # Wait for a user action, making the tool turn-based
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue

                # --- Session Management Controls ---
                if event.key == pygame.K_s: # Save and Reset
                    if recorded_experiences:
                        filename = f"{demonstration_name}_{session_counter}.pkl"
                        output_path = os.path.join(output_dir, filename)
                        print(f"\nSaving {len(recorded_experiences)} experiences to {output_path}...")
                        with open(output_path, 'wb') as f:
                            pickle.dump(recorded_experiences, f)
                        print("Save complete!")
                        session_counter += 1
                    else:
                        print("\nCurrent demonstration is empty. Nothing to save.")

                    recorded_experiences.clear()
                    current_piece_move_history.clear()
                    game.reset()
                    print(f"--- Starting new demonstration: {demonstration_name}_{session_counter} ---")
                    continue

                if event.key == pygame.K_r: # Reset/Discard
                    print("\nDiscarding current attempt and resetting...")
                    recorded_experiences.clear()
                    current_piece_move_history.clear()
                    game.reset()
                    print(f"--- Restarted demonstration: {demonstration_name}_{session_counter} ---")
                    continue

                # --- Undo Control ---
                if event.key == pygame.K_BACKSPACE:
                    if current_piece_move_history:
                        last_state = current_piece_move_history.pop()
                        game.current_piece.x, game.current_piece.y, game.current_piece.rotation, game.current_piece.shape_coords = last_state
                        print("Undo last move.")
                    else:
                        print("No more moves to undo for this piece.")
                    continue

                if game.game_over: continue

                # --- Piece Movement Controls ---
                action_map = {
                    pygame.K_LEFT: 'left', pygame.K_RIGHT: 'right', pygame.K_DOWN: 'soft_drop',
                    pygame.K_UP: 'rotate_cw', pygame.K_x: 'rotate_cw',
                    pygame.K_z: 'rotate_ccw', pygame.K_c: 'hold'
                }
                action = action_map.get(event.key)
                if action:
                    # Save the piece's state BEFORE the move for the undo stack
                    current_piece_move_history.append(
                        (game.current_piece.x, game.current_piece.y, game.current_piece.rotation, game.current_piece.shape_coords)
                    )
                    game.execute_atomic_action(action)
                    if action == 'hold': # A new piece is now active
                        current_piece_move_history.clear()

                # --- Lock and Record Control ---
                if event.key == pygame.K_SPACE:
                    print("\nLocking piece and recording experience...")

                    # 1. Capture info BEFORE locking the piece
                    action_info = (game.current_piece.type, game.current_piece.rotation, game.current_piece.x, game.current_piece.y)
                    temp_grid = game._place_piece_on_grid(game.current_piece, game.grid)

                    # 2. Generate the state tensor (s') and auxiliary labels for this outcome
                    height, holes, _, _ = game._calculate_grid_metrics(temp_grid)
                    _, lines_cleared_count = game._simulate_line_clear(temp_grid)
                    aux_labels = {'lines': lines_cleared_count, 'holes': holes, 'height': height}
                    final_coords = [(game.current_piece.y + ro, game.current_piece.x + co) for ro, co in game.current_piece.shape_coords]
                    s_prime_tensor = game._convert_grid_to_cnn_input(temp_grid, final_coords)

                    # 3. Lock the piece in the actual game to get the reward and advance the state
                    reward_info_dict, done_after_action, _ = game._lock_piece()
                    total_reward_this_step = reward_info_dict.get("final_reward", 0.0)

                    # 4. Calculate next_best_q_value from the NEW game state (with the next piece)
                    next_best_q_value = 0.0
                    if not done_after_action:
                        next_moves = game.get_all_possible_next_states_and_features()
                        if next_moves:
                            _ , next_best_q_value = agent.get_best_action_and_value(next_moves)

                    # 5. Assemble and store the complete experience tuple
                    experience = (s_prime_tensor, total_reward_this_step, next_best_q_value, done_after_action, aux_labels)
                    recorded_experiences.append(experience)
                    print(f"Success! Experience #{len(recorded_experiences)} for this session recorded.")

                    # 6. A new piece has spawned, clear the undo history for it
                    current_piece_move_history.clear()

    # --- Cleanup ---
    if pygame.get_init():
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tetris Expert Data Annotation Tool")
    parser.add_argument('--model_path', type=str, required=True, help="Path to a pre-trained agent model to calculate Q-values.")
    parser.add_argument('--name', type=str, default="expert_demo", help="Base name for the demonstration type (e.g., 'dt_cannon').")
    args = parser.parse_args()

    print("Loading config and agent...")
    game_config = load_config()
    ai_agent = DQNAgent(game_config)
    success, _ = ai_agent.load_weights(args.model_path)

    if not success:
        print(f"Error: Could not load model from {args.model_path}. Exiting.")
    else:
        record_expert_play(game_config, ai_agent, args.name)
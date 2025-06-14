# Advanced DQN Tetris AI

## Overview

This project is an advanced Tetris AI powered by a Deep Q-Network (DQN) built with PyTorch and Pygame. The primary mission is not just to play well, but to train an agent to learn and master a specific, expert-level strategy: the **Side 2-Wide (s2w) well**, which enables high-scoring, long-running combos.

The project features a complete training and evaluation framework, including detailed logging, checkpointing, and multiple operational modes for play, analysis, and visualization. Through an iterative process of sophisticated reward shaping and feature engineering, the AI has successfully learned this complex strategy and achieved superhuman performance.


[DEMO]
(ai_playback_ep18800_20250607_233308.mp4)



## Key Features

-   **Expert Strategy Goal**: AI is specifically trained to learn the `s2w` (Side 2-Wide well) combo strategy.
-   **Advanced State Representation**: Utilizes a 7-dimensional feature vector, including custom-designed features like "s2w-aware bumpiness," "well occupancy," and "completed lines" for precise state evaluation.
-   **Sophisticated Reward Shaping**: Employs a complex reward function with a high-value, incremental combo bonus system and targeted penalties to guide behavior.
-   **Multiple Operational Modes**: Includes `manual` play, AI `train` mode, AI `playback` mode, and an interactive `analyze` mode.
-   **Interactive Analysis Mode**: A powerful tool to visualize the AI's decision-making process in real-time, showing the V-score for every possible move.
-   **Built-in Game Recorder**: The AI playback mode can be configured to record gameplay as a high-quality MP4 or GIF file with a set time limit.
-   **Comprehensive Training Framework**: Supports checkpointing to resume training, detailed CSV logging of all reward components, and annotated evaluation screenshots.

## Tech Stack

-   **Game Engine**: Pygame
-   **Neural Network**: PyTorch
-   **Numerical Computation**: NumPy
-   **Configuration**: PyYAML

## Setup and Installation

1.  **Install PyTorch**: First, install PyTorch by following the official instructions for your system (CPU or CUDA-enabled GPU). Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the correct command.

2.  **Install Dependencies**: Create a file named `requirements.txt` with the following content:

    ```
    pygame
    PyYAML
    numpy
    imageio
    imageio-ffmpeg
    Pillow
    ```
    Then, in your activated Python virtual environment, run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the project from your terminal using `main.py` with the `--mode` flag.

-   **Manual Mode**:
    ```bash
    python main.py --mode manual
    ```
-   **AI Playback Mode** (loads the latest model automatically):
    ```bash
    python main.py --mode ai
    ```
-   **Interactive Analysis Mode** (requires a trained model):
    ```bash
    python main.py --mode analyze --model_path weights/your_model.pth
    ```
-   **Training Mode**:
    * To start a new training run:
        ```bash
        python main.py --mode train
        ```
    * To continue training from a specific checkpoint:
        ```bash
        python main.py --mode train --model_path weights/your_model.pth
        ```
---

### 4. Major Evolution Milestones

1.  **Initial Architecture (2025-06-07)**: Established the project architecture using Pygame and PyTorch, based on a DQN variant that learns a state-value function `V(s')` for post-move states.
2.  **Reward Shaping Iteration (2025-06-07)**: Overcame the "lazy AI" problem by removing a large immediate `piece_drop_reward` and introducing a high-value, delayed `combo_base_reward`, forcing the AI to pursue line-clearing combos.
3.  **Advanced Feature Engineering (2025-06-07)**: Expanded the state vector from 4 to 7 dimensions, adding **"Completed Lines," "Resulting Combo Count,"** and **"Well Occupancy"** to give the AI better foresight and situational awareness.
4.  **Custom Game Mechanics for s2w (2025-06-07)**: Implemented a custom **"s2w-aware bumpiness"** calculation that ignores the two rightmost columns, creating a "penalty-free zone" crucial for learning the s2w well structure.
5.  **Training Stability Improvements (2025-06-07)**: Implemented Min-Max feature scaling and down-weighted the influence of the volatile "Generalized Wells" feature to stabilize the training process.
6.  **Breakthrough Performance (2025-06-07)**: The AI achieved superhuman performance in evaluation mode (clearing >1200 lines), validating the effectiveness of the advanced features and "harsh but fair" reward parameters.
7.  **Visualization & Analysis Tools (2025-06-07)**: Developed the interactive analysis mode and GIF/MP4 recording functionality to allow for deep, qualitative analysis of the AI's behavior and for sharing results.

### 5. Recently Resolved Issues

-   (2025-06-07) Solved the AI's tendency to misuse the "penalty-free zone" as a dumping ground by adding a direct `well_occupancy_penalty`.
-   (2025-06-07) Fixed a critical bug in the SRS rotation logic where S and Z tetrominoes had incorrect shape definitions.
-   (2025-06-07) Optimized the action enumeration process by de-duplicating moves for symmetrical pieces, increasing decision-making efficiency.
-   (2025-06-07) Resolved the paradox of high `Loss` values co-occurring with massive performance gains by identifying it as a symptom of rapid "value function reassessment."
-   (2025-06-07) Addressed and fixed multiple `AttributeError` and `ModuleNotFoundError` issues to enable all game modes, including manual, analysis, and video recording.

### 6. TODO / Next Priorities

1.  **Primary Goal: Continued Training & Convergence**: The immediate priority is to continue the current successful training run long-term. The goal is to allow Epsilon to fully decay and observe if the average performance (`AvgLines100`, etc.) converges to a stable, high-level plateau.
2.  **Performance Ceiling Analysis**: Determine the ultimate performance limit of the current AI under the existing parameter set.
3.  **Fine-Tuning (Post-Convergence)**: Once performance has plateaued, experiment with reducing the `learning_rate` to allow for final, fine-grained optimization of the policy.
4.  **Strategic Analysis**: Use the `analyze` mode and recorded videos to perform a deep qualitative analysis of the AI's high-level strategy, identifying any remaining subtle flaws or potential areas for improvement.
5.  **Future Experiments (Optional)**: After the current model is finalized, use it as a baseline to explore new challenges, such as teaching it different strategies (e.g., T-Spins) with a redesigned reward function or testing its robustness in environments with increasing speed.




-----

## 🧠 Interactive Neural Network Visualizations

To better understand the inner workings of our DQN agent, we've created two interactive visualization scripts using Pygame. These scripts demystify the core concepts of **forward propagation** (how the AI makes a decision) and **backward propagation** (how the AI learns from its mistakes) in a simplified neural network.

### Forward Pass: From Features to Q-Value

This visualization demonstrates how the network processes a set of input features (like board height, number of holes, etc.) to produce a single output Q-value, which represents the quality of a potential move.

**Key Concepts Demonstrated:**

  * **Step-by-Step Calculation:** Press `[SPACE]` to advance one step at a time, watching each neuron's activation value get computed.
  * **Weighted Sum & Bias:** The formula at the bottom of the screen shows how inputs from the previous layer are multiplied by their corresponding weights.
  * **Activation Function:** See the result of the calculation before and after the **ReLU** activation function is applied.
  * **Data Flow:** Visually trace the flow of information from the input layer, through the hidden layers, to the final output.

-----

## 🧠 Interactive Neural Network Visualizations

These Pygame scripts offer a step-by-step, interactive visualization of the neural network's core processes. They are designed to provide an intuitive understanding of how the AI model works, from decision-making to learning.

### Forward Pass: From Features to Q-Value

This script demonstrates how a set of input features are processed layer-by-layer to compute a final Q-Value. As shown in the screenshot below, you can advance the calculation step-by-step to see how each neuron's activation is derived from the previous layer's values and its connecting weights.
![Forward Pass Visualization](forward_pass.png)

### Backward Pass: Learning from Mistakes

This script visualizes the backpropagation algorithm. Starting from a final `Loss`, it shows how the error signal is propagated backward to update the network's weights. The visualization below highlights the accumulation of error at each neuron and the subsequent weight updates, illustrating how the network "learns" from its prediction errors.
![Backward Pass Visualization](backward_pass.png)

### How to Run

**1. Execute the scripts:**

```bash
# To run the forward pass visualization
python visualize_forward_pass_stepwise.py

# To run the backward pass visualization
python visualize_backward_pass_stepwise.py
```

**2. Controls:**

  * **`[SPACE]`**: Advance the animation by a single step.
  * **`[R]`**: Reset the animation with a new set of random weights.
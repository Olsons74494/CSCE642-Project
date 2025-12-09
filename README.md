# Vineyard Irrigation RL Simulator

This project provides a simulation environment for optimizing irrigation in a vineyard using Reinforcement Learning (RL). It includes a surrogate model of vineyard growth and water dynamics, RL agents for decision-making on irrigation, and tools for evaluation against a baseline fixed-schedule irrigation strategy.

The simulation models grape growth across stages (Bud Break, Flowering, Fruit Set, Ripening, Harvest), accounting for factors like evapotranspiration, rainfall, and water stress penalties on crop quantity and quality.

## Files
- **Full_Simulator.ipynb**: Jupyter notebook for training and saving RL agents (centralized and local agents) using a policy gradient method.
- **Evaluate_Model.ipynb**: Jupyter notebook for evaluating the trained RL agents against a baseline irrigation schedule over multiple simulations, including statistical comparisons and visualizations.

## Requirements
- Python 3.12+ (tested with 3.12.3)
- Libraries: numpy, torch, matplotlib, pandas, seaborn (install via `pip install numpy torch matplotlib pandas seaborn`)
- Jupyter Notebook or JupyterLab for running the .ipynb files.

## Setup
1. Clone or download the repository.
2. Install dependencies: `pip install -r requirements.txt` (if provided; otherwise, install the libraries listed above).
3. Ensure a `models/` directory exists (it will be created automatically during training).
4. (Optional) For results storage, ensure a `results/` directory exists.

## Training RL Agents
1. Open **Full_Simulator.ipynb**.
2. Run all cells up to the final one (this sets up the environment, model, and training function).
3. Modify the final cell to customize training parameters if needed:
   - `alpha_quantity`, `alpha_quality`, `alpha_steps`: Weights for quantity, quality, and simulation duration in the reward function.
   - `beta`: Penalty weight for water usage.
   - `num_episodes`: Number of training episodes (default: 1000; reduce if computationally intensive).
   - `lr`: Learning rate (default: 0.001).
   - `gamma`: Discount factor (default: 1).
   - `num_locals`: Number of local regions/agents (default: 4, for a 2x2 grid split).
   - `end_on_death`: Whether to end episodes early on crop death (default: True).
   - `rand_et`: Enable random evapotranspiration variability (default: True).
   - `rain_chance`, `rain_amounts`, `rain_probs`: Rainfall simulation parameters.
   - `save_path`: Directory to save trained models (default: "./models").

   Example call (as in the notebook):
   central, locals_list, rewards_history, episode_durations = train_agents(
	alpha_quantity=1.0, alpha_quality=1.0, alpha_steps=5.0, beta=0.1,
	num_episodes=1000, lr=0.001, gamma=1, num_locals=4,
	end_on_death=True, rand_et=True, rain_chance=0.05,
	rain_amounts=[0.25, 0.5, 0.75, 1.0], rain_probs=[0.45, 0.3, 0.15, 0.1],
	save_path="./models"
	)
	
4. Run the final cell to train and save the models to the specified `save_path`. This will display live plots of rewards and episode durations during training.

Trained models will be saved as `central.pth` and `local_0.pth` to `local_3.pth` (for 4 locals).

## Evaluating Models
1. Ensure trained models are saved in the `models/` directory (from the training step).
2. Open **Evaluate_Model.ipynb**.
3. Run all cells up to the evaluation section.
4. Modify the evaluation calls if needed:
- In the RL evaluation: Adjust `num_simulations` (default: 1000; reduce for faster runs), `num_locals`, etc.
- In the baseline evaluation: Adjust `num_simulations`, `irrigation_schedule` (fixed amount, default: 0.5), `irrigate_every` (frequency in steps, default: 5).
5. Run the evaluation cells to:
- Load and simulate with RL agents.
- Simulate the baseline.
- Compute statistics (mean/std for quantity, quality, water usage, rewards; death rates).
- Generate box plots for comparisons (rewards, quantity, quality, water usage, efficiencies) and a bar chart for death rates.
6. Results are saved as CSV files in `./results/` (e.g., `rl_agent_results.csv`, `baseline_results.csv`).

## Computational Notes
- **Training and Evaluation Intensity**: Training with 1000 episodes and evaluating with 1000 simulations can be computationally heavy (e.g., CPU-bound due to NumPy/Torch operations). On modest hardware, this may take hours.
- Reduce `num_episodes` (e.g., to 100-500) for training.
- Reduce `num_simulations` (e.g., to 100-200) for evaluation.
- **Google Colab Alternative**: You can upload the notebooks to Google Colab for free GPU/CPU resources.
- Minor modifications: Use `!pip install <library>` for dependencies.
- For file saving/loading, use Google Drive mounting (`from google.colab import drive; drive.mount('/content/drive')`) and adjust paths (e.g., `/content/drive/MyDrive/models/`).
- Enable GPU runtime in Colab for faster training if needed (though this project is more CPU-intensive).

## Limitations and Future Work
- The surrogate model is simplified (e.g., no real diffusion implemented).
- RL uses a basic policy gradient; advanced methods (e.g., PPO, A2C) could improve stability.
- Tune hyperparameters for better performance.
- Real-world validation would require integration with actual vineyard data.

For questions or contributions, feel free to open an issue or pull request!
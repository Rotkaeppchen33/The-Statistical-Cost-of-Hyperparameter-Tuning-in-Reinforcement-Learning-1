# The Statistical Cost of Hyperparameter Tuning in Reinforcement Learning

This repository contains code for analyzing the sensitivity of reinforcement learning algorithms to hyperparameter configurations.

The project is adapted from an existing open-source baseline and extends it with additional preprocessing and analysis scripts for fair algorithm comparison under fixed interaction budgets.

------------------------------------------------------------

Setup

pip install -r requirements.txt

------------------------------------------------------------

Running Experiments

Training scripts are located in src/.

Example:
python src/ppo_continuous_action.py --alg_type PPO --env_name ant

------------------------------------------------------------

Analysis

Post-processing and analysis scripts are located under sec5/.

They are used to organize experiment outputs, normalize rewards, compute AUC-style metrics, and generate plots and summary tables.

------------------------------------------------------------

Notes

- Large raw training logs are not included.
- Paths are written to be relative and should be adjusted to local data locations.
- System- or user-specific files should be removed before public release.

------------------------------------------------------------

License

See LICENSE.
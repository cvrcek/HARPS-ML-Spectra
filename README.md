# HARPS-ML-Spectra
Spectra research code (learning, inference, tests).

This repository contains the companion code for the paper ["Stellar parameter prediction and spectral simulation using machine learning"](https://arxiv.org/abs/2412.09002) by Cvrček et al.

## Setup Instructions
To set up this project on your machine, follow these steps:

1. Make sure you have Python 3.10 installed on your system.

2. Run the setup script to create a virtual environment and install dependencies:
   ```bash
   ./HARPS_DL_env.sh
   ```
   This will:
   - Create a virtual environment called 'replicated_env'
   - Activate the environment
   - Install the package in development mode with all dependencies

3. Add the project folder to your PYTHONPATH by adding this line to your .bashrc file:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/full/path/to/HARPS-ML-Spectra
   ```
   Replace `/full/path/to` with the actual path to where you cloned this repository.

4. When working with the project, activate the virtual environment:
   ```bash
   source replicated_env/bin/activate
   ```

## Running the Code
After setup, you can run the demo notebooks in the `demo_notebooks` directory to see examples of:
- Label prediction
- Spectra reconstruction

## Project Structure
- `src/` - Source code for the HARPS_DL package
- `demo_notebooks/` - Jupyter notebooks demonstrating usage
- `models/` - Pre-trained models

## References
This project is the companion code to:
- Cvrček, V., Romaniello, M., Šára, R., Freudling, W., & Ballester, P. (2024). "Stellar parameter prediction and spectral simulation using machine learning". [arXiv:2412.09002](https://arxiv.org/abs/2412.09002)

The CNN models in this work are based on the design described in:
- Sedaghat, N., et al. (2021). "Machines Learn to Infer Stellar Parameters Just by Looking at a Large Number of Spectra." The Astrophysical Journal, 908(2), 217. [arXiv:2009.12872](https://arxiv.org/abs/2009.12872)

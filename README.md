# spectra_DL
Spectra research code (learning, inference, tests).

# setup instructions
- Run The script setup_env.sh to setup virtual environment 'replicate_env' that replicates environment used during development.
- Add project folder to the environment variable PYTHONPATH, by adding this line to .bashrc:
    ./setup_env.sh
    export PYTHONPATH=$PYTHONPATH:{path_to_project}/HARPS-ML-Spectra
- 

# container instructions
We provide instructions to set up a container that we used to training and testing our models.
With this container, it is possible to run our demo code and learn or fine-tune our models.

- install apptainer according to these instructions https://apptainer.org/docs/user/latest/quick_start.html#installation
- cd container
- apptainer build --fakeroot spectra_container.sif spectra_container.def

The code can be executed as follows:
apptainer exec --nv spectra_container.sif python3.10 script.py

Alternatively, one can set up virtual environment that is described in the container/spectra_container.def.
Which is sufficient to run the models on CPU.

# 

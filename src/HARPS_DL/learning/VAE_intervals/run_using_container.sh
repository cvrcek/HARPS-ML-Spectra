container=/home/cv/Documents/singularity_tests/nvidia_py310_venv/spectra_container.sif 
#singularity exec --bind /media --nv $container bash -c 'python3.10 learn_VAE_intervals.py --print_config > config_new.yaml'
singularity exec --bind /media --nv $container bash -c 'python3.10 learn_VAE_intervals.py --config config_new.yaml'

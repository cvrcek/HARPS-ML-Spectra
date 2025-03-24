from setuptools import setup, find_packages

setup(
    name="HARPS_DL",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "HARPS_DL": ["datasets/*.json", "datasets/*.yaml"],
    },
    install_requires=[
        "numpy==1.24.3",
        "scipy==1.10.1",
        "scikit-learn==1.2.2",
        "lightning==2.0.2",
        "tensorboard==2.12.2",
        "astropy==5.2.2",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
        "pandas==2.0.1",
        "pyyaml==6.0",
        "jsonargparse[signatures]==4.21.0",
        "pytest==7.3.1",
        "statsmodels==0.13.5",
        "ipywidgets==8.0.6",
        "jupyterlab==4.0.2",
        "notebook==6.5.4",
        "ipympl==0.9.3",
        "neptune==1.3.1",
    ],
)

# Providing Previously Unseen Users Fair Recommendations Using Variational Autoencoders
This is the official repository for the paper "Providing Previously Unseen Users Fair Recommendations Using Variational Autoencoders", appearing in the RecSys 2023 conference.

## Requirements(version used)
The packages required for running the code are the following:
- numpy (1.22.1)
- scikit-learn (1.0.2)
- pandas (1.3.5)
- scipy (1.7.2)
- torch (1.10.1+cu113)
- SLIM (2.0.0)

A "pip freeze" dump of the development python environment is also provided in requirements.txt. However, this also list numerous packages that will be installed when installing the listed packages, auxilliary packages used for formatting and prototyping, and packages like SLIM that are not listed by pip.

## Running the code
train.py is used for training and evaluating all setups of the VAE models and supports argument parsing through pythons argparse package. To list the available arguments, descriptions and instructions, run "python train.py -h". setups.txt lists the model configurations presented in the main results of the paper.

We also provide the code we used for evaluating the SLIM baseline, which can be found in eval_slim.py. For this file we do not provide argument parsing as the configurations only depend on the considered dataset.

### Logging and plotting
The "verbose" flag can be used to enable logging and plotting through Weights & Biases. Users who want to use other logging tools must change the code accordingly.

## Specific python packages and code inspiration
The SLIM baseline was evaluated using the public implementation of the authors, which can be found here: https://github.com/KarypisLab/SLIM. The code for evaluating SLIM is found in eval_slim.py and requires that SLIM has been built and installed.

The "base" setup of the proposed models takes inspiration from the public implementation of the baseline VAE model found here: https://github.com/dawenl/vae_cf.  

util.py contains some borrowed code. The specific part is clearly marked and sourced.

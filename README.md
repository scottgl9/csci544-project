# Attenuating Bias in Pre-trained Language Models using MABEL

This project is towards the course project for CSCI 544, USC.


# Setup

**Option 1**: Create a conda environment using the given `environment.yml` file

```bash
conda env create -f environment.yml
```

**Option 2**: Use the `requirements.txt` file to install required packages

```bash
pip install -r requirements.txt
```

Run the prelaunch bash script to create requisite directories and install the requirements for the project

```bash
chmod +x prelaunch.sh  # Make sure that the script has execute permissions
./prelaunch.sh
```

# Quick Start

Preprocessing the data
* Loads the dataset from huggingface (SNLI/MNLI) datasets
* Loads word pairs based on the wordlist shared in the gn_glove repository
* Augments the huggingface dataset with the word pairs. Refer to the [preprocessing example notebook](./source/preprocessing_example.ipynb) for more information on how to perform Counterfactual Data Augmentation

```bash
python main.py data_prep -d snli --wl-path data/wordlist/male_word_file.txt --wl-path data/wordlist/female_word_file.txt
```

The augmented dataset is stored in `source/data/data_aug.csv` by default.

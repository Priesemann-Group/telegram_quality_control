# Telegram quality control

This repository contains the code for the paper "TeraGram: A Structured Longitudinal Dataset of the Telegram Messenger".

If you use the dataset, please cite it as follows: 

```
@misc{TeraGram,
  title = {TeraGram: A Structured Longitudinal Dataset of the Telegram Messenger},
  author = {Golovin, Anastasia and Mohr, Sebastian B. and Gottwald, Arne I. and Hvid, Ulrik and Trivedi, Srushhti and Neto, Joao P. and Schneider, Andreas C. and Priesemann, Viola},
  note = {in review},
}
```

## How to access the dataset

As the paper is currently under review, the dataset is not yet available for the public. A preview of the dataset in CSV format is available [here](https://zenodo.org/records/18262126).

## How to install the project

We use [Poetry](https://python-poetry.org/) to manage Python environments and dependencies. To install Poetry on Linux, Windows or macOS, go to the [documentation](https://python-poetry.org/docs/#installation). Then, use

```bash
poetry install
```

to create a new environment for the project and install all dependencies. 

To install topic modeling dependencies, you need to install additional GPU libraries with `poetry install -E cu130`. To plot the entity-relation diagram, you need to [install GraphViz](https://graphviz.org/download/) on your system and then call `poetry insatll -E erd`. 

## How to connect to the database

To connect to a locally running copy of the database, you need to provide your credentials in the `.env` file. The file `example.env` provides the environment variables that need to be set. Copy the file, rename it to `.env` and set the correct credentials. 

The variables `OUTPUT_FOLDER` and `SCRATCH_FOLDER` set the directories where the results will be stored. The `OUTPUT_FOLDER` contains final results and the `SCRATCH_FOLDER` is used for caching intermediate states. The cache can get large in size or contain many files that would overload the backup infrastructure, so it makes sense to separate those folders on a cluster. If those two things are not an issue in your environment, both folders can be set to local subfolders of the project. 

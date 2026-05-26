# Telegram quality control

This repository contains the code for the paper "TeraGram: A Structured Longitudinal Dataset of the Telegram Messenger".

If you use the dataset, please cite it as follows: 

```
@misc{TeraGram,
  title = {TeraGram: A Structured Longitudinal Dataset of the Telegram Messenger},
  author = {Golovin, Anastasia and Mohr, Sebastian B. and Gottwald, Arne I. and Hvid, Ulrik and Trivedi, Srushhti and Neto, Joao P. and Schneider, Andreas C. and Priesemann, Viola},
  note = {accepted to ICWSM 2026},
}
```

## How to access the dataset

There are two versions of the dataset:
- The full dataset is available [here](https://doi.org/10.25625/GDCXQK) in Parquet format. Large tables are split into batches with 100M rows in each. 
- A preview of the dataset in CSV format is available [here](https://zenodo.org/records/18262126). This dataset contains messages from 1% of chats from the full dataset. It is intended for quick exploration and prototyping. 

## How to install the project

We use [Poetry](https://python-poetry.org/) to manage Python environments and dependencies. To install Poetry on Linux, Windows or macOS, go to the [documentation](https://python-poetry.org/docs/#installation). Then, use

```bash
poetry install
```

to create a new environment for the project and install all dependencies. 

To install topic modeling dependencies, you need to install additional GPU libraries with `poetry install -E cu130` (you will need to adjust the Cuda version in `pyproject.toml` depending on your setup). To plot the entity-relation diagram, you need to [install GraphViz](https://graphviz.org/download/) on your system and then call `poetry install -E erd`. Those installations are optional and are only needed for a few specific analyses. 

## How to ingest the dataset into a Postgres database

Due to the large size of the dataset, we strongly recommend to load the dataset into a relational database instead of working with individual parquet files. For convenience, we provide a docker compose file and an ingestion script to closely replicate our setup. The steps here refer to the full dataset in Parquet format. 

### Prerequisites

1. **Download the data files**: Download the parquet files from the [storage location](https://doi.org/10.25625/GDCXQK) and store them in a local folder. 

2. **Configure the environment**: Add the following variables to your `.env` file (see `example.env` for reference):
   - `DATABASE_FOLDER`: The folder where the docker container for the Postgres database will be created. We recommend allocating at least 4TB of storage for this. 
   - `RAW_DATA_FOLDER`: The path to the folder containing the downloaded compressed parquet files
   - `DB_NAME`: Name of the database to create
   - `DB_USER`: Postgres username
   - `DB_PASSWORD`: Postgres password
   - `DB_HOST`: Database hostname (e.g., `localhost`)
   - `DB_PORT`: Database port (e.g., `5432`)
   - `PGADMIN_EMAIL`: Email for pgAdmin admin panel
   - `PGADMIN_PASSWORD`: Password for pgAdmin admin panel

Additionally, those environment variables are needed if you want to reproduce our results:

  - `OUTPUT_FOLDER`: output of the analyses will be saved here
  - `SCRATCH_FOLDER`: a cache folder for storing intermediate results. Note that those files can get very large in size (~100GB). 

### Ingestion Steps

1. **Start the database services**:
   ```bash
   docker-compose up -d
   ```
   This starts a PostgreSQL database and pgAdmin (accessible at `http://localhost:5050`).

2. **Run the ingestion script**:
   ```bash
   poetry run python3 scripts/dataset_ingestion.py
   ```

   This script will:
   - Create a Postgres database and the tables. Note that if the database already exists, the script will try to delete the old database and recreate it. 
   - Load all parquet files from the `RAW_DATA_FOLDER` into the database
   - Apply constraints and indexes to optimize queries

The ingestion process reads all parquet files from the configured `RAW_DATA_FOLDER` and inserts their data into the corresponding PostgreSQL tables. Ensure all parquet files are in the correct folder structure before running the script. The script assumes the same folder structure as in the storage, e.g., that the files are named either `<tablename>.parquet` or are stored in a folder `<tablename>/<tablename>_batch_xx.parquet`. 

If the ingestion script throws any errors, please create a Github issue and we will be happy to help! 
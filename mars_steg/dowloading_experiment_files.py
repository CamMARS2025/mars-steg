import json
import re
import ray
import wandb
import pandas as pd
import argparse
import logging
import os
import yaml
import logging.config



def setup_logging(config):
    logging.config.dictConfig(config['LOGGING_INFO'])
    logger = logging.getLogger(__name__)
    return logger


def download_files(id_experiment, output_dir, config, test=False):
    API = wandb.Api(timeout=29)
    logger = setup_logging(config)
    run = API.run(f"{id_experiment}") #WATCH TO NOT OVERRIDE EXISTING NICKNAME
    print(f"Downloading files from experiment: {run.name}")
    if test:
        step = re.search(r"step_(\d+)", run.name)
        output_root = os.path.join(output_dir, "test", f"step_{step.group(1)}")
    else:
        output_root = os.path.join(output_dir, "train")
    if os.path.exists(output_root):
        print("✅ Folder with relevent files exists")
        return output_root
    else:
        os.makedirs(output_root)
        if test:
            name_batch = "Merged_test_Batch_prompt_data"
        else:
            name_batch = "Merged_Batch_prompt_data"
        for artifact_ref in run.logged_artifacts():
            if name_batch in artifact_ref.name:
                logger.debug(f"Scanning: {artifact_ref.name}")
                number_tries = 0
                while number_tries < 5:
                    try:
                        for file in artifact_ref.files():
                            if file.name.endswith(".json"):
                                output_path = os.path.join(output_root, f"{artifact_ref.name}_{file.name}")
                                print(f"Downloading {file.name} -> {output_path}")
                                file.download(root=output_root, replace=True)
                        break

                    except Exception as e:
                        print(f"❌ Failed to process {artifact_ref.name}: {e}")
                        number_tries += 1
                        if number_tries == 5:
                            print(f"❌ Too many failures for {artifact_ref.name}. Skipping.")
                            break

    print("✅ Done downloading all relevant artifacts.")
    return output_root

def create_csv_from_json_files(output_root, config, return_df=False):
    if os.path.exists(os.path.join(output_root, "merged_data.csv")):
        logger = setup_logging(config)
        logger.warning("✅ Merged CSV already exists. Skipping CSV creation.")
        if return_df:
            return pd.read_csv(os.path.join(output_root, "merged_data.csv"))
    logger = setup_logging(config)
    all_dfs = []
    for filename in os.listdir(output_root):
        if filename.endswith(".json"):
            file_path = os.path.join(output_root, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            df = pd.DataFrame(data['data'], columns=data['columns'])
            logger.debug(f"Processing file: {filename}")
            if 'test' in output_root:
                df['step'] = re.search(r"step_(\d+)", file_path).group(1)  # Extract step from filename
            all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(os.path.join(output_root, "merged_data.csv"), index=False)
    if return_df:
        return merged_df
    
def download_and_create_csv_for_a_seed(output_dir, train_id_experiment, test_ids_experiment, config):
    logger = setup_logging(config)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        logger.warning("✅ Folder with this experiment already exists. Please check not overriding existing data.")
    
    if isinstance(test_ids_experiment, str):
        test_ids_experiment = [test_ids_experiment]

    logger.info(f"Downloading files from train experiment: {train_id_experiment}")
    train_output_root = download_files(train_id_experiment, output_dir, config, test=False)
    create_csv_from_json_files(train_output_root, config, return_df=False)

    list_dfs = []
    for i, test_id_experiment in enumerate(test_ids_experiment):
        logger.info(f"Downloading files from test experiment: {test_id_experiment}")
        test_output_root_k = download_files(test_id_experiment, output_dir, config, test=True)
        logger.info(f"Creating CSV from test experiment: {test_id_experiment}")
        list_dfs.append(create_csv_from_json_files(test_output_root_k, config, return_df=True))
    merged_test_df = pd.concat(list_dfs, ignore_index=True)
    merged_test_df.to_csv(os.path.join(output_dir, "test", "merged_data.csv"), index=False)




@ray.remote(num_cpus=1)
def process_seed(seed, experiments, config, model_name):
    logger = setup_logging(config)
    train_id_experiment = experiments['train_experiment_id']
    test_id_experiment = experiments['test_experiment_ids']

    logger.debug(f"Train experiment ID: {train_id_experiment}")
    logger.debug(f"Test experiment IDs: {test_id_experiment}")

    if train_id_experiment is None:
        logger.warning(f"❌ No train experiment found for seed {seed}. Skipping.")
        return

    if not test_id_experiment:
        logger.warning(f"❌ No test experiment found for seed {seed}. Skipping.")
        return

    output_dir = os.path.join(config['output_dir'], f"{model_name}_seed_{seed}")
    download_and_create_csv_for_a_seed(output_dir, train_id_experiment, test_id_experiment, config)
    logger.info(f"CSV files created for seed {seed}.")



def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    entity = config['entity_name']
    test_project = config['test_project_name']
    train_project = config['train_project_name']
    seed_to_download = config['seed_to_download']
    model_name = config['model_name']

    logger = setup_logging(config)
    logger.info("Starting the script to download experiment files from W&B.")

    logger.info(f"Fetching runs from {entity}/{train_project} and {entity}/{test_project}")
    API = wandb.Api()
    train_runs = API.runs(f"{entity}/{train_project}")
    test_runs = API.runs(f"{entity}/{test_project}")

    experiments_dict = {seed : {'train_experiment_id' : None,
                                'test_experiment_ids': []} for seed in seed_to_download}

    logger.info(f"Found {len(train_runs)} train runs and {len(test_runs)} test runs.")
    for seed in seed_to_download:
        logger.info(f"Searching for model {model_name} and seed {seed} in train and test runs.")
        for run in test_runs:
            logger.debug(f"Checking test run: {run.name}")
            if f'seed_{seed}' in run.name and model_name in run.name:
                experiments_dict[seed]['test_experiment_ids'].append(f'{entity}/{test_project}/{run.id}')
                step = re.search(r"step_(\d+)", run.name).group(1)  # Extract step from filename
                
                logger.info(f"Found test experiment for seed {seed}: {run.id}. Step {step}")
        for run in train_runs:
            logger.debug(f"Checking train run: {run.name}")
            if f'seed_{seed}' in run.name and model_name in run.name:
                experiments_dict[seed]['train_experiment_id'] = f'{entity}/{train_project}/{run.id}'
                logger.info(f"Found train experiment for seed {seed}: {run.id}")
                break

    logger.info("All experiments found. Starting download and CSV creation.")
    ray.init(ignore_reinit_error=True)
    futures = [
        process_seed.remote(seed, experiments, config, model_name)
        for seed, experiments in experiments_dict.items()
    ]
    ray.get(futures)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download experiment files from W&B.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    main(args.config_file)


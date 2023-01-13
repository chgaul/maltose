"""
Script to evaluate the estimation errors of a given model on various test sets.
"""
import os.path
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import importlib

import evaluation
from multitask_data import DATASET_NAMES


base_dir = os.path.join(Path(__file__).parent, "..")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True,
    help="Name of the configuration to be trained, e.g., multitask_model_v01.")
parser.add_argument(
    "--model-base-dir",
    default=os.path.join(base_dir, "models"),
    help="Parent directory for the trained model.")
parser.add_argument(
    "--data-base-dir",
    default=os.path.join(base_dir, "data"),
    help="Base directory where the primary data is located.")
parser.add_argument(
    "--device", default="cpu",
    help="Device for running the training. For example 'cpu', 'cuda', 'cuda:2'")
parser.add_argument(
    "--test", required=False, nargs='*',
    help="Names of the tests to be executed. Default is to execute all tests.")
args = parser.parse_args()

def model_dir(model_name):
    return os.path.join(args.model_base_dir, model_name)

def load_model(model_name):
    mdir = model_dir(model_name)
    state_path = os.path.join(mdir, 'best_model_state.pth')
    model_path = os.path.join(mdir, 'best_model')
    if os.path.exists(model_path) and (
            not os.path.exists(state_path)
            or os.path.getmtime(state_path) < os.path.getmtime(model_path)):
        print('Loading', model_path)
        model = torch.load(model_path, map_location=args.device)
        model.eval()
        print("Saving the model's state dictionary to", state_path)
        torch.save(model.state_dict(), state_path)
    if os.path.exists(state_path):
        module_name = 'configs.{}'.format(model_name)
        print('Importing module {}...'.format(module_name))
        config = importlib.import_module(module_name)
        model = config.build_model()
        print('Loading', state_path)
        model.load_state_dict(torch.load(state_path))
        model.eval()
        model.to(args.device)
    else:
        raise RuntimeError('model_name {} not found at {}'.format(
                model_name, mdir))
    return model, os.path.getmtime(state_path)

if args.test is None:
    TESTS = set(DATASET_NAMES + ['Kuzmich2017'])
else:
    TESTS = set(args.test)

model_name = args.config
data_base_dir = args.data_base_dir
model, model_timestamp = load_model(model_name)

# Properties estimated by the model
est_properties = evaluation.get_available_properties(model=model)

# Check if the computation has been done before
summary_file = os.path.join(model_dir(model_name), 'deviations_summary.json')
if os.path.exists(summary_file) and not os.path.getmtime(summary_file) < model_timestamp:
    print('Summary file {} exists and is up to date (will not re-compute).'.format(
        summary_file))
    summary = pd.read_json(os.path.join(model_dir(model_name), 'deviations_summary.json'))
else:
    summary = pd.DataFrame(columns=[
        'test', 'property', 'mean(error)', 'std(error)', 'MAE', 'RMSE', 'size'])

# Run all available tests except those that are already in the summary file:
TESTS = TESTS - set(summary['test'])

# Compute and dump the full error distribution

def compute_deviations_general(test):
    target_file = os.path.join(model_dir(model_name), test + '_deviations.npz')
    if not os.path.exists(target_file) or os.path.getmtime(target_file) < model_timestamp:
        data = evaluation.evaluate_unified(
            model, test,
            n_points=None, seed=None,
            device=args.device,
            data_base_dir=data_base_dir)
        devs = {}
        print(test)
        for p in data['tgt'].keys():
            if p in data['est']:
                print('  ', p)
                devs[p] = data['est'][p] - data['tgt'][p]
        np.savez(target_file, **devs)
        return devs
    else:
        print('Target file {} up to date (will not re-compute).'.format(
            target_file))
    return np.load(target_file)


# Special treatment of Kuzmich2017, which is not in the form of an ase db:
def compute_deviations_kuzmich():
    test = 'Kuzmich2017'
    target_file = os.path.join(model_dir(model_name), test + '_deviations.npz')
    if not os.path.exists(target_file) or os.path.getmtime(target_file) < model_timestamp:
        data = evaluation.evaluate_kuzmich(
            model, data_base_dir, device=args.device)
        devs = {}
        print(test)
        for p in data['tgt'].keys():
            if p in data['est']:
                print('  ', p)
                devs[p] = data['est'][p] - data['tgt'][p]
        np.savez(target_file, **devs)
        return devs
    else:
        print('Target file {} up to date (will not re-compute).'.format(
            target_file))
        return np.load(target_file)

def compute_deviations(test):
    if test=='Kuzmich2017':
        return compute_deviations_kuzmich()
    else:
        return compute_deviations_general(test)


# Compute and summarize the deviations (DataFrame, json)
for test in TESTS:
    devs = compute_deviations(test)
    for p, dev in devs.items():
        summary = pd.concat([
            summary,
            pd.DataFrame({
                'test': test,
                'property': p,
                'mean(error)': np.mean(dev),
                'std(error)': np.std(dev),
                'MAE': np.mean(np.abs(dev)),
                'RMSE': np.sqrt(np.mean(np.square(dev))),
                'size': len(dev),
            }, index=[0])], ignore_index=True)
summary.to_json(summary_file, indent=2, orient='records')
print(summary)

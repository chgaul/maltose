"""
Script to evaluate the estimation errors of a given model on various test sets.
"""
import os.path
import argparse
import numpy as np
import pandas as pd
import torch
import importlib

import evaluation

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True,
    help="Name of the configuration to be trained, e.g., multitask_model_v01.")
parser.add_argument(
    "--model-base-dir", default="./models",
    help="Parent directory for the trained model.")
parser.add_argument(
    "--data-base-dir", default="./data",
    help="Base directory where the primary data is located.")
parser.add_argument(
    "--device", default="cpu",
    help="Device for running the training. For example 'cpu', 'cuda', 'cuda:2'")
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


RANDOMSEED = 26463461 # For shuffling the items in a reproducible way
model_name = args.config
data_base_dir = args.data_base_dir
model, model_timestamp = load_model(model_name)

# Compute and dump the full error distribution
target_file = os.path.join(model_dir(model_name), 'deviations.npz')
if not os.path.exists(target_file) or os.path.getmtime(target_file) < model_timestamp:
    est_properties = evaluation.get_available_properties(model=model)
    tgt_est = evaluation.compute_regular_data(
        model, data_base_dir, n_points=None, seed=RANDOMSEED,
        device=args.device)
    evaluation.add_kuzmich(
        tgt_est, model, data_base_dir, n_points=-1, seed=RANDOMSEED,
        device=args.device)
    devs = {}
    for test, data in tgt_est.items():
        print(test)
        for p in data['tgt'].keys():
            if p in data['est']:
                print('  ', p)
                devs[test + ':' + p] = data['est'][p] - data['tgt'][p]
    np.savez(target_file, **devs)
else:
    print('Target file {} up to date (will not re-compute).'.format(
        target_file))

# Summarize the deviations (DataFrame, json)
summary_file = os.path.join(model_dir(model_name), 'deviations_summary.json')
if not os.path.exists(summary_file) or os.path.getmtime(summary_file) < os.path.getmtime(target_file):
    devs = np.load(target_file)
    summary = pd.DataFrame(columns=[
        'test', 'property', 'mean(error)', 'std(error)', 'MAE', 'RMSE', 'size'])
    for k, dev in devs.items():
        test, p = k.split(':')
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
else:
    print('Summary file {} exists up to date (will not re-compute).'.format(
        summary_file))
    summary = pd.read_json(os.path.join(model_dir(model_name), 'deviations_summary.json'))

print(summary)

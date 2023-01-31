"""
Visualize the estimation errors of the trained models

Assume:
- that the models are already trained,
- that the compute_errors script has been execuded.

Thus, in the model_base_dir, there are subfolders, one for each model
configuration, containing:
- best_model_state.pth: the state dictionary of the trained model
- deviations_summary.json: Previously computed error statistics (evaluated on
  the full test sets).
"""
import argparse
import os
import string
from pathlib import Path
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import evaluation

DATASET_NAMES = ['qm9', 'alchemy', 'oe62', 'hopv']

base_dir = os.path.join(Path(__file__).parent, "..")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target-dir", type=str,
    default=os.path.join(base_dir, "figures"),
    help="Directory where the plots are saved.")
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
    "--show", action="store_true",
    help="Call matplotlib.pyplot.show() on the plots (for interactive revisions)")
parser.add_argument(
    "--figure", type=int, choices=[4, 5],
    help="which of the figures to generate")
args = parser.parse_args()


if args.figure == 4:
    configs = [
        "multitask_model_v08_sum_f",
        "multitask_model_v08_avg",
        "multitask_model_v08"
    ]
    tgts = ['HOMO-B3LYP', 'LUMO-B3LYP']
    plot_size = 4
elif args.figure == 5:
    configs = [
        "multitask_model_v08",
        "multitask_model_v06",
        "multitask_model_v07",
        "multitask_model_v01",
        "multitask_model_v05",
    ]
    tgts = ['HOMO-B3LYP', 'HOMO-PBE0', 'LUMO-B3LYP', 'LUMO-PBE0']
    plot_size = 3.2
else:
    raise RuntimeError("Bad ")
device = args.device
data_base_dir = args.data_base_dir
model_base_dir = args.model_base_dir
device = args.device

def model_dir(model_name):
    return os.path.join(model_base_dir, model_name)

def load_model(model_name):
    mdir = model_dir(model_name)
    state_path = os.path.join(mdir, 'best_model_state.pth')
    if os.path.exists(state_path):
        module_name = 'configs.{}'.format(model_name)
        print('Importing module {}...'.format(module_name))
        config = importlib.import_module(module_name)
        model = config.build_model()
        print('Loading', state_path)
        model.load_state_dict(torch.load(state_path, map_location=device))
        model.eval()
        model.to(device)
    else:
        raise RuntimeError('model_name {} not found at {}'.format(
                model_name, mdir))
    return model


def make_plot(
        ax, tgt_est: dict, qt_tgt: str, qt_est: str, n_points: int = -1,
        suppress_title=False):
    def measure_errors(x, y):
        dev = np.array(x) - np.array(y)
        mae = '{:.2f}eV'.format(np.mean(np.abs(dev)))
        rmse = '{:.2f}eV'.format(np.sqrt(np.mean(np.square(dev))))
        return mae, rmse
    def lookup_errors(test, prop):
        test_row = summary[(summary['test']==test) & (summary['property']==prop)]
        assert len(test_row) == 1
        mae = '{:.3f}eV'.format(float(test_row['MAE']))
        rmse = '{:.3f}eV'.format(float(test_row['RMSE']))
        return mae, rmse
    plot_empty = True
    plotname = '{}-{}'.format(model_name, qt_tgt);
    if qt_est != qt_tgt:
        plotname += '-cross'
    for dataset_name, te in tgt_est.items():
        try:
            x = te['tgt'][qt_tgt]
            y = te['est'][qt_est]
        except Exception as e:
            print(e)
            continue
        if qt_tgt == qt_est:
            try:
                mae, rmse = lookup_errors(test=dataset_name, prop=qt_est)
            except Exception as e:
                print(e)
                print("""Something went wrong looking up {}, {}. Measure 
                errors from plot data.""".format(dataset_name, qt_est))
                mae, rmse = measure_errors(x, y)
        else:
            mae, rmse = measure_errors(x, y)
        print('{}: MAE={}, RMSE={}'.format(dataset_name, mae, rmse))
        ax.scatter(
            x[:n_points], y[:n_points], s=16, color=te['color'],
            label='{dataset} (MAE={mae})'.format(
                dataset=dataset_name, mae=mae)
        )
        ax.axline((np.mean(x), np.mean(x)), slope=1)
        ax.set_xlabel('{} target (eV)'.format(qt_tgt))
        ax.set_ylabel('{} estimate (eV)'.format(qt_est))
        plot_empty = False
    if plot_empty:
        print("{}/{} empty for {}.".format(qt_tgt, qt_est, model_name))
    else:
        if not suppress_title:
            ax.set_title(model_name)
        ax.grid()
        ax.legend()


plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels

properties = ['HOMO', 'LUMO']
theories = ['B3LYP', 'PBE0']

for m, model_name in enumerate(configs):
    model = load_model(model_name)

    summary_file = os.path.join(model_dir(model_name), 'deviations_summary.json')
    if os.path.exists(summary_file):
        summary = pd.read_json(summary_file)
    else:
        print('Please run the "compute_errors" script for model {}'.format(model_name))
        summary = None

    n_points = 25
    RANDOMSEED = 26463461 # For shuffling the items in a reproducible way
    est_properties = evaluation.get_available_properties(model=model)
    tgt_est = {
        dataset_name: evaluation.evaluate_unified(
            model, dataset_name,
            n_points=n_points, seed=RANDOMSEED,
            device=device,
            data_base_dir=data_base_dir) for dataset_name in DATASET_NAMES
    }
    tgt_est['Kuzmich2017'] = evaluation.evaluate_kuzmich(
        model, data_base_dir,
        seed=RANDOMSEED, n_points=n_points, device=device)

    # Define a fixed color code for each test set
    for k, color in {
        'qm9': 'orange',
        'alchemy': 'red',
        'oe62': 'purple',
        'hopv': 'blue',
        'Kuzmich2017': 'black',
    }.items():
        if k in tgt_est:
            tgt_est[k]['color'] = color

    fig, axes = plt.subplots(
        len(tgts), 1, figsize=(plot_size, 0.9*len(tgts)*plot_size))

    # Target-estimate plots for each property and theory (diagonal and cross)
    for q, q_tgt in enumerate(tgts):
        if q_tgt.endswith('PBE0') and model_name in ["multitask_model_v08", 
                      "multitask_model_v06"]:
            q2 = q_tgt[:-len('PBE0')] + 'B3LYP'
            axes[q].set_facecolor('lightgray')
        else:
            q2 = q_tgt
        
        make_plot(axes[q], tgt_est, q_tgt, q2, n_points=n_points, suppress_title=True)

    tgt_dir = args.target_dir
    os.makedirs(tgt_dir, exist_ok=True)
    tgt_file = os.path.join(tgt_dir, "Figure{}{}.pdf".format(
            args.figure, string.ascii_lowercase[m]))
    print('Saving plot to {}'.format(tgt_file))
    plt.tight_layout(h_pad=1.5, w_pad=2)
    
    plt.savefig(tgt_file, dpi=200)
    if args.show:
        plt.show()

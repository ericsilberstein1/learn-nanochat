import os
import tempfile
import zipfile
import shutil
import yaml
import csv
import time
import json
import random
import torch
import argparse
from contextlib import nullcontext

from my_nanochat.my_common import get_base_dir, print0, log_memory_stats, download_file_with_lock, get_dist_info, autodetect_device_type, compute_init, compute_cleanup
from my_nanochat.my_core_eval import evaluate_task
from my_nanochat.my_engine import Engine
from my_nanochat.my_checkpoint_manager import load_model
from my_nanochat.my_report import get_report



# ~162MB of data needed to evaluate the CORE metric
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"placed eval_bundle dir at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1, tasks_to_run=None):
    # tasks_to_run - for debugging, specify a list of tasks to run by label e.g. ['hellaswag', 'jeopardy']

    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, 'eval_bundle.zip', postprocess_fn=place_eval_bundle)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']


    # see what this csv looks like in challenge 21 notebook
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)


    results = {}
    centered_results = {}
    for task in tasks:
        if tasks_to_run is not None and task['label'] not in tasks_to_run:
            continue
        start_time = time.time()
        label = task['label']
        task_meta = {
            'label': label, # for debug
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        # chatgpt: what happened in 1337? The year 1337 is best known as the beginning of the Hundred Years’ War
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        'results': results,
        'centered_results': centered_results,
        'core_metric': core_metric
    }
    return out

def main():
    parser = argparse.ArgumentParser()
    # TODO support HF model eval
    parser.add_argument('--max-per-task', type=int, default=-1)
    parser.add_argument('-i', '--source', type=str, default='base', help="Source of the model: base|sft|mid|rl")
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('--tasks-to-run', type=str, default=None, help="Tasks to run. Default = all tasks. Use | to split multiple tasks.")

    args = parser.parse_args()

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(args.source, device, phase='eval', model_tag=args.model_tag)
    model_name = f"base_model (step {meta['step']})" # just for logging
    model_slug = f"base_model_{meta['step']:06d}" # for the output csv file
    engine = Engine(model, tokenizer)

    tasks_to_run = None if args.tasks_to_run is None else args.tasks_to_run.split('|')

    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task, tasks_to_run=tasks_to_run)

    core_metric = None
    centered_results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, 'base_eval', f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out['results']
        centered_results = out['centered_results']
        core_metric = out['core_metric']
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        print0('='*80)
        print0(f"Model: {model_name}")
        print0('='*80)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            print0(f.read())

    
    get_report().log(section='Base model evaluation', data=[
        {
            "Model": model_name,
            "CORE metric": core_metric,
        },
        centered_results,
    ])

    compute_cleanup()

if __name__ == "__main__":
    main()
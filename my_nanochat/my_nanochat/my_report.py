import subprocess
import torch
import socket
import platform
import psutil
import os
import re
import datetime
import shutil

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except:
        return None

def get_git_info():
    info = {}
    info['commit'] = run_command('git rev-parse --short HEAD') or 'unknown'
    info['branch'] = run_command('git rev-parse --abbrev-ref HEAD') or 'unknown'

    status = run_command('git status --porcelain')
    info['dirty'] = bool(status) if status is not None else False

    info['message'] = run_command('git log -1 --pretty=%B') or ''
    info['message'] = info['message'].split('\n')[0][:80]

    return info

def get_gpu_info():
    if not torch.cuda.is_available():
        return {'available': False}

    num_devices = torch.cuda.device_count()
    info = {
        'available': True,
        'count': num_devices,
        'names': [],
        'memory_gb': []
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info['names'].append(props.name)
        info['memory_gb'].append(props.total_memory / (1024**3))

    info['cuda_version'] = torch.version.cuda or 'unknown'

    return info

def get_system_info():
    info = {}

    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info

def generate_header():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    # TODO cost info

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info['branch']}
- Commit: {git_info['commit']} {'(dirty)' if git_info['dirty'] else '(clean)'}
- Message: {git_info['message']}

### Hardware
- Platform: {sys_info['platform']}
- CPUs: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)
- Memory: {sys_info['memory_gb']:.1f} GB
"""

    if gpu_info.get('available'):
        gpu_names = ', '.join(set(gpu_info['names']))
        total_vram = sum(gpu_info['memory_gb'])
        header += f"""- GPUs: {gpu_info['count']}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info['cuda_version']}
"""
    else:
        header += '- GPUs: None available\n'

    # TODO cost info

    header += f"""
### Software
- Python: {sys_info['python_version']}
- PyTorch: {sys_info['torch_version']}

"""

    # TODO bloat info

    return header

def slugify(text):
    return text.lower().replace(' ', '-')

EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "midtraining.md",
    "chat-evaluation-mid.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]

# the metrics we're currently interested in
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]

def extract(section, keys):
    if not isinstance(keys, list):
        keys = [keys]
    out = {}
    for line in section.split('\n'):
        for key in keys:
            if key in line:
                out[key] = line.split(':')[1].strip()
    return out

def extract_timestamp(content, prefix):
    for line in content.split('\n'):
        if line.startswith(prefix):
            time_str = line.split(':', 1)[1].strip()
            return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    assert False

class Report:

    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir


    def log(self, section, data):
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    continue
                if isinstance(item, str):
                    f.write(item)
                else:
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v > 10_000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"Generating report to {report_file}")
        final_metrics = {} # the most important final metrics we'll add as table at the end
        start_time = None
        end_time = None
        with open(report_file, "w", encoding="utf-8") as out_file:
            header_file = os.path.join(report_dir, 'header.md')
            if os.path.exists(header_file):
                with open(header_file, 'r', encoding='utf-8') as in_file:
                    header_content = in_file.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, 'Run started:')
                    # TODO bloat data
            else:
                start_time = None
                print(f"Warning: {header_file} does not exist. Did you forget to run `nanochat reset`?")
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"Warning: {section_file} does not exist, skipping")
                    continue
                with open(section_file, 'r', encoding='utf-8') as in_file:
                    section = in_file.read()
                end_time = extract_timestamp(section, 'timestamp:')
                if file_name == 'base-model-evaluation.md':
                    final_metrics['base'] = extract(section, 'CORE')
                elif file_name == 'chat-evaluation-mid.md':
                    final_metrics['mid'] = extract(section, chat_metrics)
                elif file_name == 'chat-evaluation-sft.md':
                    final_metrics["sft"] = extract(section, chat_metrics)
                elif file_name == 'chat-evaluation-rl.md':
                    final_metrics['rl'] = extract(section, "GSM8K")

                out_file.write(section)
                out_file.write('\n')

            out_file.write('## Summary\n\n')
            # TODO write bloat data

            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())

            all_metrics = sorted(all_metrics, key=lambda x: (x != 'CORE', x == 'ChatCORE', x))
            stages = ['base', 'mid', 'sft', 'rl']
            metric_width = 15
            value_width = 8
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # Write table rows
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # Calculate and write total wall clock time
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        
        print(f"Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        report_file = os.path.join(self.report_dir, 'report.md')
        if os.path.exists(report_file):
            os.remove(report_file)

        header_file = os.path.join(self.report_dir, 'header.md')
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n--\n\n")
        print(f"Reset report and wrote header to {header_file}")

class DummyReport:
    def log(self, *args, **kwargs):
        pass
    def reset(self, *args, **kwargs):
        pass

def get_report():
    from my_nanochat.my_common import get_base_dir, get_dist_info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), 'report')
        return Report(report_dir)
    else:
        return DummyReport()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate or reset nanochat training reports.")
    parser.add_argument("command", nargs="?", default="generate", choices=["generate", "reset"], help="Operation to perform (default: generate)")
    args = parser.parse_args()
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()
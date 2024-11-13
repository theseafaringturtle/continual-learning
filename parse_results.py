import re
from typing import Tuple
import numpy as np
import scipy.stats as stats


# grep cifar100_part1.txt -e 'Accuracy of' -A11 -B15


def parse_context(s):
    values = []
    for line in s.split('\n'):
        tokens = re.split('Context (.+): ', line)
        if len(tokens) < 2:
            continue
        acc = tokens[-1]
        if acc:
            values.append(float(acc))
    return values


def collate(episodes):
    final = []
    # print(episodes[0])
    for i in range(len(episodes[0])):
        final_value = 0
        for results in episodes:
            final_value += results[i]
        final_value /= len(episodes)
        final.append(round(final_value * 100, 1))
    return final


def parse_file(file_name: str):
    with open(file_name, 'r') as f:
        contents = f.read()
    pattern = r'(PARAMETER STAMP \*.*?)(?=PARAMETER STAMP \*|$)'
    episodes = re.compile(pattern, flags=re.DOTALL).findall(contents)
    return episodes


def parse_episode(ep: str) -> Tuple[str, str]:
    replay = re.compile(r'--> replay:\s+(.+)').search(ep)
    if replay:
        replay = replay.group(1)
    memory = re.compile(r'--> memory buffer:\s+(.+)').search(ep)
    if memory:
        memory = memory.group(1)
    results = parse_context(ep)
    if not results:
        print(f"No results found for {replay} - {memory}, interrupted? Continuing")
        return None
    if not replay and not memory:
        return None
    if replay == 'buffer':
        num_samples = re.compile(r'b([0-9]+)random').match(memory).group(1)
        return 'ER', int(num_samples), results
    elif replay and 'A-GEM' in replay:
        num_samples = re.compile(r'b([0-9]+)random').match(memory).group(1)
        return 'A-GEM', int(num_samples), results
    elif replay and 'CFA' in replay:
        num_samples = re.compile(r'b([0-9]+)random').match(memory).group(1)
        return 'CFA', int(num_samples), results
    if not replay and memory:
        num_samples = re.compile(r'.*b([0-9]+)herding').match(memory).group(1)
        return 'iCarl', int(num_samples), results
    else:
        print(f"Not recognised: {replay}, {memory}")
        return None


def get_conf_inter(entries):
    if np.mean(entries) == 0:
        return np.nan
    conf_int_lower, conf_int_upper = stats.t.interval(0.95, len(entries) - 1, loc=np.mean(entries),
                                                      scale=stats.sem(entries))
    return np.mean(entries) - conf_int_lower # This is specular anyway so no upper # , np.mean(entries) - conf_int_upper


exp_list = []
episodes_1 = parse_file('cifar100_part1.txt')
episodes_2 = parse_file('cifar100_part2.txt')
episodes = episodes_1 + episodes_2
# episodes = parse_file('cifar100_gfsl.txt')
for ep in episodes:
    exp = parse_episode(ep)
    if exp:
        exp_list.append(exp)

for n in [10, 20, 50, 100]:
    for exp_name in 'CFA', 'A-GEM', 'ER', 'iCarl':
        print(f"Method: {exp_name}, Samples: {n}")
        # Filter experiments by name and samples
        exps = list(filter(lambda ep: ep[0] == exp_name and ep[1] == n, exp_list))
        # Convert to np for easier 2d slicing
        exp_results = np.array([exp[2] for exp in exps])
        num_rows = exp_results.shape[0]
        num_tasks = exp_results.shape[1]
        for task in range(num_tasks):
            task_entries = exp_results[:, task]
            mean = np.mean(task_entries)
            conf_inter = get_conf_inter(task_entries)
            print(f"Task {task}, Mean: {round(mean * 100, 2)}, Int: {round(conf_inter * 100, 2)}")

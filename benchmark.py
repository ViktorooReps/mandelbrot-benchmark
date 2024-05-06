import argparse
import re
import subprocess
from itertools import product
from random import shuffle

import cpuinfo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

REGEX = re.compile(r'Operations: \d+, Time: (\d+) ns, ns/op: (\d+\.\d+)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Mandelbrot benchmark and collect results.')

    parser.add_argument('--start', type=int, default=500, help='Start value for the range')
    parser.add_argument('--end', type=int, default=5500, help='End value for the range')
    parser.add_argument('--step', type=int, default=500, help='Step size for the range')
    parser.add_argument('--n_repeats', type=int, default=10, help='Number of repetitions per step')
    parser.add_argument('--exe_path', type=str, default='cmake-build-debug\\mandelbrot_benchmark.exe',
                        help='Path to the executable')
    parser.add_argument('--output_csv', type=str, default='results.csv', help='Output CSV file for results')
    parser.add_argument('--output_figure_ns', type=str, default='results_ns.png',
                        help='Output image file for the time results graph')
    parser.add_argument('--output_figure_ns_op', type=str, default='results_ns_op.png',
                        help='Output image file for the time per operation graph')

    args = parser.parse_args()

    results = pd.DataFrame(columns=['Impl', 'N', 'ns', 'ns/op'])

    all_ns = list(range(args.start, args.end, args.step)) * args.n_repeats
    versions = ['naive', 'optimized']
    version_ids = [0, 1]
    shuffle(all_ns)

    for version_id, n in tqdm(product(version_ids, all_ns), total=len(version_ids) * len(all_ns), smoothing=0):
        result = subprocess.run(
            [args.exe_path, str(version_id), '-2', '1', '-1.5', '1.5', str(n), str(n), '100'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        ns, ns_op = re.findall(REGEX, result.stdout)[0]
        results.loc[len(results.index)] = [versions[version_id], n, ns, ns_op]

    results.to_csv(args.output_csv, index=False)
    results = pd.read_csv(args.output_csv)

    cpu_info = cpuinfo.get_cpu_info()
    title = f"{cpu_info.get('brand_raw', 'Unknown brand')}, {cpu_info.get('hz_actual_friendly', 'Unknown frequency')}"

    plt.figure(figsize=(10, 5))
    sns.lineplot(results, x='N', y='ns/op', hue='Impl')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(args.output_figure_ns_op)
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.lineplot(results, x='N', y='ns', hue='Impl')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(args.output_figure_ns)
    plt.show()

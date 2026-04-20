import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from itertools import product


DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
DEFAULT_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5]
DEFAULT_SEEDS = [2021, 2022, 2023]


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Unsupported value encountered.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full PACS UDA sweeps for DSAN with source-val model selection."
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/speed-scratch/qiaoyu/speed-hpc/datasets/PACS/pacs_data/pacs_data",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--nepoch", type=int, default=50)
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output_dir", type=str, default="experiments/pacs_uda")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--nclass", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=30)
    parser.add_argument("--log_interval", type=int, default=10)
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def validate_split_directories(root_path, domains):
    missing = []
    for domain in domains:
        for suffix in ("train", "val", "test"):
            split_dir = os.path.join(root_path, f"{domain}_{suffix}")
            if not os.path.isdir(split_dir):
                missing.append(split_dir)

    if missing:
        missing_preview = "\n  - " + "\n  - ".join(missing[:12])
        if len(missing) > 12:
            missing_preview += f"\n  - ... and {len(missing) - 12} more"
        raise FileNotFoundError(
            "Missing PACS split folders required by this experiment runner."
            f"{missing_preview}\n"
            "Expected folders like <domain>_train / <domain>_val / <domain>_test under --root_path.\n"
            "If your dataset currently only has raw domain folders (e.g., 'art_painting', 'cartoon'), "
            "create splits first with split_pacs_uda.py."
        )


def all_tasks():
    return [(s, t) for s, t in product(DOMAINS, DOMAINS) if s != t]


def tail_text(path, n_lines=40):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])


def run_single(main_py, args, src_domain, tar_domain, weight, seed, output_dir):
    task = f"{src_domain}->{tar_domain}"
    run_name = f"src-{src_domain}_tar-{tar_domain}_w-{weight:g}_seed-{seed}"
    logs_dir = os.path.join(output_dir, "logs")
    run_results_dir = os.path.join(output_dir, "run_results")
    ensure_dir(logs_dir)
    ensure_dir(run_results_dir)

    log_path = os.path.join(logs_dir, f"{run_name}.log")
    result_json_path = os.path.join(run_results_dir, f"{run_name}.json")

    cmd = [
        args.python_bin,
        main_py,
        "--root_path",
        args.root_path,
        "--src_train",
        f"{src_domain}_train",
        "--src_val",
        f"{src_domain}_val",
        "--tar_train",
        f"{tar_domain}_train",
        "--tar_test",
        f"{tar_domain}_test",
        "--batch_size",
        str(args.batch_size),
        "--nepoch",
        str(args.nepoch),
        "--pretrained",
        str(args.pretrained).lower(),
        "--gpu",
        str(args.gpu),
        "--weight",
        str(weight),
        "--seed",
        str(seed),
        "--nclass",
        str(args.nclass),
        "--num_workers",
        str(args.num_workers),
        "--early_stop",
        str(args.early_stop),
        "--log_interval",
        str(args.log_interval),
        "--result_json",
        result_json_path,
    ]

    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        log_tail = tail_text(log_path)
        detail = f"\n--- Last log lines ---\n{log_tail}" if log_tail else ""
        raise RuntimeError(
            f"Training failed for {run_name} with return code {proc.returncode}. See {log_path}{detail}"
        )
    if not os.path.exists(result_json_path):
        raise FileNotFoundError(
            f"Expected result JSON not found: {result_json_path}. See {log_path}"
        )

    with open(result_json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    result["task"] = task
    result["source_domain"] = src_domain
    result["target_domain"] = tar_domain
    result["training_log_path"] = log_path
    result["result_json_path"] = result_json_path
    return result


def write_all_runs_csv(all_runs, output_csv):
    fieldnames = [
        "task",
        "source_domain",
        "target_domain",
        "src_train",
        "src_val",
        "tar_train",
        "tar_test",
        "seed",
        "weight",
        "best_source_val_acc",
        "target_test_acc",
        "best_checkpoint",
        "training_log_path",
        "result_json_path",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_runs:
            writer.writerow({k: row.get(k) for k in fieldnames})


def summarize_task_by_weight(task_runs, seeds):
    by_weight = {}
    for r in task_runs:
        by_weight.setdefault(float(r["weight"]), []).append(r)

    weight_stats = []
    for w, rows in by_weight.items():
        mean_src_val = sum(float(x["best_source_val_acc"]) for x in rows) / len(rows)
        weight_stats.append((w, mean_src_val, rows))

    # Higher mean source_val first; on tie, smaller weight wins.
    weight_stats.sort(key=lambda x: (-x[1], x[0]))
    selected_weight = weight_stats[0][0]

    weight_rows = []
    for weight, mean_source_val_acc, rows in weight_stats:
        rows_by_seed = {int(r["seed"]): r for r in rows}
        seed_target_accs = [float(rows_by_seed[s]["target_test_acc"]) for s in seeds]
        mean_target = statistics.mean(seed_target_accs)
        std_target = statistics.pstdev(seed_target_accs)

        summary = {
            "task": rows[0]["task"],
            "weight": weight,
            "is_selected_weight": weight == selected_weight,
            "mean_source_val_acc": mean_source_val_acc,
            "mean_target_test_acc": mean_target,
            "std_target_test_acc": std_target,
        }
        for idx, seed in enumerate(seeds, start=1):
            summary[f"seed{idx}_target_test_acc"] = rows_by_seed[seed]["target_test_acc"]
            summary[f"seed{idx}"] = seed

        weight_rows.append(summary)

    return weight_rows


def write_final_summary_csv(final_rows, out_csv):
    fieldnames = [
        "task",
        "weight",
        "is_selected_weight",
        "mean_source_val_acc",
        "seed1_target_test_acc",
        "seed2_target_test_acc",
        "seed3_target_test_acc",
        "mean_target_test_acc",
        "std_target_test_acc",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_text_summary(final_rows, out_txt):
    lines = []
    lines.append("PACS UDA Final Summary (all explored weights)")
    lines.append("===========================================")
    grouped = {}
    for row in final_rows:
        grouped.setdefault(row['task'], []).append(row)

    for task in sorted(grouped.keys()):
        lines.append(f"\n{task}")
        lines.append("-" * len(task))
        for row in grouped[task]:
            selection_tag = " [selected]" if row["is_selected_weight"] else ""
            lines.append(
                f"weight={row['weight']}{selection_tag}, "
                f"mean_source_val_acc={row['mean_source_val_acc']:.4f}, "
                f"target_test_mean={row['mean_target_test_acc']:.4f}, "
                f"target_test_std={row['std_target_test_acc']:.4f}"
            )
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    validate_split_directories(args.root_path, DOMAINS)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(script_dir, "main.py")
    if not os.path.exists(main_py):
        raise FileNotFoundError(f"Could not find main.py at {main_py}")

    tasks = all_tasks()
    all_runs = []

    total = len(tasks) * len(args.weights) * len(args.seeds)
    i = 0
    for src_domain, tar_domain in tasks:
        for weight in args.weights:
            for seed in args.seeds:
                i += 1
                print(
                    f"[{i}/{total}] Running {src_domain}->{tar_domain}, weight={weight}, seed={seed}",
                    flush=True,
                )
                result = run_single(
                    main_py=main_py,
                    args=args,
                    src_domain=src_domain,
                    tar_domain=tar_domain,
                    weight=weight,
                    seed=seed,
                    output_dir=args.output_dir,
                )
                all_runs.append(result)

    all_runs_json = os.path.join(args.output_dir, "all_runs.json")
    with open(all_runs_json, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    all_runs_csv = os.path.join(args.output_dir, "all_runs.csv")
    write_all_runs_csv(all_runs, all_runs_csv)

    final_rows = []
    for src_domain, tar_domain in tasks:
        task = f"{src_domain}->{tar_domain}"
        task_runs = [r for r in all_runs if r["task"] == task]
        final_rows.extend(summarize_task_by_weight(task_runs, args.seeds))

    final_summary_csv = os.path.join(args.output_dir, "final_summary.csv")
    write_final_summary_csv(final_rows, final_summary_csv)

    final_summary_txt = os.path.join(args.output_dir, "final_summary.txt")
    write_text_summary(final_rows, final_summary_txt)

    print("\nFinished PACS UDA experiment sweep.")
    print(f"Raw JSON: {all_runs_json}")
    print(f"All-runs CSV: {all_runs_csv}")
    print(f"Final summary CSV: {final_summary_csv}")
    print(f"Final summary text: {final_summary_txt}")


if __name__ == "__main__":
    main()

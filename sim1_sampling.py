import subprocess
import csv
import os.path as osp
import os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import time
import numpy as np
from module.Metrics import evaluate_rendered_images
import trimesh



@hydra.main(version_base="1.1", config_path="config", config_name="config")
def sim(cfg: DictConfig):
    sen_cfg = cfg.sensing
    full_smp, min_smp, smp_interval = sen_cfg.max_samples,sen_cfg.min_samples,sen_cfg.smp_interval
    num_samples_list = list(range(min_smp,full_smp+1,smp_interval))
    obj = cfg.obj_model

    results = []
    original_mesh_path = osp.join("obj_models", obj, "nontextured.stl")
    original_mesh = trimesh.load(original_mesh_path)

    reconstructed_mesh_path = osp.join("data", "sim", obj, "full_coverage", "recon_stl")

    # Generate tactile data: run sensing.py with Hydra override for max_samples
    print("(1) Running 1_1_sensing.py...")
    proc_main = subprocess.run(
        ["python", "1_1_sensing.py", f"sensing.max_samples={full_smp}"],
        capture_output=True,
        text=True
    )
    if proc_main.returncode != 0:
        print(f"[ERROR] main.py execution failed (max_samples={full_smp}):")
        print(proc_main.stderr)
    else:
        print(proc_main.stdout)

    # Wrap the experiments loop with tqdm for better progress visualization
    for num_samples in tqdm(num_samples_list, desc="Running experiments", unit="experiment"):
        print(f"\n\n\033[1;33mStarting experiment with num_samples:{num_samples}\033[0m")
        start_time = time.time()


        # Perform reconstruction: run 2_reconstruction.py
        print("(2) Running 2_reconstruction.py...")
        proc_recon = subprocess.run(
            ["python", "2_reconstruction.py", f"sensing.num_samples={num_samples}"],
            capture_output=True,
            text=True
        )
        if proc_recon.returncode != 0:
            print(f"[ERROR] 2_reconstruction.py execution failed num_samples={num_samples}):")
            print(proc_recon.stderr)
            continue
        else:
            print(proc_recon.stdout)

        elapsed_time = time.time() - start_time

        print("(3) Evaluating reconstruction performance...")
        sample_mesh =osp.join(reconstructed_mesh_path,f"{num_samples}_re_mesh.stl")
        reconstructed_mesh = trimesh.load(sample_mesh)
        metrics = evaluate_rendered_images(original_mesh, reconstructed_mesh, resolution=(256, 256))
        # print(f"[RESULT] Metrics for num_samples={num_samples}: {metrics}")
        metrics["num_samples"] = num_samples
        results.append(metrics)

        print(f" [DONE] Experiment with max_samples={num_samples} completed. elapsed_time={elapsed_time}\n")

    print(f"\n [INFO] Saving metrics to CSV file...")
    results_dir = "sim_result"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f" [INFO] Created directory: {results_dir}")

    existing_files = [f for f in os.listdir(results_dir) if
                      f.startswith("reconstruction_metrics") and f.endswith(".csv")]
    index = len(existing_files) + 1
    csv_filename = osp.join(results_dir, f"reconstruction_metrics_{index:03d}.csv")


    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = ["PSNR","SSIM","MSE","num_samples"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n [DONE] All experiments completed. Metrics saved in '{csv_filename}'\n.")

    # for root, dirs, files in os.walk(os.getcwd()):
    #     for file in files:
    #         if file.endswith(".log"):
    #             file_path = os.path.join(root, file)
    #             try:
    #                 os.remove(file_path)
    #                 print(f" [INFO] Removed log file: {file_path}")
    #             except Exception as e:
    #                 print(f"[WARNING] Failed to remove log file {file_path}: {e}")


if __name__ == "__main__":
    sim()

import csv
import os.path as osp
import os
import hydra
from omegaconf import DictConfig
import importlib as imp

from module.result.Metrics import compute_surface_metrics

from sim_main.step1_sensing import whole_sensing



def snr_sweep(cfg):
    sim_cfg = cfg.sim
    obj = sim_cfg.obj_model
    comm_cfg = sim_cfg.comm
    ch_type = comm_cfg.channel_type

    module = imp.import_module('sim_main.step2_recon_by_scheme')
    scheme = getattr(module, cfg.sim.scheme)

    min_snr,max_snr,snr_interval = comm_cfg.min_snr,comm_cfg.max_snr,comm_cfg.snr_interval
    snr_list = list(range(min_snr, max_snr+1, snr_interval))

    original_mesh_path = osp.join("obj_models", obj, "nontextured.stl")

    all_results = []
    for snr in snr_list:
        print(f"\033[1;33m\nSNR sweep experiments {ch_type} (SNR={snr}dB)\033[0m")

        # Perform reconstruction
        print("(2) Reconstructing mesh...")
        comm_cfg.snr = snr
        scheme_class = scheme(cfg)
        scheme_class.run()
        pred_k_tuple = scheme_class.map_reconstruction()

        print("(3) Evaluating reconstruction performance...")

        sample_mesh_path =osp.join(sim_cfg.snr_sweep.stl_dir,sim_cfg.stl_filename)

        result = {'num_samples': cfg.sensing.num_samples,
                  'channel': scheme_class.channel.channel_type,
                  'snr': snr}
        metric = compute_surface_metrics(original_mesh_path, sample_mesh_path, pred_k_tuple, cfg)
        result.update(metric)
        for key, value in result.items():
            # key를 대문자로 변환, 소수점 4자리까지 출력
            print(f"\033[1;36m * {key.upper()}: {value}\033[0m")

        all_results.append(result)


        print(f" [DONE] Experiment with SNR={snr}dB completed.")

    return all_results


def num_samples_sweep(cfg):
    sim_cfg = cfg.sim
    obj = sim_cfg.obj_model
    sen_cfg = sim_cfg.sampling

    module = imp.import_module('sim_main.step2_recon_by_scheme')
    scheme = getattr(module, cfg.sim.scheme)

    min_samp, max_samp, samp_interval = sen_cfg.min_samples, sen_cfg.max_samples, sen_cfg.smp_interval
    num_samples_list = list(range(min_samp, max_samp + 1, samp_interval))

    original_mesh_path = osp.join("obj_models", obj, "nontextured.stl")

    all_results = []
    scheme_class = scheme(cfg)
    scheme_class.run()
    for num_samples in num_samples_list:
        print(f"\033[1;33m\nNum samples sweep experiments (num_samples:{num_samples})\033[0m")

        # Perform reconstruction
        scheme_class.cfg.sensing.num_samples = num_samples
        print("(2) Reconstructing mesh...")
        # pred_k_tuple = reconstruction(cfg, image_dir, mask_dir, base_dir)
        pred_k_tuple = scheme_class.map_reconstruction()
        print("(3) Evaluating reconstruction performance...")

        sample_mesh_path =osp.join(sim_cfg.smp_sweep.stl_dir,sim_cfg.stl_filename)

        result = {'num_samples': num_samples,
                  'channel': scheme_class.channel.channel_type,
                  'snr': scheme_class.channel.snr}
        metric = compute_surface_metrics(original_mesh_path, sample_mesh_path, pred_k_tuple, cfg)
        result.update(metric)
        for key, value in result.items():
            # key를 대문자로 변환, 소수점 4자리까지 출력
            print(f"\033[1;36m * {key.upper()}: {value}\033[0m")

        all_results.append(result)

        print(f" [DONE] Experiment with max_samples={num_samples} completed.")

    return all_results

@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def sim(cfg: DictConfig):
    sim_cfg = cfg.sim
    init_num_samples = sim_cfg.sampling.max_samples

    print(f"\033[1;35mStarting experiment with {sim_cfg.scheme}\033[0m")

    for it in range(0,3):
        print(f"\033[1;37mIter:{it+1}\033[0m")

        #================Step1================
        if sim_cfg.sampling.require:
            # Generate tactile data: run sensing.py with Hydra override for max_samples
            print("(1) Sensing process...")
            cfg.sensing.num_samples = init_num_samples
            whole_sensing(cfg)
        else:
            print("(1) Skipping step1_sensing.py...")

        #================Step2================
        cfg.sensing.num_samples = init_num_samples
        if sim_cfg.sweep == "snr":
            results = snr_sweep(cfg)
            results_dir = sim_cfg.snr_sweep.csv_dir
            os.makedirs(results_dir,exist_ok=True)
        elif sim_cfg.sweep == "num_samples":
            results = num_samples_sweep(cfg)
            results_dir = sim_cfg.smp_sweep.csv_dir
            os.makedirs(results_dir, exist_ok=True)
        else:
            raise ValueError("Invalid sweep type")

        print(f"\n [INFO] Saving metrics to CSV file...")
        existing_files = [f for f in os.listdir(results_dir) if
                          f.startswith("Scheme") and f.endswith(".csv")]
        index = len(existing_files) + 1
        csv_filename = osp.join(results_dir, f"{sim_cfg.scheme}_results_{index:03d}.csv")


        with open(csv_filename, "w", newline="") as csvfile:
            fieldnames = ["num_samples","channel","snr", "outlier_count", "stiffness_accuracy"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f" [DONE] All experiments completed. Metrics saved in '{csv_filename}'.\n")

if __name__ == "__main__":
    sim()

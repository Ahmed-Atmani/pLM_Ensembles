import pickle as pkl
import h5py as h5
import torch
import os
import time
import gc
from tqdm import tqdm
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from plm_models.model_loader import load_plms, PLMS_AVAILABLE, PLMS_classes

@contextmanager
def silence_terminal():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

def process_single_plm(plm_name, plm_class, dataset_dict, output_dir):
    file_path = os.path.join(output_dir, f"{plm_name}_embeddings.h5")
    start_time = time.perf_counter()

    # Limit seq length to 510 for AntiBERTy (all others will use 1022 long sequences)
    if plm_name == "AntiBERTy":
        sequence_selector = "sequence_cropped_510"
        crop_offset_selector = "crop_offset_510"
    else:
        sequence_selector = "sequence_cropped"
        crop_offset_selector = "crop_offset"

    # Calculate total sequences for the progress bar
    total_seqs = sum(len(subset_data) for subset_data in dataset_dict.values())

    # Load model silently
    with silence_terminal():
        if "ESM-2" in plm_name:
            plm = plm_class()
            plm.model.cuda().half()
        else:
            plm = plm_class()

    with h5.File(file_path, "w") as hf:
        plm_group = hf.create_group(plm_name)
        
        pbar = tqdm(total=total_seqs, desc=f"    {plm_name}", unit="seq", leave=False)
        
        with torch.no_grad():
            for subset, data in dataset_dict.items():
                subset_group = plm_group.create_group(subset)
                
                for seq_id, seq_data in data.items():
                    original_seq = seq_data[sequence_selector]

                    # Parse mutation index + add offset to cropped sequences                    
                    var_idx = seq_data["mut_idx"]
                    var_new_aa = seq_data["new_aa"]
                    crop_offset = seq_data[crop_offset_selector]
                    var_idx -= crop_offset

                    mutated_seq = original_seq[:var_idx] + var_new_aa + original_seq[var_idx + 1:] 
                



                    # Inside process_single_plm, right before plm(original_seq)
                    if plm_name == "AntiBERTy" and len(original_seq) > 510:
                        # If we got here, the data loading logic is flawed.
                        # We force a fix so the model doesn't crash, but we log the warning.
                        print(f"!!! WARNING: ID {seq_id} was {len(original_seq)} AAs. Force-cropping to 510.")
                        original_seq = original_seq[:510]
                        # Re-calculate mutated_seq based on the forced crop if necessary




                    embedding = plm(original_seq)
                    mut_embedding = plm(mutated_seq)
                    
                    subset_group.create_dataset(f"{seq_id}_original", data=embedding.cpu())
                    subset_group.create_dataset(f"{seq_id}_mutated", data=mut_embedding.cpu())
                        
                    pbar.update(1)
        pbar.close()

    end_time = time.perf_counter()
    runtime = end_time - start_time

    with h5.File(file_path, "a") as hf:
        hf[plm_name].attrs["runtime_seconds"] = runtime

    # Clean VRAM/RAM
    del plm
    torch.cuda.empty_cache()
    gc.collect()
    
    return runtime

def generate_embeddings():
    data_dir = "datasets/driver_mutation/"
    output_dir = "datasets/driver_mutation/embeddings/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_subsets = ["DRGN", "ALL", "DEL", "NEUT"]
    dataset_dict = {}

    for name in dataset_subsets:
        path = os.path.join(data_dir, f"{name}.pkl")
        with open(path, "rb") as file:
            dataset_dict[name] = pkl.load(file)

    plm_names = list(PLMS_AVAILABLE)
    # plm_names = ["ProtT5", "ESM1b", "CARP", "Ankh3", "ESM1v", "ESM2", "ESMC", "ESMDance", "SeqDance", "AntiBERTy"]
    plm_to_process = [(name, PLMS_classes[name]) for name in plm_names]

    print(f"=> STARTING GENERATION FOR {len(plm_to_process)} MODELS")
    
    # Outer loop for high-level progress
    for plm_name, plm_class in plm_to_process:
        print(f"Current PLM: {plm_name}")
        runtime = process_single_plm(plm_name, plm_class, dataset_dict, output_dir)
        print(f"Finished {plm_name} in {runtime:.2f}s")

    print("=> PIPELINE COMPLETE")

if __name__ == "__main__":
    generate_embeddings()
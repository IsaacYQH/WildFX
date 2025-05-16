import os
import json
import random
import numpy as np
import soundfile as sf
import tempfile
import multiprocessing
import shutil
import torch
import torch.nn as nn
import torchaudio.transforms as T
import reapy
from utils.reaper_utils import create_safe_instance_name, batch_render_fx, fx_get_metadata
from utils.global_variables import ALLOWED_FX_TYPES, PLUGIN_PRESETS_DIR
from tqdm import tqdm
from sklearn.cluster import KMeans
import argparse
import csv
from itertools import islice

def choose_random_plugin_parameters(valid_params: dict) -> dict:
    """
    Randomly selects plugin parameters from a dictionary of valid parameter values.
    Args:
        valid_params (dict): A dictionary where keys are parameter names (str) and 
                                values are lists of valid values for each parameter.
    Returns:
        dict: A dictionary containing randomly selected parameter values. 
                Parameters with "bypass" in their name are always set to 0. 
                Parameters with "buffer size" or "sample rate" in their name are ignored.
    """
    return {name: random.choice(valid_values) if valid_values else None 
            for name, valid_values in valid_params.items()}
    
# Helper function for parallel parameter generation
# This makes it easier to pass arguments to the pool worker
def generate_params_task(args):
    index, valid_params_arg = args
    # Ensure choose_random_plugin_parameters is deterministic or handles
    # randomness correctly across processes if needed (e.g., seeding)
    params_choice = choose_random_plugin_parameters(valid_params_arg)
    return params_choice

# Helper function for parallel audio loading ---
def load_process_audio_task(args):
    """Loads, normalizes, and flattens a single audio file."""
    index, output_path, target_sample_rate = args
    try:
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            return None # Skip non-existent or tiny files

        audio_output, sr = sf.read(output_path, always_2d=True)
        if sr != target_sample_rate:
            # print(f"Warning: Sample rate mismatch for {output_path}. Expected {target_sample_rate}, got {sr}. Skipping.")
            return None # Skip if sample rate doesn't match
        audio_output = audio_output.T # Transpose to (channels, samples)

        # Peak normalize
        peak = np.max(np.abs(audio_output))
        if peak > 1e-8:
            audio_output_norm = audio_output / peak
        else:
            audio_output_norm = np.zeros_like(audio_output) # Handle silence

        # Flatten stereo audio (take left channel or average for MFCC)
        if audio_output_norm.ndim > 1 and audio_output_norm.shape[0] > 1:
             audio_flat = np.mean(audio_output_norm, axis=0) # Use average across channels
        else:
             audio_flat = np.reshape(audio_output_norm, (-1))

        return (index, audio_flat) # Return original index and flattened audio
    except Exception as e:
        print(f"Error loading/processing {output_path} in worker: {e}")
        return None

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate audio effect plugin presets using parameter clustering")
    
    # Main configuration
    parser.add_argument("--input-audio", type=str, default=None,
                        help="Path to the audio sample used for cluster validation")
    parser.add_argument("--no-cluster-validation", type=bool, default=False,
                        help="Disenable cluster validation")
    parser.add_argument("--output-dir", type=str, default=f"{PLUGIN_PRESETS_DIR}",
                        help="Directory to store generated plugin presets")
    parser.add_argument("--detected-plugin-list", type=str, default="utils/reaper_plugins.csv",
                        help="CSV file genereate by 'plugin_get_list.lua' script containing detected plugins in REAPER")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Plugin selection
    # Create a required mutually exclusive group
    plugin_selection = parser.add_mutually_exclusive_group(required=True)
    plugin_selection.add_argument("--plugin-list", type=str, default=None,
                        help="Path to a CSV file containing plugin names and types (format: plugin_name,plugin_type)")
    plugin_selection.add_argument("--plugin-name", type=str, nargs=2, action="append", metavar=('NAME', 'TYPE'),
                        help="Specify a plugin to process with its type (e.g., --plugin-name 'VST3: MyPlugin' eq)")
    plugin_selection.add_argument("--use-reduced-set", action="store_true",
                        help="Use a reduced set of example plugins")
    plugin_selection.add_argument("--use-full-set", action="store_true",
                        help="Use the full set of example plugins")
    
    # Processing options
    parser.add_argument("--num-presets", type=int, default=None,
                        help="Number of presets to generate for each plugin")
    parser.add_argument("--num-generations", type=int, default=None,
                        help="Number of generations to perform for each plugin")
    parser.add_argument("--cpu-cores", type=int, default=multiprocessing.cpu_count()-1,
                        help="Number of CPU cores to use for parallel processing")
    parser.add_argument("--cuda-device", type=int, default=0,
                        help="CUDA device ID to use (default: use CUDA_VISIBLE_DEVICES env var)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for MFCC feature extraction")
    parser.add_argument("--validate_generation", action="store_true", default=True,
                        help="Validate generated presets by clustering MFCC features")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip plugins that already have generated presets")
    
    args = parser.parse_args()
    # Set random seed
    random.seed(args.seed)
    
    # Set CUDA device if specified
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    
    # Initialize plugins_to_process as a dictionary
    plugins_to_process = {}
    
    # Read plugins from file if provided
    if args.plugin_list:
        try:
            with open(args.plugin_list, 'r') as f:
                # Try to read as CSV first
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        plugin_name = row[0].strip()
                        plugin_type = row[1].strip().lower()
                        if plugin_name and plugin_type:
                            if plugin_type in ALLOWED_FX_TYPES:
                                plugins_to_process[plugin_name] = plugin_type
                            else:
                                print(f"Warning: Plugin type '{plugin_type}' for '{plugin_name}' not in ALLOWED_FX_TYPES. Skipping.")
        except Exception as e:
            print(f"Error reading plugin list file: {e}")
    
    # Add plugins specified via command line
    if args.plugin_name:
        for plugin_name, plugin_type in args.plugin_name:
            plugin_type = plugin_type.lower()
            if plugin_type in ALLOWED_FX_TYPES:
                plugins_to_process[plugin_name] = plugin_type
            else:
                print(f"Warning: Plugin type '{plugin_type}' for '{plugin_name}' not in ALLOWED_FX_TYPES. Skipping.")
        
    # Add full set of plugins if requested
    if args.use_full_set:
        full_plugins_to_process = {
            "VST3: Graphic Equalizer x16 Stereo (LSP VST3)": "eq",
            "JS: 3-Band Splitter": "splitter",
            "VST3: Gate (SocaLabs)": "gate",
            "VST3: Expander (SocaLabs)": "expander",
            "VST3: ZamCompX2 (Damien Zammit)": "compressor",
            "LV2: Calf Saturator (Calf Studio Gear)": "saturation",
            "JS: Soft Clipper/Limiter": "clipper",
            "VST3: FlyingChorus (superflyDSP)": "chorus",
            "JS: Flanger": "flanger",
            "VST3: FlyingPhaser (superflyDSP)": "phaser",
            "VST3: FlyingTremolo (superflyDSP)": "tremolo",
            "VST: Vibrato (airwindows)": "vibrato",
            "VST3: FlyingDelay (superflyDSP)": "delay",
            "VST3: FlyingReverb (superflyDSP)": "reverb",
            "VST3: Ping Pong Pan (DISTRHO)": "spatial",
            "VST3: Pisstortion (unplugred)": "distortion"
        }
        plugins_to_process.update(full_plugins_to_process)
    
    # Add reduced set of plugins if requested
    if args.use_reduced_set:
        reduced_plugins_to_process = {
            "VST3: 3 Band EQ (DISTRHO)": "eq",
            "VST3: ZamCompX2 (Damien Zammit)": "compressor",
            "VST3: Samurai Delay (discoDSP)": "delay",
            "VST3: Schroeder (discoDSP)": "reverb",
            "JS: 3-Band Splitter": "splitter",
        }
        plugins_to_process.update(reduced_plugins_to_process)

    # Validate that all plugins exist in the detected plugin list
    if os.path.exists(args.detected_plugin_list):
        available_plugins = set()
        try:
            with open(args.detected_plugin_list, 'r') as f:
                # Skip comment line if present
                first_line = f.readline()
                if first_line.startswith("//"):
                    # It's a comment, continue reading
                    pass
                else:
                    # Not a comment, rewind to start
                    f.seek(0)
                
                # Read the CSV file
                reader = csv.reader(f)
                for row in islice(reader, 1, None):  # Skip header row if there is one
                    if len(row) >= 2:  # Ensure row has at least 2 columns (Index, Name)
                        plugin_name = row[1].strip()
                        if plugin_name:
                            available_plugins.add(plugin_name)
            
            # Check plugins_to_process against available_plugins
            unavailable_plugins = []
            for plugin_name in list(plugins_to_process.keys()):
                if plugin_name not in available_plugins:
                    unavailable_plugins.append(plugin_name)
                    del plugins_to_process[plugin_name]
            
            if unavailable_plugins:
                print(f"\nWarning: The following plugins were not found in {args.detected_plugin_list}:")
                for plugin in unavailable_plugins:
                    print(f"  - {plugin}")
                print("These plugins will be skipped.")
        
        except Exception as e:
            print(f"Error reading detected plugin list {args.detected_plugin_list}: {e}")
            print("Continuing without validating plugins...")
    else:
        print(f"Warning: Detected plugin list file {args.detected_plugin_list} not found.")
        print("Continuing without validating plugins...")

    # Check if we have any plugins to process
    if not plugins_to_process:
        parser.error("No plugins specified. Use --plugin-name, --plugin-list, --use-reduced-set, or --use-full-set")
    
    print(f"\nWill process {len(plugins_to_process)} plugins")
        
    # Paths
    plugin_preset_dir = args.output_dir
    os.makedirs(plugin_preset_dir, exist_ok=True)
    audio_input_path = args.input_audio
    tmp_output_dir = tempfile.mkdtemp()
    # num_generations = 100 changed to num_generations = 2**num_valid_params * 10
    # num_presets = 10 changed to num_presets = 2**num_valid_params
    # random_sampling = args.random_sampling
    cpu = args.cpu_cores
    
    # Load sample input audio
    try:
        audio_input, sample_rate = sf.read(audio_input_path, always_2d=True)
    except FileNotFoundError:
        print(f"Audio input file not found: {audio_input_path}")
        raise
    audio_input = audio_input.T
    
    # Set up device (GPU if available)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    print(f"Using device: {device}")
    batch_size = args.batch_size
    print(f"Using effective batch size: {batch_size}")
    
    # Create MFCC transform once (reuse for efficiency)
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=20,  # Number of MFCC coefficients (adjust as needed)
        melkwargs={"n_fft": 2048, "n_mels": 128, "hop_length": 512}
    ).to(device)

    # Wrap model with DataParallel if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    if use_cuda and num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs for MFCC transform.")
        mfcc_transform = nn.DataParallel(mfcc_transform, device_ids=list(range(num_gpus)))
    
    # Connnect to current project (need to by empty)
    project = reapy.Project()
    
    # plugin_metadata_list = []
    for plugin_name in plugins_to_process:
        print(f"\n{'='*10} Processing plugin: {plugin_name} {'='*10}")
        # Create a better safe instance name for folders and files
        safe_instance_name = create_safe_instance_name(plugin_name)
        
        # Check if we already have generated presets for this plugin
        preset_filepath = os.path.join(plugin_preset_dir, f"{safe_instance_name}.json")
        if os.path.exists(preset_filepath):
            print(f"Skipping {plugin_name}...")
            continue
        
        # Get Plugin Metadata (Sequential REAPER interaction)
        try:
            print("Getting plugin metadata from REAPER...")
            # Ensure reapy is connected and project is accessible
            plugin_metadata = fx_get_metadata(project, plugin_name, plugins_to_process[plugin_name])
            # plugin_metadata['fx_type'] = plugins_to_process[plugin_name]
            valid_params = plugin_metadata['valid_params']
            n_inputs = plugin_metadata['n_inputs']
            n_outputs = plugin_metadata['n_outputs']
            print(f"Obtained plugin_metadata: {len(valid_params)} parameters, {n_inputs} input channels and {n_outputs}.")
        except Exception as e:
            print(f"Error getting plugin_metadata for {plugin_name} via REAPER: {e}")
            print("Skipping this plugin.")
            continue # Skip to the next plugin
        
        # make dual mono if necessary, vice versa
        if audio_input.shape[0] == 1 and min(n_inputs, n_outputs) > 1: # for some plugins with sidechain functionality, n_inputs == 1 does not mean mono
            audio_input = np.concatenate((audio_input, audio_input), axis=0)
        if audio_input.shape[0] == 2 and min(n_inputs, n_outputs) == 1:
            audio_input = audio_input[0:1, :]
        
        # Determine number of generations based on *interested* valid params
        num_valid_params = sum(1 for _, values in valid_params.items() if values)
        if num_valid_params == 0:
            print(f"Skipping {plugin_name} - No adjustable parameters found after filtering.")
            continue
        
        # Determine number of presets
        if args.num_presets:
            num_presets = args.num_presets
        else:
            # Calculate based on params, but cap it at a reasonable maximum
            MAX_PRESETS_FROM_PARAMS = 1024 # Example cap
            calculated_presets = 2 ** num_valid_params
            num_presets = min(calculated_presets, MAX_PRESETS_FROM_PARAMS)
            if calculated_presets > MAX_PRESETS_FROM_PARAMS:
                print(f"Warning: Calculated presets ({calculated_presets}) exceeds cap ({MAX_PRESETS_FROM_PARAMS}). Using {MAX_PRESETS_FROM_PARAMS}.")

        # Determine number of generations
        if args.num_generations:
            num_generations = args.num_generations
        else:
            # Ensure num_generations is significantly larger than num_presets
            num_generations = max(num_presets * 5, 100) # Ensure at least 100 generations
        print(f"Targeting {num_generations} parameter sets to find {num_presets} presets for {plugin_name} ({num_valid_params} adjustable params)...")

        # Check if num_presets exceeds num_generations after calculation
        if num_presets > num_generations:
            print(f"Warning: Number of presets ({num_presets}) exceeds number of generations ({num_generations}). Reducing num_presets to {num_generations}.")
            num_presets = num_generations # Adjust num_presets if necessary
        
        # Parallel Parameter Generation ---
        preset_parameters = []
        generation_tasks = [(i, valid_params) for i in range(num_generations)]

        print(f"Generating parameters using {cpu} cores...")
        try:
            with multiprocessing.Pool(processes=cpu) as pool:
                # Use imap_unordered for potential speedup and tqdm for progress
                preset_parameters = list(tqdm(pool.imap_unordered(generate_params_task, generation_tasks),
                                              total=num_presets if args.no_cluster_validation else num_generations, 
                                              desc="Generating Params"))
            print(f"Successfully generated {len(preset_parameters)} parameter sets.")
        except Exception as e:
            print(f"Error during parallel parameter generation: {e}")
            continue # Skip to next plugin

        if not preset_parameters:
            print("No parameters were generated. Skipping rendering and clustering.")
            continue
        
        
        if not args.no_cluster_validation:
            FxChain_list = [[{"fx_name": plugin_name, "params": list(params.values())}] for params in preset_parameters if params is not None]
            tmp_output_paths = [os.path.join(tmp_output_dir, f"{safe_instance_name}_{i}.wav") for i in range(len(FxChain_list))]
            batch_render_fx(project, [audio_input_path]*len(FxChain_list), tmp_output_paths, FxChain_list, batch_size = cpu)
            
            # Parallel Load Rendered Audio & Normalize ---
            print("Loading and normalizing rendered audio from temporary directory (in parallel)...")
            load_tasks = [(i, path, sample_rate) for i, path in enumerate(tmp_output_paths)]
            results = []

            try:
                with multiprocessing.Pool(processes=cpu) as pool: # Reuse cpu count or define separate for loading
                    results = list(tqdm(pool.imap_unordered(load_process_audio_task, load_tasks),
                                        total=len(load_tasks), desc="Loading Audio"))
            except Exception as e:
                print(f"Error during parallel audio loading: {e}")
                # Decide how to handle this, maybe skip clustering for this plugin
                continue

            # Process results
            flattened_audio_for_mfcc = []
            successfully_loaded_indices = []

            for result in results:
                if result is not None:
                    original_index, audio_flat = result
                    successfully_loaded_indices.append(original_index)
                    flattened_audio_for_mfcc.append(audio_flat)
            if not flattened_audio_for_mfcc:
                print(f"Error: No audio files were successfully rendered or loaded for {plugin_name}. Skipping clustering.")
                continue # Skip to next plugin

            print(f"Successfully loaded {len(flattened_audio_for_mfcc)} rendered audio files for feature extraction.")
            # Filter preset_parameters to only include those corresponding to loaded audio
            filtered_preset_parameters = [preset_parameters[i] for i in successfully_loaded_indices]

            # Batch Feature Extraction ---
            print("Extracting MFCC features in batch...")
            features_list = []

            try:
                # Stack flattened audio into a single numpy array
                audio_batch_np = np.stack(flattened_audio_for_mfcc, axis=0)
                num_samples = audio_batch_np.shape[0]

                for i in tqdm(range(0, num_samples, batch_size), desc="MFCC Mini-Batches"):
                    # Get mini-batch
                    mini_batch_np = audio_batch_np[i:i + batch_size]
                    # Convert to PyTorch tensor and send to the primary device (cuda:0 if using CUDA)
                    mini_batch_tensor = torch.tensor(mini_batch_np, dtype=torch.float32).to(device)

                    # Compute MFCCs for the mini-batch
                    # DataParallel handles splitting the batch across GPUs
                    with torch.no_grad(): # Ensure no gradients are computed
                        mfccs_mini_batch = mfcc_transform(mini_batch_tensor) # Output shape (batch, n_mfcc, time_frames)

                    # Reshape features for this mini-batch
                    # Note: Output from DataParallel might be gathered on the primary GPU (device 0)
                    features_mini_batch = mfccs_mini_batch.reshape(mfccs_mini_batch.size(0), -1)

                    # Move features to CPU and append to list
                    features_list.append(features_mini_batch.cpu().numpy())

                    # Clear GPU cache periodically if needed (optional, might not be necessary with DataParallel)
                    if use_cuda:
                        torch.cuda.empty_cache()


                # Concatenate features from all mini-batches
                features = np.concatenate(features_list, axis=0) # This is now your feature matrix features
                print(f"Successfully extracted features. Shape: {features.shape}")

            except Exception as e:
                print(f"Error during batch MFCC extraction: {e}")
                # Handle error, maybe fall back to sequential or skip clustering
                if use_cuda:
                    torch.cuda.empty_cache() # Attempt to clear memory on error
                continue

            if features.shape[0] == 0: # Check if feature extraction yielded results
                print(f"Error: No features extracted for {plugin_name}. Skipping clustering.")
                continue

            # --- Clustering and Saving ---
            print(f"Feature matrix shape: {features.shape}")
            
            # Perform clustering to discover presets
            print("Clustering...")
            # Use the determined num_presets value
            if features.shape[0] < num_presets:
                print(f"Warning: Number of generated features ({features.shape[0]}) is less than target presets ({num_presets}). Reducing clusters.")
                num_presets = features.shape[0] # Cannot have more clusters than samples

            if num_presets <= 0:
                print("Error: No presets to generate (num_presets is zero or negative). Skipping clustering.")
                continue
            
            kmeans = KMeans(n_clusters=num_presets, random_state=0, n_init="auto").fit(features)
            preset_indices = kmeans.labels_
            
            # Save preset parameters
            pruned_preset_parameters = []
            
            for preset_index in range(num_presets):
                cluster_indices = np.where(preset_indices == preset_index)[0]
                if len(cluster_indices) > 0:
                    cluster_index = cluster_indices[0]
                    preset_params = list(preset_parameters[cluster_index].values())
                    pruned_preset_parameters.append(preset_params)
            print(f"Generated {len(pruned_preset_parameters)} presets for {plugin_name}.")
        
            plugin_metadata["presets"] = pruned_preset_parameters
            
        else:
            # If no clustering is needed, just use the generated parameters
            plugin_metadata["presets"] = [list(params.values()) for params in preset_parameters if params is not None]
            print(f"Generated {len(plugin_metadata['presets'])} presets for {plugin_name} without clustering.")
        
        # Save the plugin metadata to a JSON file
        json_filepath = os.path.join(plugin_preset_dir, f"{safe_instance_name}.json")
            
        with open(json_filepath, "w") as f:
            json.dump(plugin_metadata, f, indent=4)
        print(f"Saved presets to {json_filepath}")

    # Clean up the temporary directory when done
    if os.path.exists(tmp_output_dir):
        shutil.rmtree(tmp_output_dir)
import os
import time
import json
import shutil
import reapy
import logging
import argparse
from collections import defaultdict
from utils.data_class import Project
from utils.reaper_utils import batch_render_fx, delete_all_tracks
from utils.main_utils import *
from utils import PLUGIN_PRESETS_DIR
from typing import List, Tuple, Dict, Set
import concurrent.futures
from itertools import islice

# --- Default Configuration Values ---
DEFAULT_SAVE_MODE = "human-readable"  # Options: 'training-ready' (H5/pickle) or 'human-readable' (WAV/YAML)
DEFAULT_SAVE_COMPRESSION_RATE = 4
# DEFAULT_METADATA_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proj_metadata/reaper_test/project_1.yaml')
DEFAULT_METADATA_YAML_PATH = 'proj_metadata/reaper_test.yaml'
DEFAULT_FINAL_OUTPUT_DIR = "/home/isaac/Music/reaper_test_output"
DEFAULT_BATCH_SIZE = 40
DEFAULT_PROJECT_BATCH_SIZE = 512
DEFAULT_METADATA_START_IDX = 0
DEFAULT_METADATA_END_IDX = None
DEFAULT_FILENAME_OFFSET = 0
DEFAULT_RAM_DISK_GB = 128  # 0 means disabled

# Type Alias for Task ID
TaskId = Tuple[int, int]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_time_dhms(seconds):
    """Convert seconds to days:hours:minutes:seconds format"""
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}d:{hours:02d}h:{minutes:02d}m:{seconds:02d}s"
    elif hours > 0:
        return f"{hours}h:{minutes:02d}m:{seconds:02d}s"
    else:
        return f"{minutes}m:{seconds:02d}s"

def main(save_mode=DEFAULT_SAVE_MODE, save_compression_rate=DEFAULT_SAVE_COMPRESSION_RATE,
         metadata_yaml_path=DEFAULT_METADATA_YAML_PATH, final_output_dir=DEFAULT_FINAL_OUTPUT_DIR,
         batch_size=DEFAULT_BATCH_SIZE, project_batch_size=DEFAULT_PROJECT_BATCH_SIZE,
         metadata_start_idx=DEFAULT_METADATA_START_IDX, metadata_end_idx=DEFAULT_METADATA_END_IDX,
         filename_offset=DEFAULT_FILENAME_OFFSET, ram_disk_gb=DEFAULT_RAM_DISK_GB):
    """
    Loads current_batch_projects from YAML, processes them layer by layer using parallel batch processing,
    and saves the final outputs.
    
    **Workflow and Structure:**

    1. **Initialization:**
       - Loads project definitions from YAML metadata
       - Creates temporary working directories
       - Sets up logging and configuration

    2. **Graph Construction:**
       - Builds a directed acyclic graph (DAG) of processing tasks where:
         - Nodes are FX chains identified by (project_index, chain_index) tuples
         - Edges represent signal flow between chains
       - Tracks each chain's dependencies via:
         - `processed_chains`: Set of completed tasks
         - `chain_outputs`: Maps tasks to their output file paths, specially handling splitters with multiple outputs
         - `in_degree`: Counts each task's direct predecessors
         - `predecessors`: Maps each task to its predecessor tasks with weights
         - `successors`: Maps each task to the chains that depend on its output

    3. **Layer-by-Layer Parallel Processing:**
       - Tasks are processed in "layers" (all tasks with same dependency depth)
       - For each layer:
         a) **Task Grouping:** Tasks are grouped into batches respecting:
            - Sidechain dependencies (tasks with sidechains must be in the same batch)
            - Splitter plugins (one source track→multiple output tracks)
            - Resource limits (BATCH_SIZE)
         b) **Parallel Batch Preparation:** Using ThreadPoolExecutor for:
            - Input path determination (original audio for Layer 0, predecessor outputs for others)
            - Audio mixing for tasks with multiple inputs
            - Output path creation
            - FX chain configuration conversion
            - Thread-safe updating of chain_outputs via local copies
         c) **Batch Rendering:** Via batch_render_fx with proper send maps for sidechains/splitters
         d) **State Updates:** Mark completed tasks and update graph for next layer

    4. **Parallel Final Output Processing:**
       - Finds final output chains (those without successors)
       - Copies final outputs to destination directories
       - Preserves input files for reference
       - All operations parallelized with ThreadPoolExecutor

    5. **Cleanup:**
       - Removes all temporary files
       - Clears REAPER tracks
       
    The implementation handles advanced scenarios including:
    - Mixing multiple input signals with specified weights
    - Sidechain routing between tracks
    - Splitter plugins with multiple outputs
    - Thread safety for parallel operations
    - Proper error recovery and logging

    **To Run This:**

    1. Ensure utils/data_class.py and utils/reaper_utils.py are present
    2. Verify metadata YAML file exists and follows the Project structure specification
    3. Ensure all referenced input audio files exist
    4. Have REAPER running with reapy configured
    5. Run `python main.py` with arguments
    """
    start_time = time.time()
    logging.info("Starting processing...")

    
    if not os.path.exists(metadata_yaml_path):
        logging.error(f"Metadata YAML file not found: {metadata_yaml_path}")
        return
    
    os.makedirs(final_output_dir, exist_ok=True)
    os.makedirs(os.path.join(final_output_dir, "metadata"), exist_ok=True)

    # --- Metadata Handling: Check for existing data and append if offset matches ---
    destination_metadata_path = os.path.join(final_output_dir, "metadata", "metadata.yaml")
    if os.path.exists(destination_metadata_path):
        logging.info(f"Existing metadata found at {destination_metadata_path}. Verifying filename_offset...")
        try:
            # Load existing projects to get the count
            existing_projects = Project.load_from_yaml(destination_metadata_path, num_cores = batch_size)
            num_existing_projects = len(existing_projects)
            
            if num_existing_projects == filename_offset:
                logging.info(f"Offset {filename_offset} matches existing project count {num_existing_projects}. Appending new metadata.")
                with open(metadata_yaml_path, 'r') as source_file, open(destination_metadata_path, 'a') as dest_file:
                    dest_file.write('\n') # Ensure separation between YAML documents
                    shutil.copyfileobj(source_file, dest_file)
            else:
                logging.critical(f"CRITICAL ERROR: Mismatch between existing project count ({num_existing_projects}) and filename_offset ({filename_offset}).")
                logging.critical("To prevent data corruption, processing is halted. Please correct the --filename-offset argument.")
                return # Stop execution
        except Exception as e:
            logging.critical(f"CRITICAL ERROR: Could not process existing metadata file {destination_metadata_path}: {e}")
            return # Stop execution
    else:
        # If no metadata exists, just copy the new file.
        logging.info("No existing metadata found. Copying new metadata file.")
        try:
            shutil.copy2(metadata_yaml_path, destination_metadata_path)
        except Exception as e:
            logging.error(f"Error copying initial metadata file: {e}")
            return

    # Configure RAM disk if requested
    global_tmp_dir = configure_ram_disk(ram_gb=ram_disk_gb, prefix="wildfx_main")
    logging.info(f"Created global temporary directory: {global_tmp_dir}")

    # 1. Load projects
    logging.info(f"Loading current_batch_projects from {metadata_yaml_path}...")
    projects = Project.load_from_yaml(metadata_yaml_path, num_cores = batch_size)
    logging.info(f"Loaded {len(projects)} projects.")

    # Load parameters' names of plugins
    available_plugins_param_names = {}
    os.makedirs(os.path.join(final_output_dir, "metadata"), exist_ok=True)
    for filename in os.listdir(PLUGIN_PRESETS_DIR):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(PLUGIN_PRESETS_DIR, filename), 'r') as f:
                    preset_data = json.load(f)
                    available_plugins_param_names[preset_data['fx_name']] = list(preset_data['valid_params'].keys())
                # copy to final output directory
                shutil.copy2(os.path.join(PLUGIN_PRESETS_DIR, filename), os.path.join(final_output_dir, "metadata", filename))
            except FileExistsError or PermissionError:
                print(f"Plugin presets {filename} already exists in the final output directory. Skipping copy.")
            except Exception as e:
                print(f"Error loading plugin preset file {filename}: {e}")
                
    # Copy complete metadata to final output directory
    # try:
    #     shutil.copy2(metadata_yaml_path, os.path.join(final_output_dir, "metadata", "metadata.yaml"))
    # except FileExistsError:
    #     with open(metadata_yaml_path, 'r') as original_file:
    #         original_metadata = original_file.read()
    #     num_original_metadata = original_metadata.count('\n')

    #     if filename_offset == num_original_metadata:
    #         print(f"Metadata {metadata_yaml_path} already exists in the final output directory. Appending to existing file.")
    #         with open(os.path.join(final_output_dir, "metadata", "metadata.yaml"), 'a') as f:
    #             f.write(original_metadata)
    # except Exception as e:
    #     print(f"Error copying metadata file: {e}")
    
    # Check if metadata_start_idx and metadata_end_idx are valid
    if metadata_start_idx < 0 or metadata_start_idx >= len(projects):
        logging.error(f"Invalid metadata_start_idx: {metadata_start_idx}. Must be between 0 and {len(projects)-1}.")
        return
    
    # The process would create tremendous number of tmp files, so better divide all projects into batches
    for offset in range(metadata_start_idx, metadata_end_idx if metadata_end_idx is not None and metadata_end_idx < len(projects) else len(projects), project_batch_size):
        current_batch_projects = list(islice(projects, offset, offset + project_batch_size))
        # 2. Initialize Processing State for all current_batch_projects
        processed_chains: Set[TaskId] = set()
        chain_outputs: Dict[TaskId, str] = {}
        in_degree: Dict[TaskId, int] = defaultdict(int)
        successors: Dict[TaskId, List[TaskId]] = defaultdict(list)
        predecessors: Dict[TaskId, Dict[TaskId, float | int]] = defaultdict(dict) # Needed for mixing

        logging.info(f"\n--- Processing Project {offset} - {min(offset + project_batch_size - 1, len(projects)-1)} ---") # Adjust index for the current batch
        logging.info(f"\n--- Will Save As Project {offset + filename_offset} - {min(offset + filename_offset + project_batch_size - 1, len(projects)-1)} ---") # Adjust index for the current batch

        # Build graph structure info (in_degree, successors, predecessors)
        for proj_idx, project in enumerate(current_batch_projects):
            # Map input_FxChain index to the corresponding InputAudio object for easy lookup
            input_audio_map = {ia.input_FxChain: ia.audio_path for ia in project.input_audios}

            for chain_idx, chain in enumerate(project.FxChains):
                task_id: TaskId = (proj_idx, chain_idx)

                # Build successors and predecessors from next_chains
                for next_idx, weight in chain.next_chains.items():
                    successor_task_id: TaskId = (proj_idx, next_idx)
                    successors[task_id].append(successor_task_id)
                    predecessors[successor_task_id].update({task_id: weight})
                    # In-degree will be calculated based on predecessors map later

                # Store original input paths for designated input chains
                if chain_idx in input_audio_map:
                    chain_outputs[task_id] = input_audio_map[chain_idx]
                # Initialize in_degree for all tasks (will be updated based on predecessors)
                in_degree[task_id] = 0

            # Calculate in-degree based on the built predecessors map
            for task_id in predecessors:
                in_degree[task_id] = len(predecessors[task_id])

            # Verify input chains have in-degree 0
            for chain_idx in input_audio_map:
                task_id: TaskId  = (proj_idx, chain_idx)
                if in_degree[task_id] != 0:
                    logging.error(f"Validation Error: Input chain {task_id} has predecessors {predecessors[task_id]}. Check YAML.")
                    # Potentially raise error or skip project

        # Process Layer by Layer
        current_layer = 0
        num_splitter_tasks = {}
        next_layer_tasks = [task_id for task_id, degree in in_degree.items() 
                        if degree == 0 and task_id in chain_outputs]
        reaper_project = reapy.Project() # Get a handle to the REAPER project
        
        # Track audio files by layer for memory cleanup
        layer_audio_files = {}  # layer_num -> set of file paths created in that layer

        try:
            while next_layer_tasks:
                current_layer_tasks = next_layer_tasks
                next_layer_tasks = []
                
                logging.info(f"\n--- Processing Layer {current_layer} ---")

                # Get task_id batches that respect sidechain dependencies
                tasks_in_batches, gain_batches, send_map_batches, tracks_to_unselect, current_num_splitter_tasks = process_layer_with_tracksend_awareness(current_layer_tasks, current_batch_projects, batch_size)
                num_splitter_tasks.update(current_num_splitter_tasks) # Update the global splitter task count
                
                # Replace the sequential batch preparation with parallel processing
                batches_data = {}  # Will store all results indexed by batch position

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit each batch for parallel processing with its position index
                    future_to_batch_idx = {
                        executor.submit(
                            prepare_batch, 
                            batch_idx, 
                            tasks_in_batch, 
                            current_batch_projects, 
                            current_layer, 
                            chain_outputs, 
                            predecessors, 
                            num_splitter_tasks, 
                            global_tmp_dir
                        ): batch_idx 
                        for batch_idx, tasks_in_batch in enumerate(tasks_in_batches)
                    }
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_batch_idx):
                        batch_idx = future_to_batch_idx[future]
                        try:
                            batch_result = future.result()
                            if batch_result:  # Check if batch preparation was successful
                                batches_data[batch_idx] = batch_result
                            else:
                                logging.error(f"Batch {batch_idx} preparation returned None")
                        except Exception as e:
                            logging.error(f"Batch {batch_idx} preparation failed: {e}")

                # Reconstruct ordered lists for batch_render_fx
                batches_inputs = []
                batches_outputs = []
                batches_fx_chains = []
                
                # Track mixed audio files created during batch preparation
                if current_layer not in layer_audio_files:
                    layer_audio_files[current_layer] = set()

                for batch_idx in range(len(tasks_in_batches)):
                    if batch_idx in batches_data:
                        batch_inputs, batch_outputs, batch_fx_chains, local_chain_outputs = batches_data[batch_idx]
                        batches_inputs.append(batch_inputs)
                        batches_outputs.append(batch_outputs)
                        batches_fx_chains.append(batch_fx_chains)
                        chain_outputs.update(local_chain_outputs)
                        
                        # Track any mixed input files that were created during batch preparation
                        for input_path in batch_inputs:
                            if input_path and input_path.startswith(global_tmp_dir):
                                layer_audio_files[current_layer].add(input_path)
                    else:
                        # Handle missing batch (error occurred during preparation)
                        logging.error(f"Missing data for batch {batch_idx}, using empty lists")
                        batches_inputs.append([])
                        batches_outputs.append([])
                        batches_fx_chains.append([])            
                
                # Execute batch rendering if there are tasks
                flatten_tasks_in_batches = [task for batch in tasks_in_batches for task in batch]
                if flatten_tasks_in_batches:
                    # Clean up audio files from previous layers to save memory
                    # Keep the previous layer and delete the one before that
                    if current_layer > 0:
                        logging.info(f"Cleaning up intermediate audio files from Layer {current_layer - 1}...")
                        cleanup_layer = current_layer - 1
                        if cleanup_layer in layer_audio_files:
                            files_to_delete = layer_audio_files[cleanup_layer]
                            deleted_count = 0
                            for file_path in files_to_delete:
                                try:
                                    if os.path.exists(file_path):
                                        os.remove(file_path)
                                        deleted_count += 1
                                except Exception as e:
                                    logging.warning(f"Could not delete intermediate file {file_path}: {e}")
                            
                            if deleted_count > 0:
                                logging.info(f"Cleaned up {deleted_count} intermediate audio files from Layer {cleanup_layer}")
                            
                            # Remove the layer from tracking
                            del layer_audio_files[cleanup_layer]

                    logging.info(f"Rendering {len(flatten_tasks_in_batches)} tasks in batch for Layer {current_layer}...")
                    # try:
                    # Ensure tracks are clear before batch
                    batch_render_fx(
                        reaper_project, 
                        global_tmp_dir,
                        batches_inputs, 
                        batches_outputs, 
                        batches_fx_chains, 
                        gain_batches,
                        send_map_batches, 
                        tracks_to_unselect, 
                        batch_size)
                    logging.info(f"Finished rendering batch for Layer {current_layer}.")
                    
                    # Parallelize output verification
                    file_exists_results = {}
                    
                    # Define worker function for checking file existence
                    flatten_batches_outputs = [output_path for batch in batches_outputs for output_path in batch]
                    def check_file_exists(i):
                        task_id = flatten_tasks_in_batches[i]
                        output_path = flatten_batches_outputs[i]
                        if output_path is None: # No output path for splitter source
                            return task_id, True
                        else:
                            return task_id, os.path.exists(output_path)
                    
                    # Use thread pool for file existence checks
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, len(flatten_tasks_in_batches))) as executor:
                        future_to_index = {executor.submit(check_file_exists, i): i for i in range(len(flatten_tasks_in_batches))}
                        for future in concurrent.futures.as_completed(future_to_index):
                            task_id, exists = future.result()
                            file_exists_results[task_id] = exists
                
                    # Verify outputs exist after rendering and track created files
                    for i, task_id in enumerate(flatten_tasks_in_batches):
                        if not file_exists_results[task_id]:
                            logging.error(f"Output file missing after render for task {task_id}: {flatten_batches_outputs[i]}")
                            if task_id in chain_outputs:
                                del chain_outputs[task_id]
                        else:
                            processed_chains.add(task_id)
                            # Track created audio files for this layer (rendered outputs)
                            output_path = flatten_batches_outputs[i]
                            if output_path and output_path.startswith(global_tmp_dir):
                                layer_audio_files[current_layer].add(output_path)
                            
                            for successor_task_id in successors[task_id]:
                                in_degree[successor_task_id] -= 1
                                if in_degree[successor_task_id] == 0:
                                    next_layer_tasks.append(successor_task_id)

                    # except Exception as e:
                    #     logging.error(f"Error during batch_render_fx for Layer {current_layer}: {e}")
                        # Handle error - potentially skip successors or retry
                else:
                    logging.info(f"No tasks to render in Layer {current_layer}.")

                current_layer += 1
                logging.info(f"Next layer: {next_layer_tasks}")


            # 5. Final Output Handling
            logging.info("\n--- Handling Final Outputs ---")
            all_processed_successfully = True

            # Use ProcessPoolExecutor for parallel processing
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Submit all project final output tasks with proper parameters
                future_to_proj = {
                    executor.submit(
                        process_final_output, 
                        proj_idx,     # Local index within current batch
                        offset + filename_offset,      # Current batch offset + Global starting index
                        current_batch_projects,
                        chain_outputs,
                        processed_chains,
                        available_plugins_param_names,
                        final_output_dir,
                        save_mode, 
                        save_compression_rate
                    ): proj_idx + offset + filename_offset
                    for proj_idx in range(len(current_batch_projects))
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_proj):
                    proj_idx = future_to_proj[future]
                    try:
                        success = future.result()
                        if not success:
                            all_processed_successfully = False
                    except Exception as e:
                        logging.error(f"Exception processing final output for Project {(proj_idx+offset+filename_offset)}: {e}")
                        all_processed_successfully = False

            if not all_processed_successfully:
                logging.warning("Some current_batch_projects did not complete processing successfully.")    
                
        except Exception as e:
            logging.exception(f"An unexpected error occurred during processing: {e}") # Log full traceback
            logging.info(f"Cleaning up metadata in {global_tmp_dir}...")
            if os.path.exists(os.path.join(final_output_dir, "metadata")):
                shutil.rmtree(os.path.join(final_output_dir, "metadata"), ignore_errors=True)
        finally:
            # 6. Cleanup
            logging.info(f"Cleaning up temporary directory: {global_tmp_dir}")
            if os.path.exists(global_tmp_dir):
                shutil.rmtree(global_tmp_dir)
            # Clean REAPER tracks
            try:
                logging.info("Cleaning up REAPER tracks...")
                delete_all_tracks()
            except Exception as e:
                logging.warning(f"Could not clean REAPER tracks: {e}")
            logging.info("Processing finished.")
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_formatted = format_time_dhms(elapsed_time)
    logging.info(f"Total processing time for {len(chain_outputs)} tasks in {len(projects)-metadata_start_idx} projects: {elapsed_formatted} ({elapsed_time:.2f} seconds)")


def parse_args():
    """Parse command-line arguments for the ReproFX-Graph processing tool"""
    parser = argparse.ArgumentParser(
        description="ReproFX-Graph - Process audio through FX chains defined in YAML metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument('--save-mode', type=str, default=DEFAULT_SAVE_MODE,
                        choices=['training-ready', 'human-readable'],
                        help="Output format: 'training-ready' (H5/pickle) or 'human-readable' (WAV/YAML)")
    
    parser.add_argument('--save-compression-rate', type=int, default=DEFAULT_SAVE_COMPRESSION_RATE,
                        help="Compression method for H5 files (1-9, higher means more compression)")
    
    # Path options
    parser.add_argument('--metadata-yaml', type=str, default=DEFAULT_METADATA_YAML_PATH,
                        help="Path to the YAML metadata file")
    
    parser.add_argument('--output-dir', type=str, default=DEFAULT_FINAL_OUTPUT_DIR,
                        help="Output directory for processed files")
    
    # Processing parameters
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for rendering (slightly larger than CPU cores)")
    
    parser.add_argument('--project-batch-size', type=int, default=DEFAULT_PROJECT_BATCH_SIZE,
                        help="Number of projects to process in a batch")
    
    parser.add_argument('--start-idx', type=int, default=DEFAULT_METADATA_START_IDX,
                        help="Starting index for processing projects")
    
    parser.add_argument('--end-idx', type=int, default=DEFAULT_METADATA_END_IDX,
                        help="Ending index for processing projects (None = process until end)")
    
    parser.add_argument('--filename-offset', type=int, default=DEFAULT_FILENAME_OFFSET,
                        help="Offset to add to project index for final output directory naming")
    
    parser.add_argument('--ram-disk-gb', type=float, default=DEFAULT_RAM_DISK_GB,
                        help="RAM disk size in GB for temporary files (0 = disabled)")
    
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set logging level from command line
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Call main function with parsed arguments
    main(
        save_mode=args.save_mode,
        save_compression_rate=args.save_compression_rate,
        metadata_yaml_path=args.metadata_yaml,
        final_output_dir=args.output_dir,
        batch_size=args.batch_size,
        project_batch_size=args.project_batch_size,
        metadata_start_idx=args.start_idx,
        metadata_end_idx=args.end_idx,
        filename_offset=args.filename_offset,
        ram_disk_gb=args.ram_disk_gb,
    )
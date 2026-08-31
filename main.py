import argparse
import concurrent.futures
import filecmp
import json
import logging
import os
import shutil
import tempfile
import time
from collections import defaultdict
from itertools import islice
from typing import Dict, List, Optional, Set, Tuple

import reapy

from utils import PLUGIN_PRESETS_DIR
from utils.data_class import Project
from utils.main_utils import (
    TAIL_FX_TYPES,
    configure_ram_disk,
    prepare_batch,
    process_final_output,
    process_layer_with_tracksend_awareness,
)
from utils.reaper_utils import batch_render_fx, delete_all_tracks

# --- Default Configuration Values ---
DEFAULT_SAVE_MODE = "human-readable"  # Options: 'training-ready' (H5/pickle) or 'human-readable' (WAV/YAML)
DEFAULT_SAVE_COMPRESSION_RATE = 4
# DEFAULT_METADATA_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proj_metadata/reaper_test/project_1.yaml')
DEFAULT_METADATA_YAML_PATH = None
DEFAULT_FINAL_OUTPUT_DIR = "wildfx_output"
DEFAULT_BATCH_SIZE = 40
DEFAULT_PROJECT_BATCH_SIZE = 512
DEFAULT_METADATA_START_IDX = 0
DEFAULT_METADATA_END_IDX = None
DEFAULT_FILENAME_OFFSET = 0
DEFAULT_RAM_DISK_GB = 4
DEFAULT_RENDER_TAIL_SECONDS = 3.0

# Type Alias for Task ID
TaskId = Tuple[int, int]
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def format_time_dhms(seconds: float) -> str:
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


def main(
    save_mode: str = DEFAULT_SAVE_MODE,
    save_compression_rate: int = DEFAULT_SAVE_COMPRESSION_RATE,
    metadata_yaml_path: Optional[str] = DEFAULT_METADATA_YAML_PATH,
    final_output_dir: str = DEFAULT_FINAL_OUTPUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    project_batch_size: int = DEFAULT_PROJECT_BATCH_SIZE,
    metadata_start_idx: int = DEFAULT_METADATA_START_IDX,
    metadata_end_idx: Optional[int] = DEFAULT_METADATA_END_IDX,
    filename_offset: int = DEFAULT_FILENAME_OFFSET,
    ram_disk_gb: float = DEFAULT_RAM_DISK_GB,
    render_tail_seconds: float = DEFAULT_RENDER_TAIL_SECONDS,
) -> bool:
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

    if save_mode not in {"training-ready", "human-readable", "both"}:
        logging.error("Unsupported save mode: %s", save_mode)
        return False
    if batch_size < 1 or project_batch_size < 1:
        logging.error("Batch sizes must be positive")
        return False
    if not 0 <= save_compression_rate <= 9:
        logging.error("H5 compression rate must be between 0 and 9")
        return False
    if filename_offset < 0 or render_tail_seconds < 0 or ram_disk_gb < 0:
        logging.error("Offsets, tail duration, and RAM-disk size must be non-negative")
        return False

    if metadata_yaml_path is None or not os.path.exists(metadata_yaml_path):
        logging.error(f"Metadata YAML file not found: {metadata_yaml_path}")
        return False

    # Validate an existing dataset before rendering, but commit metadata only
    # after every requested project has completed successfully.
    destination_metadata_path = os.path.join(
        final_output_dir, "metadata", "metadata.yaml"
    )
    existing_projects: List[Project] = []
    if os.path.exists(destination_metadata_path):
        logging.info(
            f"Existing metadata found at {destination_metadata_path}. Verifying filename_offset..."
        )
        try:
            # Load existing projects to get the count
            existing_projects = Project.load_from_yaml(
                destination_metadata_path, num_cores=batch_size
            )
            num_existing_projects = len(existing_projects)

            if num_existing_projects == filename_offset:
                logging.info(
                    "Offset %d matches existing project count %d.",
                    filename_offset,
                    num_existing_projects,
                )
                missing_project_dirs = [
                    index
                    for index in range(num_existing_projects)
                    if not os.path.isdir(
                        os.path.join(
                            final_output_dir,
                            f"project_{index:08d}",
                        )
                    )
                ]
                if missing_project_dirs:
                    logging.critical(
                        "Existing metadata references missing project folders: %s",
                        missing_project_dirs[:10],
                    )
                    return False
            else:
                logging.critical(
                    f"CRITICAL ERROR: Mismatch between existing project count ({num_existing_projects}) and filename_offset ({filename_offset})."
                )
                logging.critical(
                    "To prevent data corruption, processing is halted. Please correct the --filename-offset argument."
                )
                return False
        except Exception as e:
            logging.critical(
                f"CRITICAL ERROR: Could not process existing metadata file {destination_metadata_path}: {e}"
            )
            return False

    # Configure RAM disk if requested
    # 1. Load projects
    logging.info(f"Loading current_batch_projects from {metadata_yaml_path}...")
    projects = Project.load_from_yaml(metadata_yaml_path, num_cores=batch_size)
    logging.info(f"Loaded {len(projects)} projects.")
    if not projects:
        logging.error("Metadata contains no projects.")
        return False

    if metadata_start_idx < 0 or metadata_start_idx >= len(projects):
        logging.error(
            "Invalid metadata_start_idx: %d. Must be between 0 and %d.",
            metadata_start_idx,
            len(projects) - 1,
        )
        return False
    end_idx = (
        metadata_end_idx
        if metadata_end_idx is not None and metadata_end_idx < len(projects)
        else len(projects)
    )
    if end_idx <= metadata_start_idx:
        logging.error("metadata_end_idx must be greater than metadata_start_idx")
        return False
    target_project_dirs = [
        os.path.join(final_output_dir, f"project_{index:08d}")
        for index in range(
            filename_offset,
            filename_offset + end_idx - metadata_start_idx,
        )
    ]
    existing_targets = [path for path in target_project_dirs if os.path.exists(path)]
    if existing_targets:
        logging.error(
            "Refusing to overwrite existing target project folders: %s",
            existing_targets[:10],
        )
        return False

    # Load parameters' names of plugins
    available_plugins_param_names = {}
    os.makedirs(final_output_dir, exist_ok=True)
    os.makedirs(os.path.join(final_output_dir, "metadata"), exist_ok=True)
    for filename in sorted(os.listdir(PLUGIN_PRESETS_DIR)):
        if filename.endswith(".json"):
            try:
                source_path = os.path.join(PLUGIN_PRESETS_DIR, filename)
                destination_path = os.path.join(
                    final_output_dir,
                    "metadata",
                    filename,
                )
                with open(source_path, "r") as f:
                    preset_data = json.load(f)
                    available_plugins_param_names[preset_data["fx_name"]] = list(
                        preset_data["valid_params"].keys()
                    )
                if os.path.exists(destination_path):
                    if not filecmp.cmp(
                        source_path,
                        destination_path,
                        shallow=False,
                    ):
                        logging.error(
                            "Existing dataset uses different plugin metadata: %s",
                            destination_path,
                        )
                        return False
                else:
                    with tempfile.NamedTemporaryFile(
                        dir=os.path.dirname(destination_path),
                        prefix=f".{filename}.",
                        delete=False,
                    ) as temporary_file:
                        temporary_path = temporary_file.name
                    try:
                        shutil.copy2(source_path, temporary_path)
                        os.replace(temporary_path, destination_path)
                    finally:
                        if os.path.exists(temporary_path):
                            os.remove(temporary_path)
            except (FileExistsError, PermissionError) as exc:
                logging.error("Cannot copy plugin metadata %s: %s", filename, exc)
                return False
            except Exception as e:
                logging.error("Cannot load plugin metadata %s: %s", filename, e)
                return False

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

    overall_success = True
    total_tasks_processed = 0

    # The process would create tremendous number of tmp files, so better divide all projects into batches
    for offset in range(metadata_start_idx, end_idx, project_batch_size):
        current_batch_end = min(offset + project_batch_size, end_idx)
        current_batch_projects = list(islice(projects, offset, current_batch_end))
        # 2. Initialize Processing State for all current_batch_projects
        processed_chains: Set[TaskId] = set()
        chain_outputs: Dict[TaskId, str | List[str]] = {}
        in_degree: Dict[TaskId, int] = defaultdict(int)
        successors: Dict[TaskId, List[TaskId]] = defaultdict(list)
        predecessors: Dict[TaskId, Dict[TaskId, float | int]] = defaultdict(
            dict
        )  # Needed for mixing

        logging.info(
            "\n--- Processing Project %d - %d ---",
            offset,
            current_batch_end - 1,
        )
        logging.info(
            "\n--- Will Save As Project %d - %d ---",
            filename_offset + offset - metadata_start_idx,
            filename_offset + current_batch_end - metadata_start_idx - 1,
        )

        # Build graph structure info (in_degree, successors, predecessors)
        for proj_idx, project in enumerate(current_batch_projects):
            # Map input_FxChain index to the corresponding InputAudio object for easy lookup
            input_audio_map = {
                ia.input_FxChain: ia.audio_path for ia in project.input_audios
            }

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
                task_id: TaskId = (proj_idx, chain_idx)
                if in_degree[task_id] != 0:
                    raise ValueError(
                        f"Input chain {task_id} has predecessors "
                        f"{predecessors[task_id]}. Check the YAML."
                    )

        # Process Layer by Layer
        current_layer = 0
        num_splitter_tasks = {}
        next_layer_tasks = [
            task_id
            for task_id, degree in in_degree.items()
            if degree == 0 and task_id in chain_outputs
        ]
        reaper_project = reapy.Project()  # Get a handle to the REAPER project
        global_tmp_dir = configure_ram_disk(ram_gb=ram_disk_gb, prefix="wildfx_main")
        logging.info("Created temporary directory: %s", global_tmp_dir)
        remaining_consumers = {
            task_id: len(successors[task_id]) for task_id in in_degree
        }

        try:
            while next_layer_tasks:
                current_layer_tasks = next_layer_tasks
                next_layer_tasks = []

                logging.info(f"\n--- Processing Layer {current_layer} ---")

                # Get task_id batches that respect sidechain dependencies
                (
                    tasks_in_batches,
                    gain_batches,
                    send_map_batches,
                    tracks_to_unselect,
                    current_num_splitter_tasks,
                ) = process_layer_with_tracksend_awareness(
                    current_layer_tasks, current_batch_projects, batch_size
                )
                num_splitter_tasks.update(
                    current_num_splitter_tasks
                )  # Update the global splitter task count

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
                            global_tmp_dir,
                        ): batch_idx
                        for batch_idx, tasks_in_batch in enumerate(tasks_in_batches)
                    }

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_batch_idx):
                        batch_idx = future_to_batch_idx[future]
                        # future.result() deliberately propagates errors. Rendering
                        # a partial list would misalign tasks and output files.
                        batches_data[batch_idx] = future.result()

                # Reconstruct ordered lists for batch_render_fx
                batches_inputs = []
                batches_outputs = []
                batches_fx_chains = []

                for batch_idx in range(len(tasks_in_batches)):
                    if batch_idx not in batches_data:
                        raise RuntimeError(f"Missing prepared batch {batch_idx}")
                    (
                        batch_inputs,
                        batch_outputs,
                        batch_fx_chains,
                        local_chain_outputs,
                    ) = batches_data[batch_idx]
                    batches_inputs.append(batch_inputs)
                    batches_outputs.append(batch_outputs)
                    batches_fx_chains.append(batch_fx_chains)
                    chain_outputs.update(local_chain_outputs)

                # Every current task has now copied/mixed each predecessor into
                # its own render input. Release predecessor renders only after
                # their final graph consumer has prepared its input. A simple
                # layer-based cleanup is incorrect for skip edges and deletes
                # early-finishing project sinks in mixed-depth batches.
                predecessor_outputs_to_delete = set()
                for task_id in current_layer_tasks:
                    for predecessor_task_id in predecessors.get(task_id, {}):
                        remaining_consumers[predecessor_task_id] -= 1
                        if remaining_consumers[predecessor_task_id] < 0:
                            raise RuntimeError(
                                "Negative remaining-consumer count for "
                                f"{predecessor_task_id}"
                            )
                        if remaining_consumers[predecessor_task_id] == 0:
                            predecessor_output = chain_outputs.get(predecessor_task_id)
                            output_paths = (
                                predecessor_output
                                if isinstance(predecessor_output, list)
                                else [predecessor_output]
                            )
                            predecessor_outputs_to_delete.update(
                                path
                                for path in output_paths
                                if isinstance(path, str)
                                and path.startswith(global_tmp_dir)
                            )

                for file_path in predecessor_outputs_to_delete:
                    if os.path.exists(file_path):
                        os.remove(file_path)

                # Execute batch rendering if there are tasks
                flatten_tasks_in_batches = [
                    task for batch in tasks_in_batches for task in batch
                ]
                if flatten_tasks_in_batches:
                    logging.info(
                        f"Rendering {len(flatten_tasks_in_batches)} tasks in batch for Layer {current_layer}..."
                    )
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
                        batch_size,
                        render_tail_seconds=[
                            (
                                render_tail_seconds
                                if any(
                                    fx.fx_type in TAIL_FX_TYPES
                                    for task_id in set(tasks_in_batch)
                                    for fx in current_batch_projects[task_id[0]]
                                    .FxChains[task_id[1]]
                                    .FxChain
                                )
                                else 0.0
                            )
                            for tasks_in_batch in tasks_in_batches
                        ],
                    )
                    logging.info(f"Finished rendering batch for Layer {current_layer}.")

                    # A splitter appears N+1 times in the REAPER batch but is one
                    # graph task. Verify and advance it exactly once.
                    for task_id in current_layer_tasks:
                        output_value = chain_outputs.get(task_id)
                        if isinstance(output_value, list):
                            output_paths = output_value
                            outputs_exist = bool(output_paths) and all(
                                path is not None and os.path.exists(path)
                                for path in output_paths
                            )
                        else:
                            output_paths = [output_value]
                            outputs_exist = output_value is not None and os.path.exists(
                                output_value
                            )
                        if not outputs_exist:
                            raise RuntimeError(
                                f"Output missing after render for task {task_id}: "
                                f"{output_value}"
                            )

                        processed_chains.add(task_id)
                        for successor_task_id in successors[task_id]:
                            in_degree[successor_task_id] -= 1
                            if in_degree[successor_task_id] < 0:
                                raise RuntimeError(
                                    f"Negative in-degree for {successor_task_id}"
                                )
                            if in_degree[successor_task_id] == 0:
                                next_layer_tasks.append(successor_task_id)

                    # Mixed inputs are private snapshots for this render layer
                    # and are no longer needed after REAPER returns.
                    for batch_inputs in batches_inputs:
                        for input_path in batch_inputs:
                            if (
                                isinstance(input_path, str)
                                and input_path.startswith(global_tmp_dir)
                                and os.path.exists(input_path)
                            ):
                                os.remove(input_path)

                    # except Exception as e:
                    #     logging.error(f"Error during batch_render_fx for Layer {current_layer}: {e}")
                    # Handle error - potentially skip successors or retry
                else:
                    logging.info(f"No tasks to render in Layer {current_layer}.")

                current_layer += 1
                logging.info(f"Next layer: {next_layer_tasks}")

            # 5. Final Output Handling
            logging.info("\n--- Handling Final Outputs ---")
            if len(processed_chains) != len(in_degree):
                missing_tasks = sorted(set(in_degree) - processed_chains)
                raise RuntimeError(
                    f"Processing stopped before {len(missing_tasks)} tasks completed: "
                    f"{missing_tasks[:10]}"
                )
            all_processed_successfully = True

            # Final export is I/O-bound. Threads avoid copying every Project and
            # large path map into child processes and work safely in notebooks.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                output_batch_offset = filename_offset + offset - metadata_start_idx
                # Submit all project final output tasks with proper parameters
                future_to_proj = {
                    executor.submit(
                        process_final_output,
                        proj_idx,  # Local index within current batch
                        output_batch_offset,
                        current_batch_projects,
                        chain_outputs,
                        processed_chains,
                        available_plugins_param_names,
                        final_output_dir,
                        save_mode,
                        save_compression_rate,
                    ): proj_idx
                    + output_batch_offset
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
                        logging.error(
                            "Exception processing final output for Project %d: %s",
                            proj_idx,
                            e,
                        )
                        all_processed_successfully = False

            if not all_processed_successfully:
                raise RuntimeError(
                    "At least one project failed during final-output export"
                )
            total_tasks_processed += len(processed_chains)

        except Exception as e:
            logging.exception(
                f"An unexpected error occurred during processing: {e}"
            )  # Log full traceback
            overall_success = False
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

        if not overall_success:
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_formatted = format_time_dhms(elapsed_time)
    logging.info(
        "Total processing time for %d tasks in %d projects: %s (%.2f seconds)",
        total_tasks_processed,
        end_idx - metadata_start_idx,
        elapsed_formatted,
        elapsed_time,
    )
    if not overall_success:
        return False

    # Commit only the metadata for projects that were actually requested and
    # rendered. The temporary file prevents an interrupted write from corrupting
    # an existing dataset index.
    metadata_temp_path = f"{destination_metadata_path}.tmp"
    try:
        Project.save_to_yaml(
            existing_projects + projects[metadata_start_idx:end_idx],
            metadata_temp_path,
        )
        os.replace(metadata_temp_path, destination_metadata_path)
    except Exception:
        if os.path.exists(metadata_temp_path):
            os.remove(metadata_temp_path)
        logging.exception("Failed to commit dataset metadata")
        return False
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ReproFX-Graph processing tool"""
    parser = argparse.ArgumentParser(
        description="ReproFX-Graph - Process audio through FX chains defined in YAML metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration options
    parser.add_argument(
        "--save-mode",
        type=str,
        default=DEFAULT_SAVE_MODE,
        choices=["training-ready", "human-readable", "both"],
        help="Output format: 'training-ready' (H5/pickle) or 'human-readable' (WAV/YAML)",
    )

    parser.add_argument(
        "--save-compression-rate",
        type=int,
        default=DEFAULT_SAVE_COMPRESSION_RATE,
        help="Compression method for H5 files (1-9, higher means more compression)",
    )

    # Path options
    parser.add_argument(
        "--metadata-yaml",
        type=str,
        required=True,
        help="Path to the YAML metadata file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_FINAL_OUTPUT_DIR,
        help="Output directory for processed files",
    )

    # Processing parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for rendering (slightly larger than CPU cores)",
    )

    parser.add_argument(
        "--project-batch-size",
        type=int,
        default=DEFAULT_PROJECT_BATCH_SIZE,
        help="Number of projects to process in a batch",
    )

    parser.add_argument(
        "--start-idx",
        type=int,
        default=DEFAULT_METADATA_START_IDX,
        help="Starting index for processing projects",
    )

    parser.add_argument(
        "--end-idx",
        type=int,
        default=DEFAULT_METADATA_END_IDX,
        help="Ending index for processing projects (None = process until end)",
    )

    parser.add_argument(
        "--filename-offset",
        type=int,
        default=DEFAULT_FILENAME_OFFSET,
        help="Number of existing projects; new output numbering continues from this value",
    )

    parser.add_argument(
        "--ram-disk-gb",
        type=float,
        default=DEFAULT_RAM_DISK_GB,
        help="RAM disk size in GB for temporary files (0 = disabled)",
    )

    parser.add_argument(
        "--render-tail-seconds",
        type=float,
        default=DEFAULT_RENDER_TAIL_SECONDS,
        help="Tail added to layers containing delay, echo, or reverb effects",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set logging level from command line
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Call main function with parsed arguments
    succeeded = main(
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
        render_tail_seconds=args.render_tail_seconds,
    )
    raise SystemExit(0 if succeeded else 1)

import logging
import os
import pickle
import shutil
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import soundfile as sf

from utils.data_class import Project

TAIL_FX_TYPES = {"delay", "echo", "reverb"}


def configure_ram_disk(ram_gb: float = 4.0, prefix: str = "tmp") -> str:
    """Create a temporary directory in shared memory when capacity is sufficient."""
    ram_disk_path = "/dev/shm"
    if (
        ram_gb > 0
        and os.path.isdir(ram_disk_path)
        and os.access(ram_disk_path, os.W_OK)
    ):
        try:
            statvfs = os.statvfs(ram_disk_path)
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            if free_gb >= ram_gb:
                path = tempfile.mkdtemp(prefix=prefix, dir=ram_disk_path)
                logging.info(
                    "Using shared memory at %s (%.1f GiB free)",
                    path,
                    free_gb,
                )
                return path
            logging.warning(
                "Shared memory has %.1f GiB free, below the requested %.1f GiB",
                free_gb,
                ram_gb,
            )
        except OSError as exc:
            logging.warning("Cannot use %s: %s", ram_disk_path, exc)
    path = tempfile.mkdtemp(prefix=prefix)
    logging.info("Using disk-backed temporary directory %s", path)
    return path


def metadata_to_networkx(
    project: Project, available_plugins_param_names: Dict[str, List[str]]
) -> nx.DiGraph:
    """
    The graph implements three edge types:
    1. Gain edges (standard connections with mix weighting)
    2. Split edges (multi-output connections from splitter plugins)
    3. Sidechain edges (special inputs for sidechain-enabled effects)
    Data structure supports direct serialization to/from YAML for project definition
    """

    # Create a directed graph
    G = nx.DiGraph()
    # Track chains that need direct connections (empty chains). Empty chains used
    # as sidechain sources must remain as explicit passthrough nodes so the
    # control edge has an unambiguous source.
    empty_chains = []
    sidechain_source_chains = {
        fx.sidechain_input
        for chain_def in project.FxChains
        for fx in chain_def.FxChain
        if fx.sidechain_input is not None
    }
    # Triplets: (consumer chain id, consumer FX id, source chain id).
    sidechain_input_map = []
    # Create FX Chains
    for i, chain_def in enumerate(project.FxChains):
        if chain_def.FxChain:
            for j, fx in enumerate(chain_def.FxChain):
                if fx.sidechain_input is not None:
                    sidechain_input_map.append((i, j, fx.sidechain_input))
                G.add_node(
                    f"fx_{i}_{j}",
                    type="fx",
                    label=fx.fx_type,
                    instance=fx.fx_name,
                    params={
                        name: value
                        for name, value in zip(
                            available_plugins_param_names[fx.fx_name], fx.params
                        )
                        if value is not None
                    },
                )
                if j > 0:
                    # Connect to previous FX in the chain
                    G.add_edge(
                        f"fx_{i}_{j-1}",
                        f"fx_{i}_{j}",
                        type="send_signal",
                        label="main",
                        gain=1.0,
                    )  # default gain of 1.0
        else:
            # Create a empty node for empty chains
            G.add_node(f"fx_{i}_0", type="passthrough", label="passthrough")
            # Intermediate passthroughs can be contracted, but a final empty
            # chain is the graph's explicit sink and must remain serializable.
            if i not in sidechain_source_chains and chain_def.next_chains:
                empty_chains.append(f"fx_{i}_0")

        # Connect to next chains (networkx allows creating empty nodes by connecting to non-existing nodes)
        if len(chain_def.next_chains) != 0:  # not end node
            if len(chain_def.next_chains) > 1:  # splitter
                edge_type = "split_signal"
                edge_labels = [f"band_{i}" for i in range(len(chain_def.next_chains))]
            else:  # normal connection, only one target
                edge_type = "send_signal"
                edge_labels = ["main"]
            for (target_idx, gain), edge_label in zip(
                chain_def.next_chains.items(), edge_labels
            ):
                source_node = (
                    f"fx_{i}_{len(chain_def.FxChain)-1}"
                    if chain_def.FxChain
                    else f"fx_{i}_0"
                )
                G.add_edge(
                    source_node,
                    f"fx_{target_idx}_{0}",
                    type=edge_type,
                    label=edge_label,
                    gain=gain,
                )

    # Connect sidechain input nodes to fx chain nodes
    for consumer_chain_id, consumer_fx_id, source_chain_id in sidechain_input_map:
        source_chain = project.FxChains[source_chain_id]
        source_node_id = (
            f"fx_{source_chain_id}_{len(source_chain.FxChain)-1}"
            if source_chain.FxChain
            else f"fx_{source_chain_id}_0"
        )
        G.add_edge(
            source_node_id,
            f"fx_{consumer_chain_id}_{consumer_fx_id}",
            type="send_signal",
            label="control",
            gain=1.0,
        )

    # Create input audio nodes
    for i, ia in enumerate(project.input_audios):
        G.add_node(
            f"input_audio_{i}",
            type="audio",
            label=ia.audio_type,
            instance=ia.audio_path,
        )
        G.add_edge(
            f"input_audio_{i}",
            f"fx_{ia.input_FxChain}_0",
            type="send_signal",
            label="main",
            gain=1.0,
        )

    # Delete empty chains
    for empty_chain_id in empty_chains:
        # Check if the node exists in the graph
        if empty_chain_id not in G:
            raise ValueError(f"Empty chain {empty_chain_id} not found in graph")

        # Get all in and out edges
        in_edges = list(G.in_edges(empty_chain_id, data=True))
        out_edges = list(G.out_edges(empty_chain_id, data=True))

        # Iterate over all combinations of incoming and outgoing edges to create a new connection
        for u, _, in_data in in_edges:
            for _, w, out_data in out_edges:
                # Update gain attribute to in_data
                if "gain" not in in_data or "gain" not in out_data:
                    raise ValueError("Missing gain attribute in edge data")
                edge_data = dict(in_data)
                edge_data["gain"] = in_data["gain"] * out_data["gain"]

                # Preserve the incoming signal role while composing gains.
                G.add_edge(u, w, **edge_data)
        G.remove_node(empty_chain_id)

    # Check for cycles in the graph
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(
                f"ERROR: Cycles detected in graph created from metadata: {project.input_audios}"
            )
            print(f"Cycles: {cycles}")
            raise ValueError(
                f"Graph contains cycles: {cycles}. This will cause errors in topological sorting. Check metadata!"
            )
    except Exception as e:
        print(f"Error checking for cycles: {e}")
        raise

    return G


# Helper function to prepare a single batch of tasks for rendering
def prepare_batch(
    batch_idx: int,
    tasks_in_batch: list,
    projects: List[Project],
    current_layer: int,
    chain_outputs_reference: dict,
    predecessors: dict,
    num_splitter_tasks: dict,
    global_tmp_dir: str,
) -> tuple:
    """
    Prepare a single batch of tasks for rendering.

    Returns:
        The batch inputs, outputs, FX chains, and resulting chain-output map.

    Raises:
        RuntimeError: If any task cannot be prepared. A partial render batch is
            unsafe because its outputs no longer correspond to its task list.
    """
    batch_inputs = []
    batch_outputs = []
    batch_fx_chains = []
    local_chain_outputs = {}
    splitter_stem_indices = defaultdict(lambda: -1)

    for prev_task_id, task_id, next_task_id in zip(
        [None] + tasks_in_batch[:-1],
        tasks_in_batch,
        tasks_in_batch[1:] + [None],
    ):
        proj_idx, chain_idx = task_id
        project = projects[proj_idx]
        chain = project.FxChains[chain_idx]

        # Determine input path for this task
        if current_layer == 0:
            if (
                prev_task_id == task_id
            ):  # For splitter handling, receiving tracks don't need input audio
                input_path = None
            else:
                # Input chains use their original audio path stored earlier - no change needed
                input_path = chain_outputs_reference[task_id]
                if not os.path.exists(input_path):
                    raise FileNotFoundError(
                        f"Input audio file not found for task {task_id}: {input_path}"
                    )
        else:
            # Use the 'predecessors' dictionary built earlier
            task_predecessors = predecessors.get(
                task_id, {}
            )  # Get predecessors for the current task

            if len(task_predecessors) == 0:
                raise RuntimeError(
                    f"Task {task_id} in layer {current_layer} has no predecessors"
                )

            if (
                prev_task_id == task_id
            ):  # Special handling for splitter receiving tracks
                input_path = None
            else:  # len(task_predecessors) >= 1
                # Mix all predecessors
                predecessor_paths = []
                for pred_task_id in task_predecessors:
                    if pred_task_id not in chain_outputs_reference:
                        raise RuntimeError(
                            f"Output from predecessor {pred_task_id} is missing "
                            f"while preparing {task_id}"
                        )

                    # Check if the predecessor is a splitter
                    if pred_task_id in num_splitter_tasks:
                        # Determine which output of the splitter this task should use
                        pred_chain = projects[pred_task_id[0]].FxChains[pred_task_id[1]]
                        # Safely convert keys to strings for comparison
                        try:
                            next_chain_keys = [
                                str(key) for key in pred_chain.next_chains.keys()
                            ]
                            output_index = next_chain_keys.index(str(chain_idx))
                            if (
                                output_index
                                >= len(chain_outputs_reference[pred_task_id])
                                or chain_outputs_reference[pred_task_id][output_index]
                                is None
                            ):
                                raise IndexError("Output not available")
                            pred_output = chain_outputs_reference[pred_task_id][
                                output_index
                            ]
                        except (IndexError, ValueError) as exc:
                            raise RuntimeError(
                                f"Splitter output from {pred_task_id} to {task_id} "
                                "is not available"
                            ) from exc
                    else:
                        # Regular task, just get the output directly
                        pred_output = chain_outputs_reference[pred_task_id]

                    predecessor_paths.append(pred_output)

                if len(predecessor_paths) != len(task_predecessors):
                    raise RuntimeError(
                        f"Expected {len(task_predecessors)} predecessor outputs for "
                        f"{task_id}, found {len(predecessor_paths)}"
                    )

                # Define path for the mixed file
                proj_tmp_dir = os.path.join(global_tmp_dir, f"proj_{proj_idx}_output")
                os.makedirs(proj_tmp_dir, exist_ok=True)
                mixed_input_path = os.path.join(
                    proj_tmp_dir, f"input_chain_{chain_idx}_mixed.wav"
                )

                # Edge gains have already been applied as REAPER track gains
                # when each predecessor was rendered. Applying weights here
                # would double-scale the signal.
                mix_audio_files(predecessor_paths, mixed_input_path)
                input_path = mixed_input_path
                logging.info(
                    "Successfully mixed %d inputs for task %s",
                    len(predecessor_paths),
                    task_id,
                )

        # Define temporary output path for this chain's result
        # Use a subfolder per task for intermediate files
        proj_tmp_dir = os.path.join(global_tmp_dir, f"proj_{proj_idx}_output")
        os.makedirs(proj_tmp_dir, exist_ok=True)
        if (
            task_id == next_task_id and prev_task_id != task_id
        ):  # If this is a splitter, the source track does not have output audio
            output_path = None
            local_chain_outputs[task_id] = [None] * num_splitter_tasks[
                task_id
            ]  # Initialize with None for each output of splitters
        elif prev_task_id == task_id:  # Receiving track of a splitter
            splitter_stem_indices[task_id] += 1
            stem_id = splitter_stem_indices[task_id]
            output_path = os.path.join(
                proj_tmp_dir, f"output_chain_{chain_idx}_stem_{stem_id}.wav"
            )
            local_chain_outputs[task_id][stem_id] = output_path
        else:
            output_path = os.path.join(proj_tmp_dir, f"output_chain_{chain_idx}.wav")
            local_chain_outputs[task_id] = output_path

        # Prepare FxChain list (convert FXSetting objects to dicts)
        fx_chain_for_render = []
        for fx_setting in chain.FxChain:
            if isinstance(fx_setting.params, list) or isinstance(
                fx_setting.params, dict
            ):
                params_for_render = fx_setting.params  # Pass the list directly
            # elif isinstance(fx_setting.params, dict):
            #     params_for_render = list(fx_setting.params.values) # Pass the dict
            else:
                raise TypeError(
                    f"Invalid params type for FX {fx_setting.fx_name} in task "
                    f"{task_id}: expected list or dict"
                )

            fx_chain_for_render.append(
                {
                    "fx_name": fx_setting.fx_name,
                    "fx_type": fx_setting.fx_type,
                    "params": params_for_render,
                    "n_inputs": fx_setting.n_inputs,
                    "n_outputs": fx_setting.n_outputs,
                    "sidechain_input": fx_setting.sidechain_input,
                }
            )

        batch_inputs.append(input_path)
        batch_outputs.append(output_path)
        batch_fx_chains.append(fx_chain_for_render)

    if len(batch_inputs) != len(tasks_in_batch):
        raise RuntimeError(
            f"Batch {batch_idx} preparation produced {len(batch_inputs)} entries "
            f"for {len(tasks_in_batch)} tasks"
        )
    return batch_inputs, batch_outputs, batch_fx_chains, local_chain_outputs


# an unselected module is needed
def process_layer_with_tracksend_awareness(
    tasks_in_layer: list, projects: List[Project], batch_size: int
) -> tuple:
    """
    Groups tasks in a layer based on sidechain dependencies and splitter requirements.

    Args:
        tasks_in_layer: List of task_ids in the current layer
        projects: List of Project objects

    Returns:
        Tuple of (batches, batch_send_maps, batch_splitter_tracks) where:
        - batches: List of lists, each containing task_ids for one batch
        - batch_send_maps: List of dictionaries mapping source indices to destination indices
        - batch_splitter_tracks: List of lists, each containing indices of splitter source tracks in the batch
    """
    # 1. Build undirected dependency graph for sidechain relationships
    sidechain_graph = defaultdict(set)

    # Track tasks with splitters (multiple outputs)
    num_splitter_tasks = {}  # Maps task_id -> num_outputs

    # Identify all sidechain relationships and splitters within this layer
    for task_id in tasks_in_layer:
        proj_idx, chain_idx = task_id
        chain = projects[proj_idx].FxChains[chain_idx]

        # Check for splitters
        if len(chain.next_chains) > 1:
            # Verify if last FX is actually a splitter
            if chain.FxChain and chain.FxChain[-1].fx_type == "splitter":
                num_splitter_tasks[task_id] = len(chain.next_chains)
            else:
                raise ValueError(
                    f"Task {task_id} has multiple next_chains but does not end "
                    "with a splitter"
                )

        # Check sidechain dependencies
        for fx_setting in chain.FxChain:
            if fx_setting.sidechain_input is not None:
                sc_source_idx = fx_setting.sidechain_input
                sc_source_task_id = (proj_idx, sc_source_idx)
                if sc_source_task_id not in tasks_in_layer:
                    raise ValueError(
                        f"Sidechain source {sc_source_task_id} is not in the same "
                        f"render layer as consumer {task_id}"
                    )
                # Create bidirectional connection (both must be in same batch)
                sidechain_graph[task_id].add(sc_source_task_id)
                sidechain_graph[sc_source_task_id].add(task_id)

    # 2. Find connected components (groups that must be processed together)
    def find_connected_components():
        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in sidechain_graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for task_id in tasks_in_layer:
            if task_id not in visited:
                component = []
                dfs(task_id, component)
                components.append(component)

        return components

    # Get groups of tasks that need to be processed together
    task_groups = find_connected_components()

    def needs_render_tail(group: list) -> bool:
        return any(
            fx.fx_type in TAIL_FX_TYPES
            for task_id in group
            for fx in projects[task_id[0]].FxChains[task_id[1]].FxChain
        )

    # Keep tail and non-tail tracks in separate REAPER render commands while
    # preserving the original order within each category.
    task_groups.sort(key=needs_render_tail)

    # 3. Form batches that respect the connected components
    batches = []
    batches_gain = []
    batch_send_maps = []
    batch_splitter_tracks = []  # One list of splitter tracks per batch
    current_batch = []
    current_batch_gain = []
    current_batch_send_map = {}
    current_batch_splitter_tracks = (
        []
    )  # Tracks splitter source tracks for current batch
    current_batch_size = 0
    current_batch_needs_tail = None

    for group in task_groups:
        # Create a new group that will include duplicated entries for splitters
        expanded_group = []
        expanded_group_gain = []
        original_to_expanded_indices = (
            {}
        )  # Maps original index -> list of expanded indices
        group_splitter_tracks = []  # Collect splitter source tracks for this group
        group_needs_tail = needs_render_tail(group)

        # First pass: Build expanded group with splitter duplications
        expanded_idx = 0
        for i, task_id in enumerate(group):
            original_to_expanded_indices[i] = [expanded_idx]
            expanded_group.append(task_id)

            proj_idx, chain_idx = task_id
            gain_list = list(
                projects[proj_idx].FxChains[chain_idx].next_chains.values()
            )
            if task_id in num_splitter_tasks:
                # The first track hosts the multi-output splitter and is not
                # rendered. Each output needs a distinct receiver track.
                expanded_group_gain.append(1.0)
            else:
                expanded_group_gain.append(gain_list[0] if gain_list else 1.0)
            expanded_idx += 1

            # If this is a splitter task, duplicate it
            if task_id in num_splitter_tasks:
                num_outputs = num_splitter_tasks[task_id]
                # Add one receiver track per splitter output. The original is
                # the hidden source track, so a N-band splitter needs N+1 tracks.
                for j in range(num_outputs):
                    expanded_group.append(task_id)
                    expanded_group_gain.append(gain_list[j])
                    original_to_expanded_indices[i].append(expanded_idx)
                    expanded_idx += 1

        # Build send map based on sidechain relationships and splitters
        send_map = defaultdict(list)

        for i, task_id in enumerate(group):
            proj_idx, chain_idx = task_id
            chain = projects[proj_idx].FxChains[chain_idx]

            # Handle sidechain relationships
            for fx_setting in chain.FxChain:
                if fx_setting.sidechain_input is not None:
                    sc_source_idx = fx_setting.sidechain_input
                    sc_source_task_id = (proj_idx, sc_source_idx)
                    if sc_source_task_id in group:
                        # Get the source chain to verify it doesn't end with a splitter
                        source_chain = projects[proj_idx].FxChains[sc_source_idx]

                        # Skip if source chain ends with a splitter (should be caught by validation,
                        # but adding this check for robustness)
                        if (
                            source_chain.FxChain
                            and source_chain.FxChain[-1].fx_type == "splitter"
                        ):
                            raise ValueError(
                                f"Chain {chain_idx} uses splitter chain "
                                f"{sc_source_idx} as a sidechain source"
                            )

                        # Find source index in group
                        source_idx = group.index(sc_source_task_id)
                        # Map source to destination in expanded indices
                        for src_exp_idx in original_to_expanded_indices[source_idx]:
                            # Send to the first track only - sufficient for sidechain input
                            send_map[src_exp_idx].append(
                                original_to_expanded_indices[i][0]
                            )

            # Handle splitter outputs
            if task_id in num_splitter_tasks:
                source_idx = i
                source_track_idx = original_to_expanded_indices[source_idx][
                    0
                ]  # Get the source track index
                group_splitter_tracks.append(
                    source_track_idx
                )  # Add to this group's splitter tracks
                for exp_idx in original_to_expanded_indices[source_idx][
                    1:
                ]:  # Skip first (original)
                    send_map[source_track_idx].append(exp_idx)

        # Check if expanded group can fit in current batch
        group_size = len(expanded_group)
        if group_size > batch_size:
            raise ValueError(
                f"Group requires {group_size} slots (exceeds batch_size={batch_size}). Consider increasing batch size or simplifying dependencies."
            )

        # If we can add this group to current batch
        if current_batch_size + group_size <= batch_size and (
            current_batch_needs_tail is None
            or current_batch_needs_tail == group_needs_tail
        ):
            # Update send map indices to account for offset
            offset_send_map = {
                (key + current_batch_size): [val + current_batch_size for val in values]
                for key, values in send_map.items()
            }

            # Update splitter track indices to account for offset
            offset_splitter_tracks = [
                track_idx + current_batch_size for track_idx in group_splitter_tracks
            ]

            # Extend current batch and update maps
            current_batch.extend(expanded_group)
            current_batch_gain.extend(expanded_group_gain)
            current_batch_send_map.update(offset_send_map)
            current_batch_splitter_tracks.extend(offset_splitter_tracks)
            current_batch_size += group_size
            current_batch_needs_tail = group_needs_tail
        else:
            # Start a new batch
            if current_batch:
                batches.append(current_batch)
                batches_gain.append(current_batch_gain)
                batch_send_maps.append(current_batch_send_map)
                batch_splitter_tracks.append(current_batch_splitter_tracks)

            # Reset for the new batch
            current_batch = expanded_group
            current_batch_gain = expanded_group_gain
            current_batch_send_map = (
                send_map  # send_map is already correct for the start of a batch
            )
            current_batch_splitter_tracks = group_splitter_tracks.copy()
            current_batch_size = group_size
            current_batch_needs_tail = group_needs_tail

    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)
        batches_gain.append(current_batch_gain)
        batch_send_maps.append(current_batch_send_map)
        batch_splitter_tracks.append(current_batch_splitter_tracks)

    return (
        batches,
        batches_gain,
        batch_send_maps,
        batch_splitter_tracks,
        num_splitter_tasks,
    )


def process_final_output(
    proj_idx: int,
    offset: int,
    current_batch_projects: List[Project],
    chain_outputs: dict,
    processed_chains: set,
    available_plugins_param_names: Dict[str, List[str]],
    final_output_dir: str,
    save_mode: str,
    save_compression_rate: int,
) -> bool:
    """Export one project and publish its directory atomically."""
    staging_dir: Optional[str] = None
    try:
        project = current_batch_projects[proj_idx]
        # Find the final chain index (empty next_chains)
        final_chain_idx = next(
            (
                i
                for i, chain_def in enumerate(project.FxChains)
                if not chain_def.next_chains
            ),
            None,
        )

        if final_chain_idx is None:
            logging.error(f"No final chain found for Project {(proj_idx+offset)}")
            return False

        final_task_id = (proj_idx, final_chain_idx)
        final_output_dir_proj = os.path.join(
            final_output_dir, f"project_{(proj_idx+offset):08d}"
        )
        os.makedirs(final_output_dir, exist_ok=True)
        if os.path.exists(final_output_dir_proj):
            raise FileExistsError(
                f"Refusing to overwrite existing project output: "
                f"{final_output_dir_proj}"
            )
        staging_dir = tempfile.mkdtemp(
            prefix=f".project_{(proj_idx + offset):08d}.",
            dir=final_output_dir,
        )

        # Process final output
        success = True
        save_training = save_mode in {"training-ready", "both"}
        save_human = save_mode in {"human-readable", "both"}
        if not save_training and not save_human:
            raise ValueError(f"Unsupported save mode: {save_mode}")
        if final_task_id in chain_outputs and final_task_id in processed_chains:
            temp_output_path = chain_outputs[final_task_id]
            if not isinstance(temp_output_path, str) or not os.path.isfile(
                temp_output_path
            ):
                raise FileNotFoundError(
                    f"Final render is missing for project {proj_idx + offset}: "
                    f"{temp_output_path}"
                )
            processed_audio, processed_sample_rate = sf.read(
                temp_output_path,
                always_2d=True,
                dtype="float32",
            )
            if processed_audio.shape[0] == 0 or processed_audio.shape[1] == 0:
                raise ValueError(f"Rendered audio is empty: {temp_output_path}")
            if not np.all(np.isfinite(processed_audio)):
                raise ValueError(
                    f"Rendered audio contains non-finite samples: {temp_output_path}"
                )
            # Determine final output filename
            output_filename = (
                os.path.basename(project.output_audio)
                if project.output_audio
                else "output.wav"
            )
            if output_filename in {
                "",
                ".",
                "..",
                "audio_data.h5",
                "metadata.yaml",
                "mixing_graph.pickle",
                "stems",
            }:
                raise ValueError(
                    f"Unsafe or reserved output filename: {output_filename!r}"
                )
            # Ensure final path is within the project's output dir
            final_output_path = os.path.join(staging_dir, output_filename)

            # try:
            if save_training:
                h5_path = os.path.join(staging_dir, "audio_data.h5")
                h5_temp_path = f"{h5_path}.tmp"
                try:
                    import h5py

                    # Write atomically so a failed export cannot leave a file
                    # that looks valid to a downstream training job.
                    with h5py.File(h5_temp_path, "w") as h5f:
                        # Load and store processed audio
                        output = h5f.create_dataset(
                            "output",
                            data=processed_audio.T,
                            compression="gzip",
                            compression_opts=save_compression_rate,
                        )
                        output.attrs["sample_rate"] = processed_sample_rate

                        # Store all input stems
                        stems_group = h5f.create_group("stems")
                        stems_group.attrs["n_stems"] = len(project.input_audios)
                        for i, ia in enumerate(project.input_audios):
                            stem_audio, sr = sf.read(
                                ia.audio_path, always_2d=True, dtype="float32"
                            )
                            if not np.all(np.isfinite(stem_audio)):
                                raise ValueError(
                                    f"Stem contains non-finite samples: {ia.audio_path}"
                                )
                            stem = stems_group.create_dataset(
                                f"input_audio_{i}",
                                data=stem_audio.T,
                                compression="gzip",
                                compression_opts=save_compression_rate,
                            )
                            stem.attrs["index"] = i
                            stem.attrs["sample_rate"] = sr
                            stem.attrs["audio_path"] = ia.audio_path
                            stem.attrs["audio_type"] = ia.audio_type
                            stem.attrs["input_FxChain"] = str(ia.input_FxChain)
                    os.replace(h5_temp_path, h5_path)
                    logging.info(
                        f"Saved H5 training data for Project {(proj_idx+offset)} to: {h5_path}"
                    )
                except Exception as h5_err:
                    logging.error(
                        f"Failed to create H5 file for Project {(proj_idx+offset)}: {h5_err}"
                    )
                    if os.path.exists(h5_temp_path):
                        os.remove(h5_temp_path)
                    success = False

                # Save the NetworkX graph in multiple formats
                pickle_path = os.path.join(staging_dir, "mixing_graph.pickle")
                pickle_temp_path = f"{pickle_path}.tmp"
                try:
                    G = metadata_to_networkx(
                        current_batch_projects[proj_idx],
                        available_plugins_param_names,
                    )
                    # Save as pickle (best for preserving Python data types)
                    with open(pickle_temp_path, "wb") as f:
                        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
                    os.replace(pickle_temp_path, pickle_path)
                    logging.info(
                        f"Saved graph for Project {(proj_idx+offset)} to: {pickle_path}"
                    )
                except Exception as graph_err:
                    logging.error(
                        f"Failed to save graph for Project {(proj_idx+offset)}: {graph_err}"
                    )
                    if os.path.exists(pickle_temp_path):
                        os.remove(pickle_temp_path)
                    success = False
            if save_human:
                try:
                    # Save final output audio file
                    shutil.copy2(temp_output_path, final_output_path)
                    # Save the original input audio files
                    for i, ia in enumerate(project.input_audios):
                        # input_basename = os.path.basename(ia.audio_path)
                        source_suffix = os.path.splitext(ia.audio_path)[1] or ".wav"
                        dest_path = os.path.join(
                            staging_dir,
                            "stems",
                            f"input_audio_{i}{source_suffix}",
                        )
                        os.makedirs(
                            os.path.join(staging_dir, "stems"), exist_ok=True
                        )  # Create stems directory
                        # dest_path = os.path.join(final_output_dir_proj, "stems", f"{ia.audio_type.lower()}.wav")
                        shutil.copy2(ia.audio_path, dest_path)
                    # Save project metadata using the built-in method
                    metadata_path = os.path.join(staging_dir, "metadata.yaml")
                    Project.save_to_yaml([project], metadata_path)
                except PermissionError:
                    logging.warning(
                        f"Permission denied when copying {ia.audio_path}. File may already exist with restricted permissions."
                    )
                    success = False
                except Exception as e:
                    logging.error(
                        f"Failed to move audio for Project {(proj_idx+offset)}: {e}"
                    )
                    success = False

            # except Exception as e:
            #     logging.error(f"Failed to create final output files for Project {(proj_idx+offset)}: {e}")
            #     success = False

            if success:
                os.replace(staging_dir, final_output_dir_proj)
                staging_dir = None
                logging.info(
                    "Published Project %d atomically to %s",
                    proj_idx + offset,
                    final_output_dir_proj,
                )
        else:
            logging.error(
                f"Final output chain {final_task_id} for Project {(proj_idx+offset)} was not processed or output file is missing."
            )
            success = False
        return success

    except Exception as e:
        logging.error(
            f"Error processing final output for Project {(proj_idx+offset)}: {e}"
        )
        return False
    finally:
        if staging_dir is not None and os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)


def mix_audio_files(
    input_files: List[str],
    output_path: str,
    channels: Optional[int] = None,
    sample_rate: Optional[int] = None,
    clipping_prevent_mode: Optional[str] = None,
    weights: Optional[Sequence[float]] = None,
    trim_silence: bool = False,
    output_subtype: str = "FLOAT",
) -> None:
    """
    Loads multiple audio files, mixes them, and saves the result.

    Args:
        input_files: List of paths to input audio files.
        output_path: Path to save the mixed audio file.
        channels: Required channel count, or inferred from the first input.
        sample_rate: Required sample rate, or inferred from the first input.
        clipping_prevent_mode: Optional ``normalize`` or ``hard_clip`` policy.
        weights: Optional linear gain for each input.
        trim_silence: Whether to remove exact trailing zeros.
        output_subtype: SoundFile output subtype. ``FLOAT`` avoids hidden clipping
            and quantization in intermediate files.

    Raises:
        ValueError: If input files have different sample rates or channel counts.
        FileNotFoundError: If any input file is not found.
    """
    if not input_files:
        raise ValueError("mix_audio_files requires at least one input file")
    if weights is None:
        weights = [1.0] * len(input_files)
    if len(weights) != len(input_files):
        raise ValueError(
            f"Expected {len(input_files)} weights, received {len(weights)}"
        )
    if not all(np.isfinite(float(weight)) for weight in weights):
        raise ValueError("Mix weights must be finite numbers")

    audio_data = []
    max_len = 0

    # Load audio files and find max length, check consistency
    for file_path, weight in zip(input_files, weights):
        try:
            data, sr = sf.read(
                file_path, always_2d=True
            )  # Read as 2D array (samples, channels)
        except FileNotFoundError:
            logging.error(f"Input file not found during mixing: {file_path}")
            raise
        except Exception as e:
            logging.error(f"Error reading audio file {file_path}: {e}")
            raise

        if not np.all(np.isfinite(data)):
            raise ValueError(f"Input audio contains non-finite samples: {file_path}")

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch: Expected {sample_rate}, got {sr} for {file_path}"
            )

        if channels is None:
            channels = data.shape[1]
        elif data.shape[1] != channels:
            # Attempt mono-to-stereo conversion if needed, or raise error
            if channels == 2 and data.shape[1] == 1:
                logging.warning(
                    f"Converting mono file {file_path} to stereo for mixing."
                )
                data = np.repeat(data, 2, axis=1)  # Duplicate mono channel
            elif channels == 1 and data.shape[1] == 2:
                logging.warning(
                    f"Converting stereo file {file_path} to mono (averaging) for mixing."
                )
                data = np.mean(data, axis=1, keepdims=True)
            else:
                raise ValueError(
                    f"Channel count mismatch: Expected {channels}, got {data.shape[1]} for {file_path}"
                )

        audio_data.append(data * float(weight))
        if data.shape[0] > max_len:
            max_len = data.shape[0]

    # Pad shorter files to the length of the longest one
    padded_audio_data = [
        np.pad(data, ((0, max_len - data.shape[0]), (0, 0)), mode="constant")
        for data in audio_data
    ]

    # Sum all padded audio data together in one go
    mixed_signal = np.sum(padded_audio_data, axis=0)

    if trim_silence:
        nonzero_mask = np.any(mixed_signal != 0, axis=1)
        if np.any(nonzero_mask):
            last_nonzero_idx = int(np.max(np.nonzero(nonzero_mask)[0]))
            mixed_signal = mixed_signal[: last_nonzero_idx + 1]
        else:
            mixed_signal = mixed_signal[:1]

    peak = float(np.max(np.abs(mixed_signal)))
    if clipping_prevent_mode == "normalize" and peak > 1.0:
        mixed_signal /= peak
    elif clipping_prevent_mode == "hard_clip":
        mixed_signal = np.clip(mixed_signal, -1.0, 1.0)
    elif clipping_prevent_mode is not None:
        raise ValueError(
            "clipping_prevent_mode must be None, 'normalize', or 'hard_clip'"
        )
    elif peak > 1.0:
        logging.warning(
            "Mixed signal peak %.3f exceeds full scale in %s; preserving it in "
            "floating-point output",
            peak,
            output_path,
        )

    try:
        sf.write(output_path, mixed_signal, sample_rate, subtype=output_subtype)
        logging.debug(f"Successfully mixed {len(input_files)} files to {output_path}")
    except Exception as e:
        logging.error(f"Error writing mixed audio file {output_path}: {e}")
        raise


# For each layer, track tasks that need to be processed together due to sidechain dependencies
# def process_layer_with_sidechain_awareness(tasks_in_layer, projects):
#     # 1. Build undirected dependency graph for sidechain relationships
#     sidechain_graph = defaultdict(set)

#     # Identify all sidechain relationships within this layer
#     for task_id in tasks_in_layer:
#         proj_idx, chain_idx = task_id
#         chain = projects[proj_idx].FxChains[chain_idx]

#         # Check if this chain requires any sidechains from the same layer
#         for fx_setting in chain.FxChain:
#             if fx_setting.sidechain_input is not None:
#                 sc_source_task_id = (proj_idx, fx_setting.sidechain_input)
#                 if sc_source_task_id in tasks_in_layer:
#                     # Create bidirectional connection (both must be in same batch)
#                     sidechain_graph[task_id].add(sc_source_task_id)
#                     sidechain_graph[sc_source_task_id].add(task_id)

#     # 2. Find connected components (groups that must be processed together)
#     def find_connected_components():
#         visited = set()
#         components = []

#         def dfs(node, component):
#             visited.add(node)
#             component.append(node)
#             for neighbor in sidechain_graph[node]:
#                 if neighbor not in visited:
#                     dfs(neighbor, component)

#         for task_id in tasks_in_layer:
#             if task_id not in visited:
#                 component = []
#                 dfs(task_id, component)
#                 components.append(component)

#         return components

#     # Get groups of tasks that need to be processed together
#     task_groups = find_connected_components()

#     # 3. Form batches that respect the connected components
#     batches = []
#     batches_send_map = [] # send key to values
#     current_batch = []
#     current_batch_send_map = {} # send key to values
#     current_batch_size = 0

#     for group in task_groups:
#         send_map = defaultdict(list) # send key to values
#         for i, task_id in enumerate(group):
#             num_next_chains = len(projects[proj_idx].FxChains[chain_idx].next_chains)
#             if num_next_chains > 1:
#                 group[i:i] = [task_id] * num_next_chains # Add to group if it has multiple next chains because of splitters
#                 send_map[i].append([j+1 for j in range(num_next_chains)]) # Add to send_map

#         for i, task_id in enumerate(group):
#             chain = projects[proj_idx].FxChains[chain_idx].FxChain
#             for fx_setting in chain:
#                 if fx_setting.sidechain_input is not None:
#                     sc_source_task_id = (proj_idx, fx_setting.sidechain_input)
#                     if sc_source_task_id not in group:
#                         raise ValueError(f"Sidechain source {sc_source_task_id} not in group {group}.")
#                     else:
#                         send_map[group.index(sc_source_task_id)].append(i) # Add to send_map

#         # If this group can fit in current batch
#         if len(group) > batch_size:
#             raise ValueError(f"Group {group} requires more CPU cores than available ({batch_size}).")
#         if current_batch_size + len(group) <= batch_size:
#             current_batch_send_map.update({(key+current_batch_size):([value+current_batch_size for value in values]) for key, values in send_map.items()}) # Update send map indices to account for offset
#             current_batch.extend(group)
#             current_batch_size += len(group)
#         else:
#             # Finish current batch and start a new one
#             if current_batch:
#                 batches.append(current_batch)
#                 batches_send_map.append(current_batch_send_map)
#             current_batch = group
#             current_batch_send_map = send_map
#             current_batch_size = len(group)

#     # Add the last batch if not empty
#     if current_batch:
#         batches.append(current_batch)

#     return batches

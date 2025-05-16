import os
import logging
import numpy as np
import soundfile as sf
from collections import defaultdict
from typing import List, Dict
import networkx as nx
import h5py
import pickle
import shutil
from utils.data_class import Project, InputAudio

def metadata_to_networkx(project: Project, available_plugins_param_names: Dict[str, List[str]]):
    """
    The graph implements three edge types: 
    1. Gain edges (standard connections with mix weighting)
    2. Split edges (multi-output connections from splitter plugins)
    3. Sidechain edges (special inputs for sidechain-enabled effects)
    Data structure supports direct serialization to/from YAML for project definition
    """

    # Create a directed graph
    G = nx.DiGraph()
    # Track chains that need direct connections (empty chains)
    empty_chains = []
    # Sidechain input map: list of triplets (sidechain_input_chain_id, sidechain_input_node_id, sidechain_output_chain_id
    sidechain_input_map = []
    # Create FX Chains
    for i, chain_def in enumerate(project.FxChains):
        if chain_def.FxChain:
            for j, fx in enumerate(chain_def.FxChain):
                if fx.sidechain_input:
                    sidechain_input_map.append((i,j,fx.sidechain_input))
                G.add_node(
                    f"fx_{i}_{j}", 
                    type='fx', 
                    label=fx.fx_type, 
                    instance=fx.fx_name, 
                    params={name: value for name, value in zip(available_plugins_param_names[fx.fx_name], fx.params) if value is not None}
                    )
                if j > 0:
                    # Connect to previous FX in the chain
                    G.add_edge(f"fx_{i}_{j-1}", 
                               f"fx_{i}_{j}", 
                               type='send_signal', 
                               label='main', 
                               gain = 1.0) # default gain of 1.0
        else: 
            # Create a empty node for empty chains
            G.add_node(f"fx_{i}_0")
            empty_chains.append(f"fx_{i}_0")
            
        # Connect to next chains (networkx allows creating empty nodes by connecting to non-existing nodes)
        if len(chain_def.next_chains) != 0: # not end node
            if len(chain_def.next_chains) > 1: # splitter
                edge_type = f'split_signal'
                edge_labels = [f"band_{i}" for i in range(len(chain_def.next_chains))]
            else: # normal connection, only one target
                edge_type = 'send_signal'
                edge_labels = ['main']
            for (target_idx, gain), edge_label in zip(chain_def.next_chains.items(), edge_labels):
                G.add_edge(f"fx_{i}_{len(chain_def.FxChain)-1}", 
                           f"fx_{target_idx}_{0}", 
                           type=edge_type, 
                           label=edge_label, 
                           gain=gain)

    # Connect sidechain input nodes to fx chain nodes
    for sidechain_input_chain_id, sidechain_input_node_id, sidechain_output_chain_id in sidechain_input_map:
        G.add_edge(
            f'fx_{sidechain_output_chain_id}_{len(project.FxChains[i].FxChain)-1}', 
            f'fx_{sidechain_input_chain_id}_{sidechain_input_node_id}', 
            type='send_signal', 
            label='control', 
            gain=1.0
            )
        
    # Create input audio nodes
    for i, ia in enumerate(project.input_audios):
        G.add_node(f'input_audio_{i}', type='audio', label=ia.audio_type, instance=ia.audio_path)
        G.add_edge(f'input_audio_{i}', f'fx_{ia.input_FxChain}_0', type='send_signal', label='main', gain=1.0)
    
    # Delete empty chains
    for empty_chain_id in empty_chains:
        # Check if the node exists in the graph
        if empty_chain_id not in G:
            raise ValueError(f"Empty chain {empty_chain_id} not found in graph")
        
        # Get all in and out edges
        in_edges = list(G.in_edges(empty_chain_id, data = True))
        out_edges = list(G.out_edges(empty_chain_id, data = True))
        
        # Iterate over all combinations of incoming and outgoing edges to create a new connection
        for u, _, in_data in in_edges:
            for _, w, out_data in out_edges:
                # Update gain attribute to in_data
                if 'gain' not in in_data or 'gain' not in out_data:
                    raise ValueError(f"Missing gain attribute in edge data")
                in_data['gain'] = in_data['gain'] * out_data['gain']
                
                # Add new edge (u -> w): use atttributes from the in edge
                G.add_edge(u, w, **in_data)
        G.remove_node(empty_chain_id)
        
    # Check for cycles in the graph
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(f"ERROR: Cycles detected in graph created from metadata: {project.input_audios}")
            print(f"Cycles: {cycles}")
            raise ValueError(f"Graph contains cycles: {cycles}. This will cause errors in topological sorting. Check metadata!")
    except Exception as e:
        print(f"Error checking for cycles: {e}")
        raise

    return G

# Helper function to prepare a single batch of tasks for rendering
def prepare_batch(batch_idx, tasks_in_batch, projects, current_layer, chain_outputs_reference, 
                  predecessors, num_splitter_tasks, global_tmp_dir):
    """
    Prepare a single batch of tasks for rendering.
    
    Returns:
        Tuple of (batch_inputs, batch_outputs, batch_fx_chains) or None if preparation fails
    """
    batch_inputs = []
    batch_outputs = []
    batch_fx_chains = []
    # Create a local copy to work with
    local_chain_outputs = {}
    
    try:
        for prev_task_id, task_id, next_task_id in zip([None]+tasks_in_batch[:-1], tasks_in_batch, tasks_in_batch[1:]+[None]):
            proj_idx, chain_idx = task_id
            project = projects[proj_idx]
            chain = project.FxChains[chain_idx]

            # Determine input path for this task
            if current_layer == 0:
                if prev_task_id == task_id: # For splitter handling, receiving tracks don't need input audio
                    input_path = None
                else:
                    # Input chains use their original audio path stored earlier - no change needed
                    input_path = chain_outputs_reference[task_id]
                    if not os.path.exists(input_path):
                        logging.error(f"Input audio file not found for task {task_id}: {input_path}. Skipping task.")
                        continue
            else:
                # Use the 'predecessors' dictionary built earlier
                task_predecessors = predecessors.get(task_id, {}) # Get predecessors for the current task

                if len(task_predecessors) == 0:
                    # This case should ideally be caught by validation, but handle defensively
                    logging.error(f"Task {task_id} in Layer {current_layer} has no predecessors according to graph structure. Skipping.")
                    continue

                if prev_task_id == task_id: # Special handling for splitter receiving tracks
                    input_path = None
                else: # len(task_predecessors) >= 1
                    # Mix all predecessors
                    weights = []
                    predecessor_paths = []
                    missing_preds = False
                    for pred_task_id, weight in task_predecessors.items():
                        if pred_task_id not in chain_outputs_reference:
                            logging.error(f"Output from predecessor {pred_task_id} not found for mixing into {task_id}. Skipping mix.")
                            missing_preds = True
                            break
                            
                        # Check if the predecessor is a splitter
                        if pred_task_id in num_splitter_tasks:
                            # Determine which output of the splitter this task should use
                            pred_chain = projects[pred_task_id[0]].FxChains[pred_task_id[1]]
                            # Safely convert keys to strings for comparison
                            try:
                                next_chain_keys = [str(key) for key in pred_chain.next_chains.keys()]
                                output_index = next_chain_keys.index(str(chain_idx))
                                if output_index >= len(chain_outputs_reference[pred_task_id]) or chain_outputs_reference[pred_task_id][output_index] is None:
                                    raise IndexError("Output not available")
                                pred_output = chain_outputs_reference[pred_task_id][output_index]
                            except (IndexError, ValueError) as e:
                                logging.error(f"Error accessing splitter output for {pred_task_id} to {task_id}: {e}")
                                missing_preds = True
                                break
                        else:
                            # Regular task, just get the output directly
                            pred_output = chain_outputs_reference[pred_task_id]
                            
                        predecessor_paths.append(pred_output)
                        weights.append(weight)

                    if missing_preds:
                        continue # Skip this task if inputs are missing

                    if len(predecessor_paths) != len(task_predecessors):
                        logging.error(f"Mismatch in predecessor outputs for task {task_id}. Expected {len(task_predecessors)}, found {len(predecessor_paths)}. Skipping task.")
                        continue
                    
                    # Define path for the mixed file
                    proj_tmp_dir = os.path.join(global_tmp_dir, f"proj_{proj_idx}_output")
                    os.makedirs(proj_tmp_dir, exist_ok=True)
                    mixed_input_path = os.path.join(proj_tmp_dir, f"input_chain_{chain_idx}_mixed.wav")

                    try:
                        mix_audio_files(predecessor_paths, weights, mixed_input_path)
                        input_path = mixed_input_path
                        logging.info(f"Successfully mixed {len(predecessor_paths)} inputs for task {task_id}")
                    except Exception as mix_err:
                        logging.error(f"Error mixing audio for task {task_id}: {mix_err}. Skipping task.")
                        continue
                    

            # Define temporary output path for this chain's result
            # Use a subfolder per task for intermediate files
            proj_tmp_dir = os.path.join(global_tmp_dir, f"proj_{proj_idx}_output")
            os.makedirs(proj_tmp_dir, exist_ok=True)
            if task_id == next_task_id and prev_task_id != task_id: # If this is a splitter, the source track does not have output audio
                stem_id = -1 # stem index for output of splitters
                output_path = None
                local_chain_outputs[task_id] = [None] * num_splitter_tasks[task_id] # Initialize with None for each output of splitters
            elif prev_task_id == task_id: # Receiving track of a splitter
                stem_id += 1
                output_path = os.path.join(proj_tmp_dir, f"output_chain_{chain_idx}_stem_{stem_id}.wav")
                local_chain_outputs[task_id][stem_id] = output_path
            else:
                output_path = os.path.join(proj_tmp_dir, f"output_chain_{chain_idx}.wav")
                local_chain_outputs[task_id] = output_path


            # Prepare FxChain list (convert FXSetting objects to dicts)
            fx_chain_for_render = []
            for fx_setting in chain.FxChain:
                if isinstance(fx_setting.params, list) or isinstance(fx_setting.params, dict):
                    params_for_render = fx_setting.params # Pass the list directly
                # elif isinstance(fx_setting.params, dict):
                #     params_for_render = list(fx_setting.params.values) # Pass the dict
                else:
                    logging.error(f"Invalid params type for FX {fx_setting.fx_name} in task {task_id}. Expected list or dict.")

                fx_chain_for_render.append({
                    "fx_name": fx_setting.fx_name,
                    "fx_type": fx_setting.fx_type,
                    "params": params_for_render,
                    "n_inputs": fx_setting.n_inputs,
                    "n_outputs": fx_setting.n_outputs,
                    "sidechain_input": fx_setting.sidechain_input
                })


            batch_inputs.append(input_path)
            batch_outputs.append(output_path)
            batch_fx_chains.append(fx_chain_for_render)            
        return batch_inputs, batch_outputs, batch_fx_chains, local_chain_outputs
    except Exception as e:
        logging.error(f"Error preparing batch {batch_idx}: {e}")
        return None

# an unselected module is needed
def process_layer_with_tracksend_awareness(tasks_in_layer, projects, batch_size):
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
                logging.warning(f"Task {task_id} has multiple next_chains but last FX is not a splitter")
        
        # Check sidechain dependencies
        for fx_setting in chain.FxChain:
            if fx_setting.sidechain_input is not None:
                sc_source_idx = fx_setting.sidechain_input
                sc_source_task_id = (proj_idx, sc_source_idx)
                if sc_source_task_id in tasks_in_layer:
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
    
    # 3. Form batches that respect the connected components
    batches = []
    batch_send_maps = []
    batch_splitter_tracks = []  # One list of splitter tracks per batch
    current_batch = []
    current_batch_send_map = {}
    current_batch_splitter_tracks = []  # Tracks splitter source tracks for current batch
    current_batch_size = 0
    
    for group in task_groups:
        # Create a new group that will include duplicated entries for splitters
        expanded_group = []
        original_to_expanded_indices = {}  # Maps original index -> list of expanded indices
        group_splitter_tracks = []  # Collect splitter source tracks for this group
        
        # First pass: Build expanded group with splitter duplications
        expanded_idx = 0
        for i, task_id in enumerate(group):
            original_to_expanded_indices[i] = [expanded_idx]
            expanded_group.append(task_id)
            expanded_idx += 1
            
            # If this is a splitter task, duplicate it
            if task_id in num_splitter_tasks:
                num_outputs = num_splitter_tasks[task_id]
                # Add duplicates (one per output minus the original)
                for _ in range(num_outputs):
                    expanded_group.append(task_id)
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
                        if source_chain.FxChain and source_chain.FxChain[-1].fx_type == "splitter":
                            logging.warning(f"Skipping sidechain connection: Chain {chain_idx} trying to use chain {sc_source_idx} as sidechain input, but that chain ends with a splitter plugin.")
                            continue
                            
                        # Find source index in group
                        source_idx = group.index(sc_source_task_id)
                        # Map source to destination in expanded indices
                        for src_exp_idx in original_to_expanded_indices[source_idx]:
                            # Send to the first track only - sufficient for sidechain input
                            send_map[src_exp_idx].append(original_to_expanded_indices[i][0])
                            
            # Handle splitter outputs
            if task_id in num_splitter_tasks:
                source_idx = i
                source_track_idx = original_to_expanded_indices[source_idx][0]  # Get the source track index
                group_splitter_tracks.append(source_track_idx)  # Add to this group's splitter tracks
                for exp_idx in original_to_expanded_indices[source_idx][1:]:  # Skip first (original)
                    send_map[source_track_idx].append(exp_idx)
        
        # Check if expanded group can fit in current batch
        group_size = len(expanded_group)
        if group_size > batch_size:
            raise ValueError(f"Group requires {group_size} slots (exceeds batch_size={batch_size}). Consider increasing batch size or simplifying dependencies.")
        
        # If we can add this group to current batch
        if current_batch_size + group_size <= batch_size:
            # Update send map indices to account for offset
            offset_send_map = {
                (key + current_batch_size): [val + current_batch_size for val in values]
                for key, values in send_map.items()
            }
            
            # Update splitter track indices to account for offset
            offset_splitter_tracks = [track_idx + current_batch_size for track_idx in group_splitter_tracks]
            
            # Extend current batch and update maps
            current_batch.extend(expanded_group)
            current_batch_send_map.update(offset_send_map)
            current_batch_splitter_tracks.extend(offset_splitter_tracks)
            current_batch_size += group_size
        else:
            # Start a new batch
            if current_batch:
                batches.append(current_batch)
                batch_send_maps.append(current_batch_send_map)
                batch_splitter_tracks.append(current_batch_splitter_tracks)
            current_batch = expanded_group
            current_batch_send_map = send_map
            current_batch_splitter_tracks = group_splitter_tracks.copy()
            current_batch_size = group_size
    
    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)
        batch_send_maps.append(current_batch_send_map)
        batch_splitter_tracks.append(current_batch_splitter_tracks)
    
    return batches, batch_send_maps, batch_splitter_tracks, num_splitter_tasks

def process_final_output(
    proj_idx, 
    offset, 
    current_batch_projects, 
    chain_outputs, 
    processed_chains, 
    available_plugins_param_names, 
    FINAL_OUTPUT_DIR, 
    SAVE_MODE, 
    SAVE_COMPRESSION_RATE
    ):
    """Process final output for a single project"""
    try:
        project = current_batch_projects[proj_idx]
        # Find the final chain index (empty next_chains)
        final_chain_idx = next((i for i, chain_def in enumerate(project.FxChains) if not chain_def.next_chains), None)
        
        if final_chain_idx is None:
            logging.error(f"No final chain found for Project {(proj_idx+offset)}")
            return False
            
        final_task_id = (proj_idx, final_chain_idx)
        final_output_dir_proj = os.path.join(FINAL_OUTPUT_DIR, f"project_{(proj_idx+offset):08d}")
        os.makedirs(final_output_dir_proj, exist_ok=True)
        
        # Process final output
        success = True
        if final_task_id in chain_outputs and final_task_id in processed_chains:
            temp_output_path = chain_outputs[final_task_id]
            # Determine final output filename
            output_filename = os.path.basename(project.output_audio) if project.output_audio else "output.wav"
            # Ensure final path is within the project's output dir
            final_output_path = os.path.join(final_output_dir_proj, output_filename)
            
            # try:
            if "training-ready" in SAVE_MODE:
                try:
                    # Create H5 file path
                    h5_path = os.path.join(final_output_dir_proj,'audio_data.h5')
                    
                    # Create H5 file
                    with h5py.File(h5_path, 'w') as h5f:
                        # Load and store processed audio
                        processed_audio, sr = sf.read(temp_output_path)
                        h5f.create_dataset('output', data=processed_audio, compression="gzip", compression_opts=SAVE_COMPRESSION_RATE)
                        h5f.create_dataset('output_sr', data=sr)
                        
                        # Store all input stems
                        stems_group = h5f.create_group('stems')
                        for i, ia in enumerate(project.input_audios):
                            stem_audio, sr = sf.read(ia.audio_path)
                            stems_group.create_dataset(f"input_audio_{i}", data=stem_audio, compression="gzip", compression_opts=SAVE_COMPRESSION_RATE)
                            stems_group.create_dataset(f"input_audio_{i}_sr", data=sr)
                    logging.info(f"Saved H5 training data for Project {(proj_idx+offset)} to: {h5_path}")
                except ImportError:
                    logging.error("soundfile module not found. Install with 'pip install soundfile' to enable H5 export.")
                except Exception as h5_err:
                    logging.error(f"Failed to create H5 file for Project {(proj_idx+offset)}: {h5_err}")
                    
                # Get NetworkX graph
                G = metadata_to_networkx(current_batch_projects[proj_idx], available_plugins_param_names)
                
                # Save the NetworkX graph in multiple formats
                try:
                    # Save as pickle (best for preserving Python data types)
                    pickle_path = os.path.join(final_output_dir_proj, 'mixing_graph.pickle')
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
                    logging.info(f"Saved graph for Project {(proj_idx+offset)} to: {pickle_path}")
                except Exception as graph_err:
                    logging.error(f"Failed to save graph for Project {(proj_idx+offset)}: {graph_err}")
            if "human-readable" in SAVE_MODE:
                try: 
                    # Save final output audio file
                    shutil.move(temp_output_path, final_output_path)
                    logging.info(f"Saved final output for Project {(proj_idx+offset)} to: {final_output_path}")
                    # Save the original input audio files
                    for i, ia in enumerate(project.input_audios):
                        # input_basename = os.path.basename(ia.audio_path)
                        dest_path = os.path.join(final_output_dir_proj, "stems", f"input_audio_{i}")
                        os.makedirs(os.path.join(final_output_dir_proj, "stems"), exist_ok=True) # Create stems directory
                        # dest_path = os.path.join(final_output_dir_proj, "stems", f"{ia.audio_type.lower()}.wav")
                        shutil.copy2(ia.audio_path, dest_path)
                    logging.info(f"Saved input audio for Project {(proj_idx+offset)} to: {dest_path}")
                    # Save project metadata using the built-in method
                    metadata_path = os.path.join(final_output_dir_proj, "metadata.yaml")
                    Project.save_to_yaml([project], metadata_path)
                    logging.info(f"Saved project metadata to: {metadata_path}")
                except PermissionError:
                    logging.warning(f"Permission denied when copying {ia.audio_path}. File may already exist with restricted permissions.")
                except Exception as e:
                    logging.error(f"Failed to move audio for Project {(proj_idx+offset)}: {e}")

            # except Exception as e:
            #     logging.error(f"Failed to create final output files for Project {(proj_idx+offset)}: {e}")
            #     success = False
                
        else:
            logging.error(f"Final output chain {final_task_id} for Project {(proj_idx+offset)} was not processed or output file is missing.")
            success = False
        return success
    
    except Exception as e:
        logging.error(f"Error processing final output for Project {(proj_idx+offset)}: {e}")
        return False



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


def mix_audio_files(input_files: List[str], weights: List[float], output_path: str, channels: int = None, sample_rate: int = None, clipping_prevent_mode: str = None):
    """
    Loads multiple audio files, mixes them according to weights, and saves the result.

    Args:
        input_files: List of paths to input audio files.
        weights: List of weights (gain factors) corresponding to each input file.
        output_path: Path to save the mixed audio file.

    Raises:
        ValueError: If input files have different sample rates or channel counts.
        FileNotFoundError: If any input file is not found.
    """
    if not input_files:
        logging.warning("mix_audio_files called with no input files.")
        return

    if len(input_files) != len(weights):
        raise ValueError("Number of input files and weights must match.")

    audio_data = []
    max_len = 0

    # Load audio files and find max length, check consistency
    for i, file_path in enumerate(input_files):
        try:
            data, sr = sf.read(file_path, always_2d=True) # Read as 2D array (samples, channels)
        except FileNotFoundError:
            logging.error(f"Input file not found during mixing: {file_path}")
            raise
        except Exception as e:
            logging.error(f"Error reading audio file {file_path}: {e}")
            raise

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: Expected {sample_rate}, got {sr} for {file_path}")

        if channels is None:
            channels = data.shape[1]
        elif data.shape[1] != channels:
             # Attempt mono-to-stereo conversion if needed, or raise error
             if channels == 2 and data.shape[1] == 1:
                  logging.warning(f"Converting mono file {file_path} to stereo for mixing.")
                  data = np.repeat(data, 2, axis=1) # Duplicate mono channel
             elif channels == 1 and data.shape[1] == 2:
                  logging.warning(f"Converting stereo file {file_path} to mono (averaging) for mixing.")
                  data = np.mean(data, axis=1, keepdims=True)
             else:
                  raise ValueError(f"Channel count mismatch: Expected {channels}, got {data.shape[1]} for {file_path}")

        audio_data.append(data)
        if data.shape[0] > max_len:
            max_len = data.shape[0]

    # Pad shorter files and apply weights
    mixed_signal = np.zeros((max_len, channels), dtype=np.float32)
    for i, data in enumerate(audio_data):
        padded_data = np.pad(data, ((0, max_len - data.shape[0]), (0, 0)), mode='constant')
        mixed_signal += padded_data * weights[i]

    # Optional: Normalize or clip to prevent clipping
    # Simple peak normalization:
    if clipping_prevent_mode == "normalize":
        max_abs_val = np.max(np.abs(mixed_signal))
        if max_abs_val > 1.0:
            logging.warning(f"Mixed signal peak ({max_abs_val:.2f}) exceeds 1.0 for {output_path}. Normalizing.")
            mixed_signal /= max_abs_val
    # Or hard clipping:
    elif clipping_prevent_mode == "hard_clip":
        mixed_signal = np.clip(mixed_signal, -1.0, 1.0)

    try:
        sf.write(output_path, mixed_signal, sample_rate)
        logging.debug(f"Successfully mixed {len(input_files)} files to {output_path}")
    except Exception as e:
        logging.error(f"Error writing mixed audio file {output_path}: {e}")
        raise

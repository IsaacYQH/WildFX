import os
import random
import json
import yaml
import argparse
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from scipy.stats import beta
from utils.data_class import Project, ChainDefinition, FXSetting, InputAudio
from utils.global_variables import ALLOWED_FX_TYPES, PLUGIN_PRESETS_DIR


# FX type weights for random selection (higher = more likely)
FX_TYPE_WEIGHTS = { # do not add splitter here, splitter is handled by variable 'SPLITTER_PROBABILITY'
    "eq": 5,
    "compressor": 4,
    "reverb": 2,
    "delay": 2,
    # "distortion": 1,
    # "saturation": 1,
    # "limiter": 2,
    # "utility": 3
}

TARGET_AUDIO_TYPES = ["Bass", "Guitar"]

def locate_targeted_stems(dataset_name: str, project_folder: str, min_stems: int, max_stems: int) -> Dict[str, str]:
    """
    A FUNCTION NEEDS TO BE SELF-DEFINED BY THE USER TO LOCATE THE TARGETED STEMS IN THE PROJECT FOLDER.
    
    Args:
        project_folder (str): Path to the project folder.
        min_stems (int): Minimum number of stems to use.
        max_stems (int): Maximum number of stems to use.
    
    Returns:
        Dict[str, str]: Dictionary mapping stem paths to their types.
    """
    if dataset_name == "slakh":
        # Look for metadata.yaml file
        metadata_path = os.path.join(project_folder, "metadata.yaml")
        metadata = None
        stem_instrument_map = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                    
                # Extract stem instrument information
                if metadata and 'stems' in metadata:
                    for stem_id, stem_info in metadata['stems'].items():
                        if 'inst_class' in stem_info:
                            stem_instrument_map[stem_id] = stem_info['inst_class']
            except Exception as e:
                print(f"Error reading metadata file {metadata_path}: {e}")
        
        # Get stems from this project
        stems_folder = os.path.join(project_folder, "stems")
        all_stems = []
        # seen_instruments = set()
        for filename in os.listdir(stems_folder):
            if filename.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                stem_path = os.path.join(stems_folder, filename)
                stem_id = os.path.splitext(filename)[0]  # Extract stem ID without extension
                
                # Add stem metadata if available
                instrument_type = None
                if stem_id in stem_instrument_map:
                    instrument_type = stem_instrument_map[stem_id]
                    
                all_stems.append({
                    'path': stem_path,
                    # 'id': stem_id,
                    'instrument': instrument_type
                })
                # seen_instruments.add(instrument_type)
        
        # if not seen_instruments != set(TARGET_AUDIO_TYPES):
        #     return {} # meaning skip this project if not enough stem types
        
        target_stems = {}
        seen_instruments = set()
        if metadata and stem_instrument_map:
            for stem in all_stems:
                if stem['instrument'] and any(target.lower() == stem['instrument'].lower() for target in TARGET_AUDIO_TYPES) and stem['instrument'] not in seen_instruments:
                    seen_instruments.add(stem['instrument'])
                    target_stems[stem['path']] = stem['instrument']
            print(f"Found {len(target_stems)} stems matching target instruments: {', '.join(TARGET_AUDIO_TYPES)}")

        if len(target_stems) < min_stems:
            print(f"Not enough matching instrument stems found, skip this ")
            num_stems_to_use = min(random.randint(min_stems, max_stems), len(target_stems))
            selected_stems = random.sample(all_stems, num_stems_to_use)
            project_stems = [stem['path'] for stem in selected_stems]
        else:
            # Select between min_stems and max_stems from the target stems
            num_stems_to_use = min(random.randint(min_stems, max_stems), len(target_stems))
            selected_stems = random.sample(list(target_stems.items()), num_stems_to_use)
            project_stems = {stem_path: stem_type for stem_path, stem_type in selected_stems}
            print(f"Selected stems: {', '.join(list(project_stems.keys()))}")
        
    elif dataset_name == "custom":
        ################################### Add your own dataset logic here ###################################
        pass
        ################################## Add your own dataset logic here ###################################
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Please define your own stem selection logic.")
        
    return project_stems

def load_available_plugins_by_types() -> Dict:
    """
    Load all available plugins and their metadata from preset files
    
    Returns:
        Dict: Dictionary mapping FX types to lists of available plugins
    """
    available_plugins_by_types = defaultdict(list)
    
    # Scan preset directory for available plugins
    for filename in os.listdir(PLUGIN_PRESETS_DIR):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(PLUGIN_PRESETS_DIR, filename), 'r') as f:
                    preset_data = json.load(f)
                    
                fx_name = preset_data.get('fx_name')
                fx_type = preset_data.get('fx_type', '').lower()
                
                if fx_name and fx_type and fx_type in ALLOWED_FX_TYPES:
                    n_inputs = preset_data.get('n_inputs')
                    n_outputs = preset_data.get('n_outputs')
                    presets = preset_data.get('presets', [])
                    
                    if n_inputs and n_outputs and len(presets) > 0:
                        plugin_info = {
                            'fx_name': fx_name,
                            'fx_type': fx_type,
                            'n_inputs': n_inputs,
                            'n_outputs': n_outputs,
                            'preset_count': len(presets),
                            'supports_sidechain': n_inputs > n_outputs,  # Basic heuristic for sidechain support
                            'is_splitter': fx_type == 'splitter' and n_outputs > n_inputs
                        }
                        available_plugins_by_types[fx_type].append(plugin_info)
            except Exception as e:
                print(f"Error loading plugin preset file {filename}: {e}")
    
    return available_plugins_by_types

def create_mixing_topology(
    num_stems: int,
    complexity: float = 0.5,  # 0.0 to 1.0, affects graph complexity
    available_plugins_by_types: Dict = None,
    max_chains: int = 10,
    min_chains: int = 3,
    splitter_probability: float = 0.1
) -> Tuple[Dict[int, Set[int]], List[int], List[int], Dict[int, int], Dict[int, List[str]]]:
    """
    Generate a valid DAG topology for the mixing graph
    Returns:
        - chains: List of lists where each inner list contains FX indices for that chain
        - connections: Dict mapping from source chain to set of target chains
        - stem_assignments: List mapping stem index to chain index
        - chains_needing_splitters: List of chains that need splitters
        - output_chain_idx: Index of the output chain
        - chain_layers: Dict mapping chain index to its layer number
    """
    # Validate inputs
    assert max_chains >= min_chains, "max_chains must be greater than or equal to min_chains"
    if num_stems + 1 > min_chains:
        min_chains = num_stems + 1 # minimum number of chains should be at least num_stems + 1 (for output chain)
    assert 0 <= complexity <= 1, "complexity must be between 0.0 and 1.0"
    assert 0 <= splitter_probability <= 1, "splitter_probability must be between 0.0 and 1.0"
    assert available_plugins_by_types is not None, "available_plugins_by_types must be provided"
    assert isinstance(available_plugins_by_types, dict), "available_plugins_by_types must be a dictionary"
    assert len(available_plugins_by_types) > 0, "No available plugins found"
    assert num_stems > 0, "num_stems must be greater than 0"

    # Determine number of chains based on complexity and stems
    num_chains = max(
        min_chains,
        min(
            max_chains,
            int(num_stems * (1 + complexity * 2))  # 1x to 3x stems count
        )
    )
    
    # Create connections dict (source_idx -> set of target_idx)
    connections = {i: set() for i in range(num_chains)}
    
    # Randomly assign stems to input chains
    # Ensure each stem gets a unique input chain by using exactly as many input chains as stems
    stem_assignments = random.sample(range(num_chains), num_stems)
    
    # # Assign each stem to a unique input chain
    # stem_assignments = []
    # for i in range(num_stems):
    #     # Use modulo to handle cases where num_stems > num_chains
    #     chain_idx = input_chain_indices[i % len(input_chain_indices)]
    #     stem_assignments.append(chain_idx)
    
    # Determine available bands from splitter plugins
    available_bands = defaultdict(list)  # Always include 1 for single connections
    available_bands[1] = None
    
    if available_plugins_by_types and "splitter" in available_plugins_by_types:
        for plugin in available_plugins_by_types["splitter"]:
            if plugin.get('is_splitter', False) and plugin.get('n_inputs') and plugin.get('n_outputs'):
                # Calculate number of bands (outputs per input group)
                bands = plugin.get('n_outputs') // plugin.get('n_inputs')
                if bands > 1 and bands not in available_bands:
                    available_bands[bands].append(plugin['fx_name'])
    
    # Track which layer each chain belongs to
    chain_layers = {}
    
    # Input chains are layer 0
    for chain_idx in stem_assignments:
        chain_layers[chain_idx] = 0
    
    # Build the graph layer by layer to ensure DAG property
    remaining_chains = set(range(num_chains)) - set(stem_assignments)

    # Final output is always the last chain
    output_chain_idx = remaining_chains.pop() # available_indices is always non-empty because we set min_chains to be at least num_stems + 1
    
    layer = set(stem_assignments)
    next_layer = set()
    current_layer_num = 0
    # Add intermediate layers
    while layer:
        current_layer_num += 1
        targets_to_remove = set()
        for source_idx in layer:
            # How many targets should this source connect to?
            num_available_bands = len(available_bands)
            target_count = random.choices(list(available_bands.keys()), weights=([1-splitter_probability] + [splitter_probability/(num_available_bands-1)] * (num_available_bands-1)) 
                                          if num_available_bands > 1 else [1])[0]
            if target_count > len(remaining_chains):  # if  remaining chains are not enough, fall back to 1
                target_count = 1
            
            if not remaining_chains:
                # Connect to output if no other options
                connections[source_idx].add(output_chain_idx)
                continue
                
            # Select random targets from remaining chains
            targets = random.sample(list(remaining_chains), target_count)
            targets_to_remove.update(targets)
            for target_idx in targets:
                connections[source_idx].add(target_idx)
                next_layer.add(target_idx)
                chain_layers[target_idx] = current_layer_num
        
        # Remove targets used in this layer from remaining chains
        remaining_chains -= targets_to_remove
        
        # Move to next layer
        layer = next_layer
        next_layer = set()
        
    chain_layers[output_chain_idx] = current_layer_num
        
    # Determine which chains should have splitters (only those with multiple outgoing connections)
    chains_needing_splitters = [
        i for i in range(num_chains) 
        if len(connections[i]) > 1
    ]
    
    return connections, stem_assignments, chains_needing_splitters, chain_layers, available_bands

def assign_fx_to_chains(
    connections: Dict[int, Set[int]],
    chains_needing_splitters: List[int],
    available_plugins_by_types: Dict,
    available_bands: Dict[int, List[int]], # Dict{num_bands: list of splitters with that number of bands}
    chain_layers: Dict[int, int],
    chain_depth_distribution: List[float],
    sidechain_probability: float,
) -> List[ChainDefinition]:
    """
    Populate each chain with appropriate FX
    Returns a list of ChainDefinition objects
    
    Note on choice of sampling gain parameters:
        The advantages of using the beta distribution:
        1. Naturally bounded;
        2. flexible shape;
        3. can model the asymmetric nature of real mixing practices where engineers often prefer attenuation over boost.

    
    """
    assert not any(fx_type == "splitter" for fx_type in FX_TYPE_WEIGHTS.keys()), "Splitter should not be included in FX_TYPE_WEIGHTS!"
    assert sum(chain_depth_distribution) == 1.0, "chain_depth_distribution must sum to 1.0"
    assert 0 <= sidechain_probability <= 1, "sidechain_probability must be between 0.0 and 1.0"
    assert isinstance(connections, dict), "connections must be a dictionary"
    assert isinstance(available_plugins_by_types, dict), "available_plugins_by_types must be a dictionary"
    assert len(available_plugins_by_types) > 0, "No available plugins found"
    assert isinstance(chain_layers, dict), "chain_layers must be a dictionary"
    assert isinstance(chains_needing_splitters, list), "chains_needing_splitters must be a list"
    
    chain_definitions = []
    num_chains = len(connections)
    # Track layers that already use sidechain
    layers_with_sidechain = set()
    
    # Assign FX to each chain
    for chain_idx in range(num_chains):
        fx_chain = []
        
        # Determine number of effects in this chain
        num_fx = random.choices(
            list(range(len(chain_depth_distribution))), 
            weights=chain_depth_distribution
        )[0]
        
        # Needs a splitter if in the list and has multiple outgoing connections
        needs_splitter = chain_idx in chains_needing_splitters
        
        # Ensure chains that need splitters have at least 1 effect (the splitter)
        if needs_splitter and num_fx == 0:
            num_fx = 1
        
        # For sidechain selection
        current_layer = chain_layers.get(chain_idx, 0)
        # Add regular effects
        for fx_idx in range(num_fx):
            is_last_fx = (fx_idx == num_fx - 1)
            
            # If this is the last FX and chain needs a splitter
            if is_last_fx and needs_splitter:
                fx_type = "splitter"
            else:
                # Choose a random FX type based on weights
                fx_type = random.choices(
                    list(FX_TYPE_WEIGHTS.keys()),
                    weights=[FX_TYPE_WEIGHTS[t] for t in FX_TYPE_WEIGHTS.keys()]
                )[0]
            
            # Check if we need a specific plugin type (splitter)
            if fx_type == "splitter":
                # For required splitters, we must have plugins available
                plugins_of_type = available_plugins_by_types.get(fx_type, [])
                if not plugins_of_type:
                    raise ValueError(f"No splitter plugins available but a splitter is required for chain {chain_idx}")

                # Count how many next_chains this chain has
                next_chains_count = len(connections.get(chain_idx, set()))
                
                # Now we can choose from the suitable splitters
                plugin = random.choice(available_bands.get(next_chains_count, []))
            else:
                # For general effects, select only from types that have available plugins
                available_fx_types = [fx_t for fx_t in FX_TYPE_WEIGHTS.keys() 
                                    if available_plugins_by_types.get(fx_t, [])]
                
                if not available_fx_types:
                    raise ValueError("No valid plugins available for any effect type")
                
                # If original type has no plugins, select a new one from available types
                if not available_plugins_by_types.get(fx_type, []):
                    # Recalculate weights for available types only
                    available_weights = [FX_TYPE_WEIGHTS[t] for t in available_fx_types]
                    fx_type = random.choices(available_fx_types, weights=available_weights)[0]
                
                plugins_of_type = available_plugins_by_types.get(fx_type, [])

            # Now we're guaranteed to have plugins
            plugin = random.choice(plugins_of_type)
            
            # Choose a random preset index
            preset_index = random.randint(0, plugin['preset_count'] - 1)
            
            # Determine if this plugin should use sidechain
            sidechain_input = None
            if (plugin.get('supports_sidechain', False) and 
                random.random() < sidechain_probability and
                current_layer not in layers_with_sidechain):  # Only allow one sidechain per layer
                
                # Find valid sidechain sources that are in the same layer
                potential_sources = [
                    i for i in range(num_chains)
                    if i != chain_idx  # Don't use self as sidechain
                    and chain_layers.get(i, 0) == current_layer  # Must be in same layer
                    and i not in chains_needing_splitters # sidechain input cannot come from a splitter output
                ]
                
                if potential_sources:
                    sidechain_input = random.choice(potential_sources)
                    layers_with_sidechain.add(current_layer)  # Mark this layer as having a sidechain
            
            # Create FXSetting
            fx_setting = FXSetting(
                fx_name=plugin['fx_name'],
                fx_type=fx_type,
                preset_index=preset_index,
                n_inputs=plugin['n_inputs'],
                n_outputs=plugin['n_outputs'],
                sidechain_input=sidechain_input
            )
            
            fx_chain.append(fx_setting)
        
        # Convert connections from sets to weighted dictionaries
        next_chains_dict = {}
        for target_idx in connections.get(chain_idx, set()):
            # The advantages of using the beta distribution:
            # 1. Naturally bounded;
            # 2. flexible shape;
            # 3. can model the asymmetric nature of real mixing practices where engineers often prefer attenuation over boost.
            
            # Parameters to control shape (α>1, β>1 gives bell curve)
            alpha, beta_param = 2, 3  # Slightly weighted toward lower gains
            # gain here is not in dB, but in linear scale, which can be converted to dB by: 20 * np.log10(x)
            min_gain, max_gain = 0.1, 2.0  # -20dB to +6dB
            gain = min_gain + beta.rvs(alpha, beta_param) * (max_gain - min_gain)
            next_chains_dict[target_idx] = round(gain, 2).item()
        
        # Create the ChainDefinition
        chain_def = ChainDefinition(
            FxChain=fx_chain,
            next_chains=next_chains_dict
        )
        
        chain_definitions.append(chain_def)
    
    return chain_definitions

def create_project(
    stems_with_labels: Dict[str, str],
    stem_assignments: List[int],
    chain_definitions: List[ChainDefinition],
) -> Project:
    """
    Create a Project object from stems and chain definitions
    
    Args:
        stems_with_labels: Dictionary mapping stem paths to their types
        chain_definitions: List of ChainDefinition objects
        stem_assignments: List mapping stem index to chain index
    Returns:
        Project: The created Project object
    """
    # Create InputAudio objects
    input_audios = []
    for i, stem_path in enumerate(stems_with_labels):
        chain_idx = stem_assignments[i]
        input_audio = InputAudio(
            audio_path=stem_path,
            audio_type=stems_with_labels[stem_path],
            input_FxChain=chain_idx
        )
        input_audios.append(input_audio)

    # Create Project
    project = Project(
        FxChains=chain_definitions,
        input_audios=input_audios,
        customized=False
    )
    return project

def generate_mixing_graph(
    dataset_name: str,
    dataset_root_dir: str,
    output_path: str,
    num_projects: int = 100,
    complexity: float = 0.5,
    min_stems: int = 1,
    max_stems: int = 4,
    max_chains: int = 10,
    min_chains: int = 3,
    chain_depth_distribution: List[float] = [0.1, 0.3, 0.4, 0.2],
    sidechain_probability: float = 0.2,
    splitter_probability: float = 0.1,
    variable_density: bool = False,
    density_range: float = 0.3
):
    """
    Generate mixing graphs using a dataset with projects containing stem folders.
    
    Args:
        dataset_root_dir: Root directory of dataset (contains project folders)
        output_path: Path to save generated projects
        num_projects: Number of projects to generate
        complexity: Complexity level (0.0 to 1.0)
        min_stems: Minimum number of stems to use per project
        max_stems: Maximum number of stems to use per project
    """
    if any(fx not in ALLOWED_FX_TYPES for fx in set(FX_TYPE_WEIGHTS.keys())):
        raise ValueError("Invalid FX types in FX_TYPE_WEIGHTS. Allowed types are defined in global_variables.py")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    # Find all project folders in the dataset root directory
    # Each project should have a "stems" subfolder
    project_folders = []
    for item in os.listdir(dataset_root_dir):
        full_path = os.path.join(dataset_root_dir, item)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "stems")):
            project_folders.append(full_path)
    
    if not project_folders:
        print(f"No valid project folders found in {dataset_root_dir}")
        return
    
    original_project_count = len(project_folders)
    print(f"Found {original_project_count} original projects")
    
    # Calculate repetitions needed to reach target project count
    if original_project_count > 0:
        repetitions_needed = (num_projects + original_project_count - 1) // original_project_count
        print(f"Each project will be sampled approximately {repetitions_needed} times")
    else:
        print("No projects to process")
        return
    
    # Load available plugins
    available_plugins_by_types = load_available_plugins_by_types()
    print(f"Loaded {sum(len(plugins) for plugins in available_plugins_by_types.values())} plugins")
    
    # Generate projects
    projects = []
    project_idx = 0
    
    # Store original density parameters if using variable density
    original_complexity = complexity
    original_chain_depth = chain_depth_distribution.copy()
    original_sidechain_prob = sidechain_probability
    original_splitter_prob = splitter_probability
    
    # Process each original project multiple times if needed
    while project_idx < num_projects:
        # Cycle through all project folders
        for project_folder in project_folders:
            if project_idx >= num_projects:
                break
                
            project_name = os.path.basename(project_folder)
            print(f"Processing project: {project_name} ({project_idx + 1}/{num_projects})")
            
            # Locate exact stems you want in the dataset, NEED TO BE SELF-DEFINED!
            project_stems = locate_targeted_stems(dataset_name, project_folder, min_stems, max_stems)
            if not project_stems:
                print(f"No valid stems found in {project_folder}")
                continue
            else:
                print(f"Using {len(project_stems)} stems from {project_name}")

            
            
            # If variable density is enabled, randomize parameters for this project
            if variable_density:
                # Vary complexity within bounds
                project_complexity = min(1.0, max(0.1, 
                    original_complexity + random.uniform(-density_range, density_range)))
                
                # Vary chain depth distribution (keeping sum at 1.0)
                project_chain_depth = original_chain_depth.copy()
                for i in range(len(project_chain_depth)-1):
                    variation = random.uniform(-density_range/2, density_range/2)
                    project_chain_depth[i] = max(0.05, min(0.8, project_chain_depth[i] + variation))
                # Normalize to ensure sum is 1.0
                total = sum(project_chain_depth)
                project_chain_depth = [x/total for x in project_chain_depth]
                
                # Vary probabilities
                project_sidechain_prob = min(0.5, max(0.0, 
                    original_sidechain_prob + random.uniform(-density_range, density_range)))
                project_splitter_prob = min(0.5, max(0.0, 
                    original_splitter_prob + random.uniform(-density_range, density_range)))
                
                print(f"Using variable density: complexity={project_complexity:.2f}, "
                    f"sidechain_prob={project_sidechain_prob:.2f}, splitter_prob={project_splitter_prob:.2f}")
            else:
                # Use the fixed parameters
                project_complexity = complexity
                project_chain_depth = chain_depth_distribution
                project_sidechain_prob = sidechain_probability
                project_splitter_prob = splitter_probability

            # Then use these project-specific parameters in the function calls:
            connections, stem_assignments, chains_needing_splitters, chain_layers, available_bands = create_mixing_topology(
                num_stems=len(project_stems),
                complexity=project_complexity,
                available_plugins_by_types=available_plugins_by_types,
                max_chains=max_chains,
                min_chains=min_chains,
                splitter_probability=project_splitter_prob
            )            
            
            # Assign FX to chains
            chain_definitions = assign_fx_to_chains(
                connections=connections,
                chains_needing_splitters=chains_needing_splitters,
                available_plugins_by_types=available_plugins_by_types,
                available_bands=available_bands,
                chain_layers=chain_layers,
                chain_depth_distribution=chain_depth_distribution,
                sidechain_probability=sidechain_probability,
                
            )
            
            try:
                project = create_project(
                    stems_with_labels=project_stems,
                    stem_assignments=stem_assignments,
                    chain_definitions=chain_definitions,
                )
                projects.append(project)
                print(f"Successfully created project {project_idx + 1}")                            
            except Exception as e:
                print(f"Failed to process project {project_idx + 1}: {e}")
            
            # Increment project index regardless of success/failure
            project_idx += 1
            
            # Stop if we've reached the desired number of projects
            if project_idx >= num_projects:
                break
    
    # Save projects
    if projects:
        try:
            Project.save_to_yaml(projects, output_path)
            print(f"Successfully saved {len(projects)} projects to {output_path}")
        except Exception as e:
            print(f"Error saving projects to YAML: {e}")
    else:
        print("No valid projects were generated")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate audio mixing graphs from stems.')
    parser.add_argument('--dataset-name', type=str, required=True, help='Identifier for the dataset.')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Root directory of dataset containing project folders.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save generated projects metadata ending with ".yaml".')
    parser.add_argument('--num-projects', type=int, required=True, help='Number of projects to generate.')
    parser.add_argument('--complexity', type=float, default=0.5, help='Complexity level (0.0 to 1.0).')
    parser.add_argument('--min-stems', type=int, default=1, help='Minimum number of stems to use per project.')
    parser.add_argument('--max-stems', type=int, default=2, help='Maximum number of stems to use per project.')
    
    parser.add_argument('--max-chains', type=int, default=5, help='Maximum number of FX chains in a project.')
    parser.add_argument('--min-chains', type=int, default=3, help='Minimum number of FX chains in a project.')
    parser.add_argument('--sidechain-prob', type=float, default=0.2, help='Chance of a compatible FX using sidechain.')
    parser.add_argument('--splitter-prob', type=float, default=0.1, help='Chance of a chain ending with a splitter.')
    parser.add_argument('--chain-depth', type=str, default="0.1,0.7,0.2", 
                       help='Comma-separated probabilities for number of FX per chain. From depth 0 (empty chain) to len(depth)-1. ')
    parser.add_argument('--variable-density', action='store_true', 
                    help='Randomly vary density parameters for each project instead of using fixed values')
    parser.add_argument('--density-range', type=float, default=0.3, 
                    help='If variable-density is enabled, controls the range (+/-) for random variation')
    
    args = parser.parse_args()
    
    # Parse the chain depth distribution
    CHAIN_DEPTH_DISTRIBUTION = [float(x) for x in args.chain_depth.split(',')]
    
    # Set other constants from arguments
    MAX_CHAINS_PER_PROJECT = args.max_chains
    MIN_CHAINS_PER_PROJECT = args.min_chains
    SIDECHAIN_PROBABILITY = args.sidechain_prob
    SPLITTER_PROBABILITY = args.splitter_prob
    
    generate_mixing_graph(
        dataset_name=args.dataset_name,
        dataset_root_dir=args.dataset_dir,
        output_path=args.output_path,
        num_projects=args.num_projects,
        complexity=args.complexity,
        min_stems=args.min_stems,
        max_stems=args.max_stems,
        # Pass the newly configured constants
        max_chains=MAX_CHAINS_PER_PROJECT,
        min_chains=MIN_CHAINS_PER_PROJECT,
        chain_depth_distribution=CHAIN_DEPTH_DISTRIBUTION,
        sidechain_probability=SIDECHAIN_PROBABILITY,
        splitter_probability=SPLITTER_PROBABILITY,
        variable_density=args.variable_density,
        density_range=args.density_range
    )
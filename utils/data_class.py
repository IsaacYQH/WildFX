from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict, deque
import yaml
import os
import json
import bisect
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gen_presets import create_safe_instance_name
from .global_variables import ALLOWED_FX_TYPES, PLUGIN_PRESETS_DIR


def find_closest(sorted_list: List[float], target: float) -> Optional[float]:
    """
    Finds the single value in a sorted list of floats that is closest to the target value.

    Uses binary search (bisect_left) for O(log n) efficiency, which is the 
    fastest approach for sorted data.

    Args:
        sorted_list: A list of floats, sorted in ascending order.
        target: The float value to find the closest match for.

    Returns:
        The value in the list closest to the target,
        or None if the list is empty.
        If two numbers are equally close, returns the smaller number.
    """
    n = len(sorted_list)
    if n == 0:
        return None
    if n == 1:
        return sorted_list[0]
    if not target:
        return None

    # Find the insertion point for the target using binary search.
    # idx is the index where target would be inserted to maintain order.
    idx = bisect.bisect_left(sorted_list, target)

    # Handle cases where the target is outside the list's bounds
    if idx == 0:
        # Target is less than or equal to the first element
        return sorted_list[0]
    if idx == n:
        # Target is greater than the last element
        return sorted_list[n-1]

    # Target is within the list's range (or equal to an element).
    # The potential closest values are the element before the insertion point (idx-1)
    # and the element at the insertion point (idx).
    val_before = sorted_list[idx - 1]
    val_after = sorted_list[idx]

    # Compare the absolute differences to find the closer value
    diff_before = abs(target - val_before)
    diff_after = abs(target - val_after)

    # Return the value with the smaller difference.
    # In case of a tie (equally close), return the smaller value (val_before).
    if diff_before <= diff_after:
        return val_before
    else:
        return val_after

@dataclass
class FXSetting:
    """Represents a single FX setting within a chain."""
    fx_name: str
    fx_type: str  # Category/type of the plugin
    preset_index: Optional[int] = None  # params might be loaded from a preset file
    params: List[Optional[float]] = field(default_factory=list)  # Parameter values
    n_inputs: Optional[int] = None # Will be populated/validated from preset JSON during load
    n_outputs: Optional[int] = None # Will be populated/validated from preset JSON during load
    sidechain_input: Optional[int] = None  # Index of the FxChain providing the sidechain signal (None if disabled)
    
@dataclass
class ChainDefinition:
    """Represents a single chain (node) in the processing graph."""
    FxChain: List[FXSetting] # The actual list of FX applied in this chain
    # prev_chains removed
    next_chains: Dict[int, float] = field(default_factory=dict) # Indices of chains this one feeds into with weights/gains

@dataclass
class InputAudio:
    """Represents an input audio file and its entry point into the graph."""
    audio_path: str
    audio_type: str
    input_FxChain: int # Index of the FxChain this audio feeds into

@dataclass
class Project:
    """Represents the overall project structure defined in the metadata."""
    def __init__(self, FxChains: List[ChainDefinition], input_audios: List[InputAudio], output_audio: str=None, customized: bool = True):
        self.FxChains = FxChains
        self.input_audios = input_audios
        self.output_audio = output_audio
        self.customized = customized
        # Assign the validation method to the instance
        validation_errors = self.validate()
        if validation_errors:
            error_msg = "Validation failed during Project initialization:\n" + "\n".join(f"- {e}" for e in validation_errors)
            raise ValueError(error_msg)
                        
    @staticmethod
    def load_from_yaml(yaml_path: str) -> List['Project']:
        """Loads a list of Project objects from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        with open(yaml_path, 'r') as f:
            try:
                data_list = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file {yaml_path}: {e}")

        projects = []
        if not isinstance(data_list, list):
            raise TypeError(f"Expected top-level structure in {yaml_path} to be a list.")

        for proj_idx, project_data in enumerate(data_list):
            if not isinstance(project_data, dict):
                raise TypeError(f"Expected item {proj_idx} in the list of {yaml_path} to be a dictionary.")

            # --- Parse FxChains ---
            chain_defs_data = project_data.get('FxChains', [])
            chain_defs = []
            customized_flag = project_data.get('customized', True) # Default to True if not specified
            if not isinstance(chain_defs_data, list):
                raise TypeError("Expected 'FxChains' to be a list.")
            # --- FX Chain Parsing Logic (Handles both customized and non-customized) ---
            for chain_idx, chain_data in enumerate(chain_defs_data):
                if not isinstance(chain_data, dict):
                    raise TypeError(f"Project {proj_idx}, Chain {chain_idx}: Expected items in 'FxChains' to be dictionaries.")

                fx_settings_data = chain_data.get('FxChain', [])
                fx_settings = []
                if not isinstance(fx_settings_data, list):
                    raise TypeError(f"Project {proj_idx}, Chain {chain_idx}: Expected 'FxChain' within a chain definition to be a list.")

                # --- FXSetting Parsing Logic ---
                for fx_idx, fx_data in enumerate(fx_settings_data):
                    if not isinstance(fx_data, dict):
                        raise TypeError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx}: Expected items in 'FxChain' list to be dictionaries.")

                    fx_name = fx_data.get('fx_name')
                    fx_type = fx_data.get('fx_type') # Get the type
                    # --- Get n_inputs/n_outputs from YAML (if provided) ---
                    n_inputs_yaml = fx_data.get('n_inputs', None)
                    n_outputs_yaml = fx_data.get('n_outputs', None)
                    preset_index_yaml = fx_data.get('preset_index')
                    params_from_yaml = fx_data.get('params')

                    if fx_name is None:
                        raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx}: Missing 'fx_name'.")
                    if fx_type is None:
                        raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): Missing 'fx_type'.")
                    if fx_type not in ALLOWED_FX_TYPES:
                        raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): Invalid 'fx_type' '{fx_type}'. Allowed types: {ALLOWED_FX_TYPES}")

                    final_params = []
                    final_preset_index = preset_index_yaml

                    safe_name = create_safe_instance_name(fx_name)
                    fx_path = os.path.join(PLUGIN_PRESETS_DIR, f"{safe_name}.json")
                    # Load valid params structure for validation
                    try:
                        with open(fx_path, 'r') as pf:
                            preset_data = json.load(pf)
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx}: Preset definition file not found for '{fx_name}' at {fx_path} (needed for param validation).")
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx}: Error decoding JSON from preset file {fx_path}: {e}")
                    except Exception as e:
                        raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx}: Error processing preset file {fx_path} for '{fx_name}': {e}")
                    
                    if preset_data.get('fx_name') != fx_name:
                        raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx}: Preset file {fx_path} does not match the fx_name '{fx_name}' in YAML.")

                    # Parameter loading/validation logic (only if customized)
                    if customized_flag:                        
                        # Check if params are provided in YAML
                        if params_from_yaml:
                            if not isinstance(params_from_yaml, list):
                                raise TypeError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): 'params' must be a list or null.")
                            valid_params_dict = preset_data.get('valid_params')
                            if valid_params_dict is None or not isinstance(valid_params_dict, dict):
                                raise ValueError(f"Preset file {fx_path} for {fx_name} is missing 'valid_params' dictionary.")
                            valid_params_values = list(valid_params_dict.values()) # Get list of valid value lists

                            if len(valid_params_values) != len(params_from_yaml):
                                raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): Invalid number of params. Expected {len(valid_params_values)}, got {len(params_from_yaml)}.")

                            # Check/adjust params against valid ranges/values
                            final_params = []
                            for i, (param_val, valid_value_list) in enumerate(zip(params_from_yaml, valid_params_values)):
                                if param_val is None: # Allow null for default
                                    final_params.append(None)
                                elif isinstance(param_val, (int, float)):
                                    if not valid_value_list: # If valid_value_list is empty, the final param is None
                                        closest_valid = None
                                    else:
                                        # Assuming valid_value_list is sorted list of allowed discrete values
                                        closest_valid = find_closest(sorted(valid_value_list), float(param_val))
                                        if closest_valid is None: # Should not happen if valid_value_list is not empty
                                            raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): Could not find closest value for param {i} ({param_val}) in {fx_path}.")
                                    final_params.append(closest_valid)
                                else:
                                    raise TypeError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): Parameter {i} must be a number or null, got {type(param_val)}.")

                            final_preset_index = None # Params were provided directly

                        else:
                            # Params are empty/None, try to load from preset file using preset_index
                            if preset_index_yaml is None or not isinstance(preset_index_yaml, int) or preset_index_yaml < 0:
                                raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): Missing or invalid 'preset_index' when 'params' are not provided.")

                            presets_list = preset_data.get('presets')
                            if presets_list is None or not isinstance(presets_list, list):
                                raise ValueError(f"Preset file {fx_path} for {fx_name} is missing 'presets' list.")
                            if not (0 <= preset_index_yaml < len(presets_list)):
                                raise IndexError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): 'preset_index' {preset_index_yaml} is out of bounds for presets in {fx_path} (found {len(presets_list)} presets).")

                            final_params = presets_list[preset_index_yaml]
                            final_preset_index = preset_index_yaml # Keep original index
                        
                        # --- Load Preset JSON for n_inputs/n_outputs Validation ---
                        n_inputs_preset = None
                        n_outputs_preset = None
                        preset_load_error = None
                        # Get n_inputs and n_outputs from preset, default to None if missing
                        n_inputs_preset = preset_data.get('n_inputs')
                        n_outputs_preset = preset_data.get('n_outputs')

                        # Validate values from preset JSON
                        if n_inputs_preset is None or not isinstance(n_inputs_preset, int) or n_inputs_preset < 1:
                            preset_load_error = f"Invalid or missing 'n_inputs' ({n_inputs_preset}) in preset file {fx_path}. Must be a positive integer."
                            n_inputs_preset = None # Mark as invalid
                        if n_outputs_preset is None or not isinstance(n_outputs_preset, int) or n_outputs_preset < 1:
                            preset_load_error = f"Invalid or missing 'n_outputs' ({n_outputs_preset}) in preset file {fx_path}. Must be a positive integer."
                            n_outputs_preset = None # Mark as invalid

                        if preset_load_error:
                            # If preset loading failed, we cannot validate/populate n_inputs/n_outputs
                            raise ValueError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): {preset_load_error}")

                        # --- Determine Final n_inputs/n_outputs ---
                        final_n_inputs = n_inputs_preset
                        final_n_outputs = n_outputs_preset

                        # Validate YAML values against preset values if YAML provided them
                        if n_inputs_yaml is not None:
                            if not isinstance(n_inputs_yaml, int) or n_inputs_yaml < 1:
                                print(f"Warning: Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): 'n_inputs' provided in YAML ({n_inputs_yaml}) is invalid. Must be a positive integer. Using value from preset file.")
                            if n_inputs_yaml != n_inputs_preset:
                                print(f"Warning: Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): 'n_inputs' in YAML ({n_inputs_yaml}) does not match the value in preset file {fx_path} ({n_inputs_preset}). Using value from preset file.")
                            # If validation passes, the value is already set correctly

                        if n_outputs_yaml is not None:
                            if not isinstance(n_outputs_yaml, int) or n_outputs_yaml < 1:
                                print(f"Warning: Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): 'n_outputs' provided in YAML ({n_outputs_yaml}) is invalid. Must be a positive integer. Using value from preset file.")
                            if n_outputs_yaml != n_outputs_preset:
                                print(f"Warning: Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): 'n_outputs' in YAML ({n_outputs_yaml}) does not match the value in preset file {fx_path} ({n_outputs_preset}). Using value from preset file.")
                    else:
                        # In generated metadata, only the preset index will be given
                        presets_list = preset_data.get('presets')
                        if presets_list is None or not isinstance(presets_list, list):
                            raise ValueError(f"Preset file {fx_path} for {fx_name} is missing 'presets' list.")
                        if not (0 <= preset_index_yaml < len(presets_list)):
                            raise IndexError(f"Project {proj_idx}, Chain {chain_idx}, FX {fx_idx} ('{fx_name}'): 'preset_index' {preset_index_yaml} is out of bounds for presets in {fx_path} (found {len(presets_list)} presets).")

                        final_params = presets_list[preset_index_yaml]
                        final_preset_index = preset_index_yaml # Keep original index
                        final_n_inputs = n_inputs_yaml
                        final_n_outputs = n_outputs_yaml

                    # Append the final FXSetting with validated/fetched n_inputs/n_outputs
                    fx_settings.append(FXSetting(
                        fx_name=fx_name,
                        fx_type=fx_type,
                        preset_index=final_preset_index,
                        params=final_params,
                        n_inputs=final_n_inputs, # Use validated/fetched value
                        n_outputs=final_n_outputs, # Use validated/fetched value
                        sidechain_input=fx_data.get('sidechain_input')
                    ))
                    
                # --- End FXSetting Parsing Logic ---

                # --- Parse next_chains (now a dictionary) ---
                next_chains_data = chain_data.get('next_chains', {}) # Default to empty dict
                if not isinstance(next_chains_data, dict):
                     raise TypeError(f"Project {proj_idx}, Chain {chain_idx}: Expected 'next_chains' to be a dictionary, got {type(next_chains_data)}.")

                # Validate next_chains keys and values during load
                validated_next_chains = {}
                for next_idx_str, weight in next_chains_data.items():
                    try:
                        next_idx = int(next_idx_str)
                    except ValueError:
                         raise ValueError(f"Project {proj_idx}, Chain {chain_idx}: 'next_chains' contains non-integer key '{next_idx_str}'.")
                    if not isinstance(weight, (int, float)) or weight < 0:
                         raise ValueError(f"Project {proj_idx}, Chain {chain_idx}: 'next_chains' weight for key {next_idx} must be a non-negative number, got {weight}.")
                    validated_next_chains[next_idx] = float(weight) # Store weight as float

                chain_defs.append(ChainDefinition(
                    FxChain=fx_settings,
                    next_chains=validated_next_chains # Use validated dictionary
                ))
            # --- End FX Chain Parsing ---
            
            # --- Parse InputAudios ---
            input_audios_data = project_data.get('input_audios', [])
            input_audios = []
            if not isinstance(input_audios_data, list):
                raise TypeError("Expected 'input_audios' to be a list.")
            for audio_idx, audio_data in enumerate(input_audios_data):
                if not isinstance(audio_data, dict):
                    raise TypeError(f"Project {proj_idx}, InputAudio {audio_idx}: Expected items in 'input_audios' to be dictionaries.")
                audio_path = audio_data.get('audio_path')
                audio_type = audio_data.get('audio_type')
                input_fx_chain = audio_data.get('input_FxChain')
                if audio_path is None:
                    raise ValueError(f"Project {proj_idx}, InputAudio {audio_idx}: Missing 'audio_path'.")
                if audio_type is None:
                    raise ValueError(f"Project {proj_idx}, InputAudio {audio_idx}: Missing 'audio_type'.")
                if input_fx_chain is None or not isinstance(input_fx_chain, int):
                    raise ValueError(f"Project {proj_idx}, InputAudio {audio_idx}: Missing or invalid 'input_FxChain'.")
                input_audios.append(InputAudio(
                    audio_path=audio_path,
                    audio_type=audio_type,
                    input_FxChain=input_fx_chain
                ))

            # --- Create Project Instance (will trigger __post_init__ validation) ---
            try:
                project_instance = Project(
                    FxChains=chain_defs,
                    input_audios=input_audios,
                    output_audio=project_data.get('output_audio'),
                    customized=customized_flag
                )
                projects.append(project_instance)
            except ValueError as validation_error:
                raise ValueError(f"Validation failed for project {proj_idx} defined in {yaml_path}:\n{validation_error}")

        return projects

    @staticmethod
    def save_to_yaml(projects: List['Project'], yaml_path: str):
        """Saves a list of Project objects to a YAML file."""
        data_list = []
        for project in projects:
            project_data = {
                'FxChains': [
                    {
                        'FxChain': [fx_setting.__dict__ for fx_setting in chain_def.FxChain],
                        'next_chains': chain_def.next_chains
                    } for chain_def in project.FxChains
                ],
                'input_audios': [input_audio.__dict__ for input_audio in project.input_audios],
                'output_audio': project.output_audio,
                'customized': project.customized,
            }
            data_list.append(project_data)

        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(data_list, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise IOError(f"Error writing YAML file {yaml_path}: {e}")

    def validate(self) -> List[str]:
        """
        Performs validation checks on the graph structure defined by FxChains,
        input_audios, and next_chains. Uses topological sort to determine layers
        and check dependencies.

        Rules for a valid mixing graph based on this validation:
        1.  **Acyclicity:** The graph defined by input_audios and next_chains must be a
            Directed Acyclic Graph (DAG). (Checked implicitly by successful topological sort).
        2.  **Valid Indices:** All indices referenced in `next_chains`,
            `input_audios` (as `input_FxChain`), and `sidechain_input` must be valid
            indices within the FxChains list.
        3.  **Start Nodes (Inputs):**
            - Chains designated as entry points via `input_audios` must not have any
            predecessors within the graph structure derived from `next_chains`.
            - Any chain with no predecessors (in-degree 0) must be listed as an
            `input_FxChain` in at least one `input_audios` entry.
            - The set of chains with in-degree 0 must exactly match the set of unique
            `input_FxChain` indices.
        4.  **End Node (Output):**
            - There must be exactly ONE chain that acts as the final output. This chain
            MUST have an empty `next_chains` dictionary.
        5.  **FX Types:** All FX types must be in the ALLOWED_FX_TYPES set.
        6.  **Splitter Rules:**
            - Splitters (FX with type "splitter") must ONLY appear as the LAST effect in a chain.
            - Any chain with multiple entries in its `next_chains` dictionary MUST have a
              Splitter as its last FX.
            - A Splitter as the last FX must satisfy `n_outputs / n_inputs >= len(next_chains)`
              (assuming outputs are grouped, e.g., stereo pairs). The number of next chains
              should typically match the number of output groups.
            - The final output chain (with empty `next_chains`) CANNOT end with a Splitter.
        7.  **Sidechain Rules:**
            - `sidechain_input` must reference a valid chain index.
            - Only one sidechain-enabled plugin is allowed per chain.
            - A chain providing a sidechain signal (referenced by `sidechain_input`) must be
            processed in the same layer or an earlier layer than the receiving chain,
            as determined by the topological sort.
            - A plugin using `sidechain_input` should have `n_inputs > 1`.
            - Sidechain input cannot come from a splitter output chain.
        8.  **Empty Chains:** Chains with an empty `FxChain` list are allowed (represent pass-through).
        9.  **Next Chains Dictionary:**
            - `next_chains` must be a dictionary.
            - Keys must be valid integer chain indices.
            - Values (weights/gains) must be non-negative numbers.
        """
        errors = []
        num_chains = len(self.FxChains)
        if num_chains == 0:
            errors.append("Project has no FxChains defined.")
            return errors # Stop validation if no chains exist

        valid_chain_indices = set(range(num_chains))
        # Collect all unique chain indices designated as inputs by the input_audios list
        input_chain_indices = {audio.input_FxChain for audio in self.input_audios}

        # --- Basic Input Audio Checks ---
        if not self.input_audios:
            errors.append("Project must have at least one input_audio defined.")
        for audio_idx, audio in enumerate(self.input_audios):
            if not audio.audio_path or not isinstance(audio.audio_path, str):
                errors.append(f"InputAudio {audio_idx}: audio_path must be a non-empty string.")
                # For now, there is no check for audio_type
            if audio.input_FxChain not in valid_chain_indices:
                errors.append(f"InputAudio {audio_idx}: input_FxChain index {audio.input_FxChain} is invalid.")
                        
        
        # --- Topological Sort Initialization (Based on next_chains) ---
        in_degree = defaultdict(int)
        successors = defaultdict(list) # Store successors for traversal
        predecessors = defaultdict(list) # Store predecessors to calculate in_degree
        chain_layers = {} # Stores {chain_idx: layer_num}
        processed_chains_count = 0

        # Build graph structure (successors, predecessors) and check next_chains validity
        for current_chain_idx, chain in enumerate(self.FxChains):
            # Basic type checks
            if not isinstance(chain.FxChain, list): errors.append(f"Chain {current_chain_idx}: FxChain is not a list.")
            if not isinstance(chain.next_chains, dict): errors.append(f"Chain {current_chain_idx}: next_chains must be a dictionary.")

            # Build successors and predecessors maps from next_chains
            for next_idx, weight in chain.next_chains.items():
                 # Validate next_idx and weight (already done in load, but good for direct instantiation)
                 if not isinstance(next_idx, int):
                     errors.append(f"Chain {current_chain_idx}: next_chains contains non-integer key '{next_idx}'.")
                     continue
                 if next_idx not in valid_chain_indices:
                     errors.append(f"Chain {current_chain_idx}: next_chains contains invalid index {next_idx}.")
                     continue
                 if not isinstance(weight, (int, float)) or weight < 0:
                     errors.append(f"Chain {current_chain_idx}: Weight {weight} for next_chain {next_idx} must be non-negative.")
                     continue

                 successors[current_chain_idx].append(next_idx)
                 predecessors[next_idx].append(current_chain_idx)

        # Calculate in-degree based on the built predecessors map
        for i in range(num_chains):
            in_degree[i] = len(predecessors[i])
            
        # Identify start nodes for the queue (nodes designated as input AND having in-degree 0)
        queue = deque()
        actual_start_nodes = set()
        for idx in range(num_chains):
            if in_degree[idx] == 0:
                actual_start_nodes.add(idx)
                if idx in input_chain_indices:
                    queue.append(idx)
                else:
                    # This node has no predecessors but wasn't designated as an input
                    errors.append(f"Input Rule Violation: Chain {idx} is a start node (no predecessors) but not listed in input_audios.")

        # Check if designated input nodes actually have predecessors
        for idx in input_chain_indices:
            if in_degree[idx] != 0:
                errors.append(f"Input Rule Violation: Chain {idx} listed in input_audios but has predecessors (indices: {predecessors[idx]}).")

        current_layer = 0
        # --- Layer-by-Layer Validation ---
        while queue:
            nodes_in_layer = len(queue)
            if nodes_in_layer == 0: break # Should not happen if queue is not empty, but safe
            init_queue = list(queue)
                        
            for _ in range(nodes_in_layer):
                current_chain_idx = queue.popleft()
                chain = self.FxChains[current_chain_idx]
                chain_layers[current_chain_idx] = current_layer
                processed_chains_count += 1
                
                # 1. Check FxChain type and content type
                if not isinstance(chain.FxChain, list):
                    errors.append(f"Chain {current_chain_idx}: FxChain must be a list, got {type(chain.FxChain)}.")
                    # Skip FX checks if FxChain itself is invalid
                    continue
                else:
                    sidechain_count = 0
                    for fx_idx, fx_setting in enumerate(chain.FxChain):
                        if not isinstance(fx_setting, FXSetting):
                            errors.append(f"Chain {current_chain_idx}, FxChain item {fx_idx}: Expected FXSetting object, got {type(fx_setting)}.")
                            # Skip detailed checks for this item if it's not the right type
                            continue
                                    
                        # --- Detailed Checks for FXSetting object ---

                        # a. Check fx_name
                        if not fx_setting.fx_name or not isinstance(fx_setting.fx_name, str):
                            errors.append(f"Chain {current_chain_idx}, FX {fx_idx}: 'fx_name' must be a non-empty string, got '{fx_setting.fx_name}'.")

                        # b. Check fx_type
                        if not fx_setting.fx_type or not isinstance(fx_setting.fx_type, str):
                            errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): 'fx_type' must be a non-empty string.")
                        elif fx_setting.fx_type.lower() not in ALLOWED_FX_TYPES:
                            errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): Invalid 'fx_type' '{fx_setting.fx_type}'. Allowed types: {ALLOWED_FX_TYPES}")

                        # c. Check preset_index
                        if fx_setting.preset_index is None:
                            if fx_setting.params is None:
                                errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): 'preset_index' is null, but 'params' is also null. At least one must be provided.")
                        elif not isinstance(fx_setting.preset_index, int) or fx_setting.preset_index < 0:
                            errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): 'preset_index' must be a non-negative integer or null, got {fx_setting.preset_index}.")
                                    
                        # d. Check params structure and content
                        if not isinstance(fx_setting.params, list):
                            errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): 'params' must be a list, got {type(fx_setting.params)}.")
                        else:
                            # Check values are valid types (float, int, or None)
                            for param_idx, param_value in enumerate(fx_setting.params):
                                if param_value is not None and not isinstance(param_value, (float, int)):
                                    errors.append(f"Chain {current_chain_idx}, FX {fx_idx}: Parameter {param_idx} must be a number or None, got {type(param_value)}")
                            # Note: Checking param count against preset is complex here and better handled during loading.

                        # e. Check n_inputs and n_outputs (should be populated by load_from_yaml)
                        if not isinstance(fx_setting.n_inputs, int) or fx_setting.n_inputs < 1:
                            errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): n_inputs must be a positive integer, got {fx_setting.n_inputs}")
                        if not isinstance(fx_setting.n_outputs, int) or fx_setting.n_outputs < 1:
                            errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): n_outputs must be a positive integer, got {fx_setting.n_outputs}")
                        # Check if n_inputs and n_outputs are consistent with the preset
                        safe_name = create_safe_instance_name(fx_setting.fx_name)
                        fx_path = os.path.join(PLUGIN_PRESETS_DIR, f"{safe_name}.json")
                        try:
                            with open(fx_path, 'r') as pf:
                                preset_data = json.load(pf)
                            if preset_data.get('n_inputs') != fx_setting.n_inputs:
                                errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): n_inputs from preset ({preset_data.get('n_inputs')}) does not match n_inputs in YAML ({fx_setting.n_inputs}).")
                            if preset_data.get('n_outputs') != fx_setting.n_outputs:
                                errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): n_outputs from preset ({preset_data.get('n_outputs')}) does not match n_outputs in YAML ({fx_setting.n_outputs}).")
                        except FileNotFoundError:
                            errors.append(f"I/O Channel Validation: Preset definition file not found for '{fx_setting.fx_name}' at {fx_path} (needed for param validation).")
                        except json.JSONDecodeError as e:
                            errors.append(f"I/O Channel Validation: Error decoding JSON from preset file {fx_path}: {e}")
                        except Exception as e:
                            errors.append(f"I/O Channel Validation: Error processing preset file {fx_path} for '{fx_setting.fx_name}': {e}")
                        
                        # f. Sidechain Validation (Using Layer Info)
                        if fx_setting.sidechain_input is not None:
                            # Increment sidechain count for this chain
                            sidechain_count += 1
                            
                            # Rule 1: Only allow at most one sidechain-enabled plugin per chain
                            if sidechain_count > 1:
                                errors.append(f"Chain {current_chain_idx}: Contains multiple sidechain-enabled plugins. Only one sidechain input is allowed per chain.")
                                                    
                            sc_source_idx = fx_setting.sidechain_input
                            if not isinstance(sc_source_idx, int):
                                errors.append(f"Chain {current_chain_idx}, FX {fx_idx}: sidechain_input must be integer index or null.")
                            elif fx_setting.n_inputs <= 1: # Check n_inputs for sidechain capability
                                errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): sidechain_input is set, but plugin's n_inputs ({fx_setting.n_inputs}) is not > 1.")                                
                            elif sc_source_idx not in valid_chain_indices:
                                errors.append(f"Chain {current_chain_idx}, FX {fx_idx}: sidechain_input index {sc_source_idx} is invalid.")
                            elif sc_source_idx not in init_queue: # Check if the source is in the current processing layer, so that REAPER can set track send correctly
                                errors.append(f"Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}'): sidechain_input index {sc_source_idx} is not in the current layer ({current_layer}).")
                            else:
                                # Rule 2: The input of sidechain cannot come from a splitter output, because there is no mixed signal after the splitter
                                # (splitter outputs represent separated frequency bands, not suitable for sidechain)
                                source_chain = self.FxChains[sc_source_idx]
                                if source_chain.FxChain and source_chain.FxChain[-1].fx_type == "splitter":
                                    errors.append(f"Chain {current_chain_idx}, FX {fx_idx}: Sidechain input cannot come from a splitter output (chain {sc_source_idx}).")                                

                        # g. Splitter Validation
                        if fx_setting.fx_type.lower() == "splitter":
                            is_last_fx = (fx_idx == len(chain.FxChain) - 1)
                            if not is_last_fx:
                                errors.append(f"Splitter Rule Violation: Chain {current_chain_idx}, FX {fx_idx} ('{fx_setting.fx_name}') is a Splitter but not the last FX.")
                            else: # Is last FX and is Splitter
                                if fx_setting.n_outputs <= fx_setting.n_inputs: # Splitter must have more outputs than inputs
                                    errors.append(f"Chain {current_chain_idx}, Last FX ('{fx_setting.fx_name}'): Splitter must have n_outputs > n_inputs.")
                                else:
                                    expected_next_chains = fx_setting.n_outputs / fx_setting.n_inputs # Calculate expected next_chains count based on the Splitter's outputs vs inputs
                                    if len(chain.next_chains) > expected_next_chains:
                                        errors.append(f"Chain {current_chain_idx}: Splitter requires {expected_next_chains} next_chains (n_outputs/n_inputs), found {len(chain.next_chains)}.")
                                    elif len(chain.next_chains) < expected_next_chains:
                                        print(f"Warning: Chain {current_chain_idx}: Splitter has too few next_chains ({len(chain.next_chains)}), expected {expected_next_chains}. If this is intentional, ignore this warning.")

                # 2. Check next_chains type and content
                if not isinstance(chain.next_chains, dict):
                    errors.append(f"Chain {current_chain_idx}: next_chains must be a dictionary, got {type(chain.next_chains)}.")
                else:
                    for next_idx, weight in chain.next_chains.items():
                        if not isinstance(next_idx, int):
                            errors.append(f"Chain {current_chain_idx}: next_chains key '{next_idx}' must be an integer.")
                        elif next_idx not in valid_chain_indices:
                            errors.append(f"Chain {current_chain_idx}: next_chains key {next_idx} is an invalid chain index.")
                            
                        if not isinstance(weight, (float, int)):
                             errors.append(f"Chain {current_chain_idx}: next_chains weight for key {next_idx} must be a number, got {type(weight)}.")
                        elif weight < 0:
                             errors.append(f"Chain {current_chain_idx}: next_chains weight for key {next_idx} must be non-negative, got {weight}.")


                # Post-FX loop checks for the chain
                is_last_fx_splitter = chain.FxChain and chain.FxChain[-1].fx_type == "splitter"
                # Check count based on the number of keys in next_chains dict
                if len(chain.next_chains) > 1 and not is_last_fx_splitter:
                    errors.append(f"Splitter Rule Violation: Chain {current_chain_idx} has multiple next_chains ({len(chain.next_chains)}) but last FX is not Splitter.")

                # --- Update Successors ---
                for successor_idx in successors[current_chain_idx]:
                    in_degree[successor_idx] -= 1
                    if in_degree[successor_idx] == 0:
                        queue.append(successor_idx)
            current_layer += 1 # Move to the next layer
            
            
        # --- Post-Topological Sort Checks ---

        # 1. Cycle Detection
        if processed_chains_count != num_chains:
            unprocessed = [idx for idx in range(num_chains) if idx not in chain_layers]
            errors.append(f"Graph contains a cycle or unreachable nodes. Unprocessed/Unreachable chains: {unprocessed}")
            # Return early if cycle detected, as other checks might be invalid
            return sorted(list(set(errors)))

        # 2. Start Node Consistency Check (Re-check after sort)
        if actual_start_nodes != input_chain_indices:
             errors.append(f"Input Rule Violation: Mismatch between actual start nodes {actual_start_nodes} and designated input chains {input_chain_indices}.")

        # 3. Single Output Node Check (Nodes with empty next_chains dict)
        end_nodes = [idx for idx in range(num_chains) if not self.FxChains[idx].next_chains]
        if len(end_nodes) == 0:
            errors.append("Output Rule Violation: No end node found (chain with empty next_chains).")
        elif len(end_nodes) > 1:
            errors.append(f"Output Rule Violation: Multiple end nodes found: {end_nodes}. Expected exactly one.")
        else: # Exactly one end node
            final_chain_index = end_nodes[0]
            # Check final node doesn't end with splitter
            final_chain = self.FxChains[final_chain_index]
            if final_chain.FxChain and final_chain.FxChain[-1].fx_type == "splitter":
                errors.append(f"Splitter Rule Violation: Final output chain {final_chain_index} cannot end with a Splitter.")

        # Remove duplicates and return
        return sorted(list(set(errors)))
        
        
# Example Usage:
if __name__ == "__main__":
    yaml_file = '/workspaces/WildFX/proj_metadata/metadata.yaml' # Corrected path

    try:
        loaded_projects = Project.load_from_yaml(yaml_file)
        print(f"Successfully loaded {len(loaded_projects)} project(s).")

        # Example: Save back to a new file (optional)
        output_yaml_file = '/workspaces/WildFX/proj_metadata/metadata_saved.yaml' # Corrected path
        Project.save_to_yaml(loaded_projects, output_yaml_file)
        print(f"\nSaved projects back to {output_yaml_file}")

    except (FileNotFoundError, ValueError, TypeError, IOError) as e:
        print(f"An error occurred: {e}")

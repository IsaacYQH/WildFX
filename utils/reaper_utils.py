import os
import time
import re
import tempfile
from functools import partial
from tqdm import tqdm
import reapy
from reapy import reascript_api as RPR
from typing import Iterable
import shutil
import numpy as np
from .global_variables import NOT_INTERESTED_PARAMS

def add_send_by_channel(
    project: reapy.core.ReapyObject, 
    send_track_index: int, 
    receive_track_index: int,
    src_channel: list = [1, 2],  # Default to stereo channels 1+2
    dst_channel: list = [1, 2],  # Default to stereo channels 1+2
    send_volume: float = 1.0
):
    """
    Add a send from one track to another with precise channel routing control.
    
    Args:
        project: The REAPER project object
        send_track_index: Index of the track sending the signal
        receive_track_index: Index of the track receiving the signal
        src_channel: List of source channels [ch1, ch2]. Use single item for mono (e.g. [1])
        dst_channel: List of destination channels [ch1, ch2]. Use single item for mono (e.g. [3])
        send_volume: Volume level for the send (1.0 = 0dB)
    
    Examples:
        # Send mono from channel 1 to channel 3
        add_send_by_channel(proj, 0, 1, src_channel=[1], dst_channel=[3])
        
        # Send stereo from channels 1+2 to channels 3+4
        add_send_by_channel(proj, 0, 1, src_channel=[1, 2], dst_channel=[3, 4])
    """
    # Validate arguments
    if not isinstance(send_track_index, int) or send_track_index < 0:
        raise ValueError(f"Send track index must be a non-negative integer, got {send_track_index}")
    
    if not isinstance(receive_track_index, int) or receive_track_index < 0:
        raise ValueError(f"Receive track index must be a non-negative integer, got {receive_track_index}")
    
    if not isinstance(src_channel, list) or len(src_channel) == 0 or len(src_channel) > 2:
        raise ValueError(f"Source channel must be a list with 1 or 2 items, got {src_channel}")
    
    if not isinstance(dst_channel, list) or len(dst_channel) == 0 or len(dst_channel) > 2:
        raise ValueError(f"Destination channel must be a list with 1 or 2 items, got {dst_channel}")
    
    # Validate channel numbers
    for ch in src_channel:
        if not isinstance(ch, int) or ch < 1:
            raise ValueError(f"Channel numbers must be positive integers, got {ch} in src_channel")
    
    for ch in dst_channel:
        if not isinstance(ch, int) or ch < 1:
            raise ValueError(f"Channel numbers must be positive integers, got {ch} in dst_channel")
    
    if not isinstance(send_volume, (int, float)) or send_volume < 0:
        raise ValueError(f"Send volume must be a non-negative number, got {send_volume}")
    
    with reapy.inside_reaper():
        # Get the send track and receive track
        send_track = project.tracks[send_track_index]
        receive_track = project.tracks[receive_track_index]

        # Create a new send from the send track to the receive track
        send = send_track.add_send(receive_track)
        
        # Determine if source/destination are mono based on list length
        src_mono = len(src_channel) == 1
        dst_mono = len(dst_channel) == 1
        
        # Calculate source channel value based on REAPER's bit-masking system
        src_value = 0
        if src_mono:
            # For mono: bit 10 (1024) + (channel_num - 1)
            src_value = 1024 + (src_channel[0] - 1)
        else:
            # For stereo: the value is the index of the first channel pair
            # For channels 1+2: value=0, channels 2+3: value=1, etc.
            src_value = src_channel[0] - 1
            
        # Calculate destination channel value using the same logic
        dst_value = 0
        if dst_mono:
            dst_value = 1024 + (dst_channel[0] - 1)
        else:
            dst_value = dst_channel[0] - 1
        
        # Set the channel mapping for the send
        send.set_info('I_SRCCHAN', src_value)
        send.set_info('I_DSTCHAN', dst_value)

        # Set the volume of the send
        send.set_info('D_VOL', send_volume)
        
        return send
    
def delete_all_tracks():
    """
    Delete all tracks in the current REAPER project.
    
    This function uses the REAPER API to select all tracks and then delete them.
    """
    with reapy.inside_reaper():
        RPR.Main_OnCommand(40296, 0) # Select all tracks
        RPR.Main_OnCommand(40005, 0) # Remove selected tracks

def batch_render_fx(project: reapy.core.ReapyObject, input_audio_path_list: Iterable, output_audio_path_list: Iterable, FxChain_list: Iterable, TrackSend_list: Iterable | dict = None, tracks_to_unselect: Iterable = None, batch_size: int=os.cpu_count()-1, max_wait: int=30):    
    """
    Batch render audio through FX with different presets.
    """
    assert len(input_audio_path_list) == len(output_audio_path_list), "input_audio_path_list and output_audio_path_list must have the same length"
    assert len(input_audio_path_list) > 0 and len(FxChain_list) > 0 and len(input_audio_path_list) == len(FxChain_list), "input_audio_path_list and FxChain_list must be not empty and have the same length"
    # assert 0 < batch_size <= os.cpu_count(), "batch_size must be a positive integer less than or equal to the number of available CPU cores"

    # Check consistent nesting across all input iterables
    is_first_nested = isinstance(input_audio_path_list[0], (list, tuple))
    assert all(isinstance(x, (list, tuple)) == is_first_nested for x in input_audio_path_list), "All elements in input_audio_path_list must be consistently nested or non-nested"
    assert all(isinstance(x, (list, tuple)) == is_first_nested for x in output_audio_path_list), "output_audio_path_list nesting structure must match input_audio_path_list"
    assert all(isinstance(y, (list, tuple)) == is_first_nested for x in FxChain_list for y in x), "FxChain_list nesting structure must match input_audio_path_list"
    if TrackSend_list is not None:
        if not is_first_nested:
            raise TypeError("Only nested lists are supported for TrackSend_list")
        assert len(TrackSend_list) == len(input_audio_path_list), "TrackSend_list must have the same length as other input lists"
    if tracks_to_unselect is not None:
        if not is_first_nested:
            raise TypeError("Only nested lists are supported for tracks_to_unselect")
        assert len(tracks_to_unselect) == len(input_audio_path_list), "tracks_to_unselect must have the same length as other input lists"

    tmp_output_dir = tempfile.mkdtemp()

    # Ensure batch_size doesn't exceed the number of items
    batch_size = min(batch_size, len(input_audio_path_list)) if not is_first_nested else max([len(x) for x in input_audio_path_list])
    # Generate name set of temporary files based on REAPER's naming convention
    tmp_files_paths = [os.path.join(tmp_output_dir, f'tmp_output-{i:0{3}d}.wav') for i in range(1, batch_size+1)]

    if is_first_nested:
        input_chunks = input_audio_path_list
        output_chunks = output_audio_path_list
        fx_chunks = FxChain_list
        if TrackSend_list is not None:
            track_send_chunks = TrackSend_list
        if tracks_to_unselect is not None:
            unselect_chunks = tracks_to_unselect
    else:
        # Split inputs into chunks for batched processing
        input_chunks = [input_audio_path_list[i:i + batch_size] for i in range(0, len(input_audio_path_list), batch_size)]
        output_chunks = [output_audio_path_list[i:i + batch_size] for i in range(0, len(output_audio_path_list), batch_size)]
        fx_chunks = [FxChain_list[i:i + batch_size] for i in range(0, len(FxChain_list), batch_size)]
        if TrackSend_list is not None:
            track_send_chunks = [TrackSend_list[i:i + batch_size] for i in range(0, len(TrackSend_list), batch_size)]
        if tracks_to_unselect is not None:
            unselect_chunks = [tracks_to_unselect[i:i + batch_size] for i in range(0, len(tracks_to_unselect), batch_size)]
    
    with reapy.inside_reaper():
        # Remove all tracks before starting
        delete_all_tracks()
        
        # Render settings
        project.set_info_value('RENDER_SETTINGS', 3) # render selected tracks as stems
        project.set_info_string('RENDER_FILE', tmp_output_dir)
        project.set_info_string('RENDER_PATTERN', 'tmp_output')
        
        # Create the zip iterator
        zip_iterator = zip(input_chunks, output_chunks, fx_chunks)

        # Initialize tqdm wrapping the iterator, providing the total number of chunks
        outer_pbar = tqdm(zip_iterator, total=len(input_chunks), desc="Processing chunks", unit="chunk")

        
        # cache API references and Create partials with correct parameters
        add_audio_to_new_track = partial(RPR.InsertMedia, p1 = 1)
        for k, (input_chunk, output_chunk, fx_chunk) in enumerate(outer_pbar):
            # max_n_inputs = []
            # max_n_outputs = []
            # Create a new track for each audio file
            for j, (audio_path, FxChain) in enumerate(zip(input_chunk, fx_chunk)):
                if audio_path is None:
                    project.add_track(index = j)  # Add an empty track if no audio file is provided
                    # max_n_inputs.append(0)
                    # max_n_outputs.append(0)
                else:
                    # current_max_n_inputs = 0
                    # current_max_n_outputs = 0
                    assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"
                    add_audio_to_new_track(p0 = audio_path) # mode 1 means insert media to a new track
                    for fx_dict in FxChain:
                        assert "fx_name" in fx_dict, f"Missing 'fx_name' key in FxChain item: {fx_dict}"
                        assert "params" in fx_dict, f"Missing 'params' key in FxChain item for fx '{fx_dict.get('fx_name', 'UNKNOWN')}': {fx_dict}"
                        assert isinstance(fx_dict["params"], list) or isinstance(fx_dict["params"], dict), f"Invalid 'params' type in FxChain item for fx '{fx_dict.get('fx_name', 'UNKNOWN')}': {fx_dict}, should be a list."
                        fx = project.tracks[j].add_fx(fx_dict["fx_name"])
                        # Set FX parameters
                        if isinstance(fx_dict["params"], list):
                            preset_params = fx_dict["params"]
                        elif isinstance(fx_dict["params"], dict):
                            preset_params = list(fx_dict["params"].values())
                        assert len(preset_params) == len(fx.params), f"FX {fx_dict['fx_name']} has {len(fx.params)} parameters, but {len(preset_params)} were provided"
                        interested_params_list = [(i, v) for i, v in enumerate(preset_params) if v is not None] # if v is None meaning that we keep it as default, it will be ignored
                        for param_idx, param_value in interested_params_list:
                            fx.params[param_idx] = param_value
                        # if fx.n_inputs > current_max_n_inputs:
                        #     current_max_n_inputs = fx.n_inputs
                        # if fx.n_outputs > current_max_n_outputs:
                        #     current_max_n_outputs = fx.n_outputs
                    # max_n_inputs.append(current_max_n_inputs)
                    # max_n_outputs.append(current_max_n_outputs)
            
            # Set Track Sends (default number of channels for each track is 2)
            if TrackSend_list is not None:
                for track_send in track_send_chunks[k]:
                    # If one FxChain has multiple output FxChains and ends with a splitter, it cannot serve as a sidechain input to another track in the same layer;
                    # In our data_class.py, we have ruled out the cases when tracks with splitters to interact with tracks containing sidechain inputs
                    if fx_chunk[track_send]:
                        if fx_chunk[track_send][-1].get("fx_type") == "splitter":
                            n_output_channel = fx_chunk[track_send][-1].get("n_outputs")
                            project.tracks[track_send].set_info_value("I_NCHAN", n_output_channel)
                    # Process receive tracks
                    for i, receive_track_index in enumerate(track_send_chunks[k][track_send]):
                        sidechain_channel_info = [(fx_setting['n_inputs'], fx_setting['n_outputs']) for fx_setting in fx_chunk[receive_track_index] if fx_setting.get("sidechain_input")]
                        if len(sidechain_channel_info) > 1:
                            raise ValueError(f"Currently, only one sidechain input is supported. Found {len(sidechain_channel_info)} sidechain inputs in track {receive_track_index}.")
                        elif len(sidechain_channel_info) == 1: # currently, only support one sidechain input
                            # if the receive track has sidechain input, we need to set the input channel
                            n_input_channel, n_output_channel = sidechain_channel_info[0]
                            project.tracks[receive_track_index].set_info_value("I_NCHAN", n_input_channel)
                            source_channel = [1, 2] # default to stereo channels 1+2
                            receive_channel = list(range(n_output_channel+1, n_input_channel+1))
                        else: # splitter
                            source_channel = [2*i+item for item in [1, 2]] # handles mono and stereo with the same logic
                            receive_channel = [1, 2]
                        add_send_by_channel(project, track_send, receive_track_index, source_channel, receive_channel)
            
            
            # Select all tracks
            RPR.Main_OnCommand(40296, 0)
            # Unselect the tracks in tracks_to_unselect (typically the source tracks for splitters)
            if tracks_to_unselect is not None:
                for track_index in unselect_chunks[k]:
                    project.tracks[track_index].is_selected = False
            # Render project, using the most recent render settings
            RPR.Main_OnCommand(41824, 0)
            start_time = time.time()
            num_valid_output = len(output_chunk) - len(unselect_chunks[k]) if tracks_to_unselect is not None else len(output_chunk)
            files_pending = tmp_files_paths[:num_valid_output] if num_valid_output > 1 else [os.path.join(tmp_output_dir, f'tmp_output.wav')]
            files_done = files_pending[:] # a deep copy of the list
            
            # Verify all files were created
            while files_pending and (time.time() - start_time < max_wait):
                # Check which files still need to be created
                files_pending = [f for f in files_pending if not os.path.exists(f)]
                if files_pending:
                    # Still waiting for some files
                    time.sleep(0.5)
            if files_pending:
                missing_files = len(files_pending)
                raise TimeoutError(f"Render timed out after {max_wait} seconds. {missing_files} files were not created.")
            else:
                outer_pbar.set_postfix({"status": f"rendered {num_valid_output} files"})
            
            filtered_output_chunk = [output for output in output_chunk if output]
            # Move rendered files to the specified output directory
            for tmp_output, output_path in zip(files_done, filtered_output_chunk):
                # Move the rendered file to the output directory
                os.rename(tmp_output, output_path)
                            
            # Clean up by deleting all tracks after rendering       
            delete_all_tracks()
        # Clean up the temporary directory when done
        if os.path.exists(tmp_output_dir):
            shutil.rmtree(tmp_output_dir)
        outer_pbar.close()

def fx_get_metadata(project: reapy.core.ReapyObject, plugin_name: str, fx_type: str) -> dict:
    """
    Retrieve metadata for a specified plugin in a REAPER project.
    This function adds a temporary track to the project, inserts the specified
    plugin, retrieves its metadata (such as valid parameter values and number
    of input channels), and then removes the temporary track.
    Args:
        project (reapy.core.ReapyObject): The REAPER project where the plugin
            metadata will be retrieved.
        plugin_name (str): The name of the plugin to analyze.
    Returns:
        dict: A dictionary containing the plugin metadata with the following keys:
            - "fx_name" (str): The name of the plugin.
            - "num_channels" (int): The number of input channels for the plugin.
            - "params" (dict): A dictionary of valid parameter values for the plugin.
    """
    # Add the plugin to the track
    with reapy.inside_reaper():
        track = project.add_track()
        fx = track.add_fx(plugin_name)
            
        # get the valid values for each parameter
        valid_params = fx_get_valid_params(track.id, fx.index)
        
        # maps frequencies in Hertz (Hz) to the mel scale, which better represents human auditory perception.
        freq_keywords = [
            'hz', 'freq', 'crossover', 'frequency','pitch', 'filter', 
            'cutoff', 'spectral', 'spectrum', 'lowpass', 'highpass', 
            'bandpass', 'bandstop', 'lowcut', 'highcut'
            ]
        for param_name, param_value in valid_params.items():
            if any(freq_keyword in param_name.lower() for freq_keyword in freq_keywords):
                # Generate perceptually spaced frequencies
                valid_params[param_name] = generate_perceptual_frequency_list(param_value[0], param_value[-1], len(param_value))
        
        # store plugin details
        metadata = {
            "fx_name": plugin_name,
            "fx_type": fx_type,
            "n_inputs": fx.n_inputs,
            "n_outputs": fx.n_outputs,
            "valid_params": valid_params
        }
        
        # remove the track
        delete_all_tracks()
    
    return metadata

def fx_get_valid_params(track_id: str, fx_index: int, max_grid: int = 100) -> dict:
    """
    Retrieve all valid parameter values for a given FX effect in REAPER.
    
    This function analyzes each parameter in the specified FX effect and determines
    the set of valid values based on the parameter type. Parameters containing names
    from NOT_INTERESTED_PARAMS will have empty value lists.
    
    - For toggle/boolean parameters (is_toggle=True): Returns [min_val, max_val],
      typically [0, 1]
    - For continuous parameters (step=0) or parameters with too many steps 
      (steps > max_grid): Returns a list of evenly distributed sample values
      across the parameter's range with max_grid+1 points
    - For discrete parameters with a reasonable number of steps: Returns all valid
      incremental values from min_val to max_val using the specified step size
    - For parameters containing keywords in NOT_INTERESTED_PARAMS: Returns an empty list
    
    Parameters:
        track_id (str): The identifier of the track containing the FX
        fx_index (int): The index of the FX on the track
        max_grid (int, optional): Maximum number of steps to consider for discrete
                                 enumeration before switching to sampling.
                                 Default is 101.
    
    Returns:
        dict: A dictionary where:
              - Keys are parameter names (str)
              - Values are lists of valid normalized parameter values (list[float])
              - Parameters matching NOT_INTERESTED_PARAMS have empty lists
    """
    with reapy.inside_reaper():
        # cache API references
        get_num_params = RPR.TrackFX_GetNumParams
        # cache API references and Create partials with correct parameters
        get_param_name = partial(RPR.TrackFX_GetParamName, p0=track_id, p1=fx_index, p3="", p4=512)
        get_param = partial(RPR.TrackFX_GetParam, p0=track_id, p1=fx_index, p3=0, p4=0)
        get_param_step_size = partial(RPR.TrackFX_GetParameterStepSizes, 
                                     p0=track_id, p1=fx_index, p3=0, p4=0, p5=0, p6=0)
        
        n_params = get_num_params(track_id, fx_index)

        fx_parameters = {}
        for param_index in range(n_params):
            param_name = get_param_name(p2=param_index)[4]
            
            # Check if this is a not-interested parameter
            param_name_lower = param_name.lower()
            # The last 3 params of plugins are: Bypass, Wet, Delta. 
            # They are all built‑in parameters in REAPER.
            # To Reduce the number of params, we can ignore these.
            if any(not_interested in param_name_lower for not_interested in NOT_INTERESTED_PARAMS) or n_params-3 <= param_index < n_params:
                # If the parameter name is already in the dictionary, append an underscore
                # This would happen when parameter names of the plugin have conflict with REAPER built-in parameters
                if param_name in fx_parameters.keys():
                    param_name = f"{param_name}_{param_index}"
                # For not-interested parameters, use empty list
                fx_parameters[param_name] = []
                continue
                
            min_val, max_val = get_param(p2=param_index)[4:6]
            step, _, _, is_toggle = get_param_step_size(p2=param_index)[4:8]
            
            # Generate valid values based on parameter type
            if is_toggle:
                # Boolean parameter (toggle)
                valid_values = [min_val, max_val]
            elif step == 0 or ((max_val - min_val) / step + 1) > max_grid:
                # Too many steps - provide sampled values
                valid_values = [min_val + i * (max_val - min_val) / max_grid for i in range(max_grid + 1)]
            else:
                # Discrete parameter with reasonable number of steps
                valid_values = []
                current = min_val
                while current < max_val + (step * 0.5):  # Add half step for float precision
                    valid_values.append(current)
                    current += step
            
            fx_parameters[param_name] = valid_values
    return fx_parameters

def create_safe_instance_name(plugin_name):
    """
    Creates a safe instance name for folders and files based on the plugin name.
    Also appends the provider's name at the end.
    
    Args:
        plugin_name (str): The full plugin name
        
    Returns:
        str: A safe instance name suitable for file and folder names
    """
    # Extract provider name from within parentheses at the end, if present
    provider_match = re.search(r'\((.+?)\)$', plugin_name)
    provider = provider_match.group(1) if provider_match else "Unknown"
    
    # Remove plugin type prefix and manufacturer
    plugin_base_name = re.sub(r'^(VST3?|AU|CLAP|LV2|JS):\s*', '', plugin_name)
    plugin_base_name = re.sub(r'\(.+?\)$', '', plugin_base_name).strip()
    
    # Replace special characters with underscores, collapse multiple underscores
    safe_name = re.sub(r'[^\w\s-]', '_', plugin_base_name)
    safe_name = re.sub(r'\s+', '_', safe_name).strip('_')
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Append provider name
    safe_name = f"{safe_name}_{provider.replace(' ', '_')}"
    
    return safe_name

def generate_perceptual_frequency_list(min_freq, max_freq, num_points=100):
    """
    Generate a list of frequencies that are perceptually equally spaced
    using the mel scale.
    
    Args:
        min_freq: Minimum frequency in Hz
        max_freq: Maximum frequency in Hz
        num_points: Number of points to generate (default: 100)
        
    Returns:
        List of frequencies in Hz
    """
    # Convert to mel scale
    min_mel = 2595 * np.log10(1 + min_freq/700)
    max_mel = 2595 * np.log10(1 + max_freq/700)
    
    # Create equally spaced points in mel space
    mel_points = np.linspace(min_mel, max_mel, num_points)
    
    # Convert back to Hz
    hz_points = 700 * (10**(mel_points/2595) - 1)
    
    # Ensure exact min and max frequencies are preserved
    hz_points[0] = min_freq
    hz_points[-1] = max_freq
    
    return list(hz_points)

# def fx_get_interested_params_num(valid_params: dict) -> dict:
#     """
#     Calculates the number of parameters from the given dictionary that are considered "interested"
#     by filtering out parameters with names containing specific keywords.
#     Args:
#         valid_params (dict): A dictionary where keys are parameter names.
#     Returns:
#         int: The count of parameters that do not contain any of the following keywords 
#         (case-insensitive): 'program', 'sample rate', 'buffer size', 'bypass'.
#     """
#     names = valid_params.keys()
    
#     return len([name for name in names 
#                 if not any([not_interested_param in name.lower()
#                             for not_interested_param in NOT_INTERESTED_PARAMS])])

# def run_lua_script(script: str):
#     # Create Lua script path and output path
#     temp_dir = tempfile.mkdtemp()
#     lua_script_path = os.path.join(temp_dir, "list_plugins.lua")
    
#     # Write Lua script to file
#     with open(lua_script_path, "w") as f:
#         f.write(script)

#     # Run Reaper with the script
#     subprocess.run([
#         "reaper",
#         "-nosplash",
#         "-nonewinst",
#         lua_script_path
#     ])


# def get_all_reaper_plugins(output_dir="data"):
#     """
#     Retrieve a list of all installed Reaper plugins by executing a temporary Lua script.

#     This function creates a temporary Lua script that queries Reaper for its installed plugins
#     and writes the results to a CSV file. The CSV file includes details such as the plugin index,
#     name, identifier, type, filename, path, and whether it's an instrument. It then launches
#     Reaper in a headless mode to run the script and generate the CSV file.

#     Parameters:
#         output_dir (str): The directory where the CSV file will be stored. Defaults to "data".

#     Returns:
#         str: The file path to the generated CSV file containing the list of Reaper plugins.
#     """
#     # Ensure the output directory is absolute
#     output_dir = os.path.abspath(output_dir)
#     # Create data directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
#     # Define the output path for the CSV file
#     output_path = os.path.join(output_dir, "reaper_plugins.csv")

#     lua_script = """
#     -- Script to list all installed REAPER plugins and save to CSV
#     -- Output format: Index,Name,Identifier

#     local file = io.open("{0}", "w")

#     -- Write header
#     file:write("Index,Name,Identifier,Plugin_Type,Filename,Path,Is_Instrument\n")

#     local idx = 0
#     local count = 0
#     local retval, name, identifier

#     -- Enumerate all FX
#     repeat
#         retval, name, identifier = reaper.EnumInstalledFX(idx)
#         if retval then
#             -- Determine plugin type
#             local plugin_type = "unknown"
#             local is_instrument = "no"
#     local filename = ""
#     local path = ""

#     -- Extract filename and path from identifier
#     if identifier:find("<") then
#         -- Get the part before the < character
#         path = identifier:match("^(.+)<")
#         if path then
#             -- Extract filename from the end of the path
#             filename = path:match("([^\\/]+)$")
#             -- Remove filename from path to get directory
#             path = path:gsub(filename .. "$", "")
#         end
#     end
#     --     -- Determine plugin type based on name and identifier
#             if name:find("VST3i:") or name:find("VSTi:") or identifier:find("VSTi") then
#                 is_instrument = "yes"
#             end
            
#             if name:find("VST3:") or name:find("VST3i:") then
#                 plugin_type = "vst3"
#             elseif name:find("VST:") or name:find("VSTi:") or identifier:find(".dll") or identifier:find(".vst") or identifier:find(".so") or identifier:find(".dylib") then
#                 plugin_type = "vst2"
#             elseif name:find("Component:") or identifier:find(".component") then
#                 plugin_type = "component"
#             elseif name:find("JS:") then
#                 plugin_type = "js"
#             elseif name:find("AU:") then
#                 plugin_type = "au"
#             elseif name:find("CLAP:") or identifier:find(".clap") then
#                 plugin_type = "clap"
#             elseif name:find("LV2:") or identifier:find(".lv2") then
#                 plugin_type = "lv2"
#             end
            
#             -- Write to CSV
#             file:write(count .. "," .. name .. "," .. identifier .. "," .. 
#                     plugin_type .. "," .. filename .. "," .. 
#                     path .. "," .. is_instrument .. "\n")
#             count = count + 1
#         end
#         idx = idx + 1
#     until not retval

#     file:close()
#     """.format(output_path.replace("\\", "\\\\"))  # Escape backslashes for Windows paths

#     # Run the Lua script
#     run_lua_script(lua_script)
#     return output_path

# def add_audio_to_track(track_id: str, audio_file_path: str):
#     """
#     Add audio from a file to a REAPER track.
    
#     Args:
#         project: The REAPER project
#         track: The target track object
#         audio_file_path: Path to the audio file to add
        
#     Returns:
#         tuple: (item, take) - References to the created media item and take
#     """
#     # Create a new media item on the track
#     item = RPR.AddMediaItemToTrack(track_id)
    
#     # Add a take to the media item
#     take = RPR.AddTakeToMediaItem(item)
    
#     # Create a PCM source from the audio file and set it as the take's source
#     source = RPR.PCM_Source_CreateFromFile(audio_file_path)
#     RPR.SetMediaItemTake_Source(take, source)
    
#     # Get the source length to set the item length
#     source_length = RPR.GetMediaSourceLength(source, False)[0]
#     RPR.SetMediaItemLength(item, source_length, False)
    
#     # Position item at the start of the track
#     RPR.SetMediaItemPosition(item, 0, False)
    
#     # Update REAPER UI
#     RPR.UpdateArrange()
    
#     return source_length
    
# def render_audio_with_fx(project_id, output_file_path, start_time=0, end_time=None, **kwargs):
#     """
#     Render audio using REAPER's rendering engine and return the audio as numpy array.
    
#     Args:
#         project_id: The REAPER project ID
#         output_file_path: Path where the rendered audio will be saved
#         start_time: Start time for rendering (default: 0)
#         end_time: End time for rendering (default: project length)
    
#     Returns:
#         numpy.ndarray: Audio data in (channels, samples) format
#         int: Sample rate
#     """
#     with reapy.inside_reaper():
#         init_change_num = RPR.GetProjectStateChangeCount(project_id)
        
#         if end_time is None:
#             # Get project length if end time not specified
#             end_time = RPR.GetProjectLength(project_id)
        
#         # Set render bounds
#         RPR.GetSet_LoopTimeRange2(project_id, True, False, start_time, end_time, False)
        
#         # Split the output file path into directory and filename
#         output_dir = os.path.dirname(output_file_path)
#         output_filename = os.path.basename(output_file_path)
        
#         # Ensure output directory exists
#         os.makedirs(output_dir, exist_ok=True)
        
#         if kwargs.get("sample_rate") is not None:
#             sample_rate = kwargs["sample_rate"]
#             # Set sample rate
#             RPR.GetSetProjectInfo(project_id, "RENDER_SAMPLE_RATE", sample_rate, True)
        
#         # Set render path
#         RPR.GetSetProjectInfo_String(project_id, "RENDER_FILE", os.path.abspath(output_dir), True)
#         RPR.GetSetProjectInfo_String(project_id, "RENDER_PATTERN", output_filename, True)
 
#         # Configure render settings - 24-bit WAV
#         RPR.GetSetProjectInfo_String(project_id, "RENDER_FORMAT", "evaw", True)  # WAV format
#         RPR.GetSetProjectInfo_String(project_id, "RENDER_SETTINGS", "BITD24", True)  # 24-bit depth
        
#         # Render (execute render command)
#         RPR.Main_OnCommand(41824, 0)  # File: Render project, using the most recent render settings
        
#         # Wait for render to complete
#         max_wait = 30  # Maximum seconds to wait
#         start_time = time.time()
#         while not os.path.exists(output_file_path):
#             time.sleep(0.5)
#             if time.time() - start_time > max_wait:
#                 raise TimeoutError(f"Render timed out after {max_wait} seconds")
    
#     # Read audio using soundfile
#     audio_output, sample_rate = sf.read(output_file_path, always_2d=True)
#     # Convert to (channels, samples) format
#     audio_output = audio_output.T  # Important: transpose to match expected format
    
#     return audio_output, sample_rate
    
# def save_fx_chain_with_params(track_id, fx_index, params_dict, output_path):
#     """
#     Apply parameter settings to an FX and save it as a REAPER FX Chain file
    
#     Args:
#         track_id: REAPER track ID
#         fx_index: Index of the FX to save
#         params_dict: Dictionary of parameter names and values to apply
#         output_path: Path where the .RfxChain file will be saved
#     """
#     # First apply all parameters
#     for param_name, param_value in params_dict.items():
#         # Get the parameter index for this name
#         param_index = -1
#         for i in range(RPR.TrackFX_GetNumParams(track_id, fx_index)):
#             name = RPR.TrackFX_GetParamName(track_id, fx_index, i, "", 512)[4]
#             if name == param_name:
#                 param_index = i
#                 break
                
#         if param_index >= 0:
#             # Set the parameter value
#             RPR.TrackFX_SetParam(track_id, fx_index, param_index, param_value)
    
#     # Make sure the FX is visible in chain
#     RPR.TrackFX_Show(track_id, fx_index, 3)  # 3 = chain
    
#     # Use REAPER's save FX command
#     RPR.Main_OnCommand(40755, 0)  # Track: Save FX chain
    
#     # At this point, REAPER will show a save dialog
#     # Unfortunately, we can't directly control the save dialog via API
#     # You'll need to manually save the file when prompted
    
#     print(f"Please save the FX chain to: {output_path}")
#     return output_path
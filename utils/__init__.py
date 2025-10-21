import os,re

# Define the directory where plugin presets are stored
# Adjust this path relative to where data_class.py is run, or use an absolute path
PLUGIN_PRESETS_DIR = os.path.join(os.path.dirname(__file__), '../plugin_presets')
# Add all valid plugin types here
ALLOWED_FX_TYPES = {
    # EQ (filter)
    "eq", "splitter", 
    # Dynamics
    "compressor", "limiter", "gate", "expander", "clipper", "de-esser", "transient-shaper", 
    # Time-Based Effects
    "reverb", "delay", "echo", 
    # Modulation Effects
    "chorus", "flanger", "phaser", "tremolo", "vibrato", 
    # Distortion & Saturation
    "distortion", "saturation", "overdrive", 
    # Pitch & Time Manipulation
    "pitch-shifter", "autotune", "time-stretcher",
    # Spatial & Surround Effects
    "spatial"
    # "stereo-imager", "surround-panner", # Not yet supported. only support splitter in multi-channel output plugins
    }
# optional: you can add your own keywords to filter out parameters
NOT_INTERESTED_PARAMS = {'program', 'sample rate', 'buffer size', 'bypass'}

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

import os

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
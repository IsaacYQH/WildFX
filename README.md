<div align="center">
  <img src="pics/logo.png" alt="WildFX Logo">
</div>

# WildFX: A DAW-Powered Pipeline for In-the-Wild Audio FX Graph Modeling
This Repo is the official implementation of WildFX Dataset Generating pipeline.

We introduce WildFX, the first comprehensive end-to-end pipeline (to the best of our knowledge) for interfacing with and generating multitrack music datasets with heterogeneous AFx graphs derived from universal plugins including real, commercial plugin chains using Python. WildFX is containerized with Docker, enabling efficient execution of a professional DAW backend (specifically REAPER) on Linux-based research systems—environments where audio production software typically does not run natively. This architecture supports seamless integration of arbitrary commercial plugins across multiple formats (VST/VST3/LV2/CLAP), allowing researchers to capture the full complexity of professional audio processing, including advanced routing schemes such as sidechaining and multiband processing.
## 0. Metadata Samples with Demontrative Figure
### YAML Project Metadata Example
<pre><code class="language-yaml">
FxChains:
  - FxChain:
      - fx_name: "VST3: 3 Band EQ"
        fx_type: "eq"
        preset_index: 2
        params: []
        sidechain_input: null
    next_chains:
      1: 1
  - FxChain: []

input_audios:
  - audio_path: "vocals.wav"
    audio_type: "vocal"
    input_FxChain: 0

output_audio: "mixed_output.wav"
customized: true
</code></pre>

<pre><code class="language-json">
### JSON Plugin Preset Example
{
  "fx_name": "VST3: 3 Band EQ",
  "fx_type": "eq",
  "n_inputs": 2,
  "n_outputs": 2,
  "valid_params": {
    "Low": [0.0, 0.01, "...", 1.0],
    "Mid": [0.0, 0.01, "...", 1.0],
    "High": [0.0, 0.01, "...", 1.0]
  },
  "presets": [
    [null, null, null, 0.12, 0.69, 0.21],
    [null, null, null, 0.72, 0.63, 0.09],
    [null, null, null, 0.05, 0.00, 0.28]
  ]
}
</code></pre>

<p align="center"> <img src="pics/example.pdf" width="400" alt="Mixing Graph with the Provided Sample"> </p>

## 1. Docker Container Configuration
<div align="center">
  <img src="pics/deploy.pdf" alt="WildFX Deployment">
</div>
### 1.1. Set up host machine
#### 1.1.a. [Install docker](https://docs.docker.com/engine/install/)

#### 1.1.b. Add current user to docker group (the only step requiring sudo if docker is already installed!)
```
sudo usermod -aG docker <username>
```

#### 1.1.c. Create plugin folders in home directory
```
mkdir -p ~/.vst ~/.vst3 ~/.clap ~/.lv2
# Alternatively in /usr/local/lib or /usr/local/lib64 (if plugin is 64-bit)
# mkdir -p /usr/local/lib/vst /usr/local/lib/vst3 /usr/local/lib/clap /usr/local/lib/lv2
# mkdir -p /usr/local/lib64/vst /usr/local/lib64/vst3 /usr/local/lib64/clap /usr/local/lib64/lv2
```
Those are the locations where you should put the plugins that you want to use to generate the dataset.

### 1.2. Modify the container config file
The docker container can be built conveniently using IDEs configurate by `.devcontainer/devcontainer.json`. Here we use *Visual Studio Code* as example (or any other IDE built on Visual Studio Code including Cursor). 

Below are the arguments in `.devcontainer/devcontainer.json` that specified to user's system.

#### 1.2.1. Essential arguments
- "USER_UID": "1014",  // adjust this to your host's user ID, get by 'id -u'
- "USER_GID": "1015",  // adjust this to your host's user ID, get by 'id -g'
- "AUDIO_GID": "29"  // adjust this to your host's audio group ID, get by getent group audio
- 	"runArgs": ["--gpus=all", "--runtime=nvidia"]: if you did not set up the NVIDIA container runtime Toolkit on the host machine, you need to remove them. In fact, you basically don't need them at all in the container if you only need to generate the dataset but not train models inside the container.
- 	"mounts": ["source=/path/on/host,target=/path/in/container,type=bind,consistency=cached]: mount folder /path/on/host to /path/in/container inside of the container. Usually it could be your dataset folder and the plugin folders we mentioned in 1.1.c..

### 1.3. Build docker container
The Dockerfile is already provided in `.devcontainer/Dockerfile`. You can conveniently build the container by the *Dev Containers* Plugin in VS Code. Manual building by `docker run`, but not recommended. For manual building, we provide `.devcontainer/entrypoint.sh` to initialize the DAW.

> **Note**: by this step you should successfully get inside the container, so the following steps you should run inside of the container.

### 1.4 Install python dependencies
```
pip install -r requirements.txt
```

## 2. Install Plugins
### Install Linux plugins
You can either run the installation scripts from the provided or directly move the plugin files: `.vst`, `.vst3`, `clap`, `lv2` to the folders you made earlier to hold all your plugins.

### Install Windows plugins (.exe) via wine and yabridge
```
# Initialize wine environment
wine wineboot

# Run an application with a virtual display
xvfb-run wine <path-to-your-.exe-file> /silent # Many installers support silent mode

# Create directories for Linux VST plugins (if not exist)
mkdir -p ~/.vst ~/.vst3 ~/.clap ~/.lv2

# Add the VST2 and VST3 directory to yabridge
yabridgectl add "$HOME/.wine/drive_c/Program Files/Steinberg/VstPlugins"
yabridgectl add "$HOME/.wine/drive_c/Program Files/Common Files/VST3"

# Sync changes
yabridgectl sync
```



## 3. Start DAW (REAPER)
You can start REAPER and leave it in the background by
```
reaper -nosplash -nonewinst -noactivate &
```
or
```
tmux new-session -d -s reaper-session 'reaper -nosplash -nonewinst -noactivate' # Recommended, easier for managing
```
### 3.1. Test reapy connection
```
python utils/test_reapy.py
```
> If the test fails (either reporting error or stuck in importing), just try to kill REAPER by `tmux kill-session`, then reopen REAPER in the background.

## 4. Start Processing Your Dataset!
<div align="center">
  <img src="pics/workflow.pdf" alt="WildFX Workflow">
</div>
<div align="center">
  <img src="pics/layer.pdf" alt="WildFX Batch Processing">
</div>
### 4.1. Get your installed plugin list
Sometimes after running this commands, you need to restart REAPER.
```
reaper utils/plugin_get_list.lua
```


### 4.2. Generate presets
#### 4.2.a. Add the plugins you want to use in a .csv file if you want to process multiple plugins
Here's how you would create a plugin list file:
```
VST3: Graphic Equalizer x16 Stereo (LSP VST3),eq
VST3: Gate (SocaLabs),compressor
VST3: ZamCompX2 (Damien Zammit),compressor
VST3: FlyingDelay (superflyDSP),delay
```
#### 4.2.b. Usage Examples
```
# Use a plugin list file
python gen_presets.py --plugin-list my_plugins.csv

# Process a specific plugin with its type
python gen_presets.py --plugin-name "VST3: ZamCompX2 (Damien Zammit)" compressor

# Use the reduced set with a custom input file
python gen_presets.py --use-reduced-set --input-audio "/path/to/your/sample.wav"
```
### 4.3. Generate projects to YAML file
You can also read the docstrins in `gen_projects.py`
```
python gen_projects.py \
usage: gen_projects.py [-h] --dataset-name DATASET_NAME --dataset-dir DATASET_DIR --output-path OUTPUT_PATH --num-projects NUM_PROJECTS [--complexity COMPLEXITY] [--min-stems MIN_STEMS]
                       [--max-stems MAX_STEMS] [--max-chains MAX_CHAINS] [--min-chains MIN_CHAINS] [--sidechain-prob SIDECHAIN_PROB] [--splitter-prob SPLITTER_PROB] [--chain-depth CHAIN_DEPTH]
                       [--variable-density] [--density-range DENSITY_RANGE]

Generate audio mixing graphs from stems.

options:
  -h, --help            show this help message and exit
  --dataset-name DATASET_NAME
                        Identifier for the dataset.
  --dataset-dir DATASET_DIR
                        Root directory of dataset containing project folders.
  --output-path OUTPUT_PATH
                        Path to save generated projects metadata ending with ".yaml".
  --num-projects NUM_PROJECTS
                        Number of projects to generate.
  --complexity COMPLEXITY
                        Complexity level (0.0 to 1.0).
  --min-stems MIN_STEMS
                        Minimum number of stems to use per project.
  --max-stems MAX_STEMS
                        Maximum number of stems to use per project.
  --max-chains MAX_CHAINS
                        Maximum number of FX chains in a project.
  --min-chains MIN_CHAINS
                        Minimum number of FX chains in a project.
  --sidechain-prob SIDECHAIN_PROB
                        Chance of a compatible FX using sidechain.
  --splitter-prob SPLITTER_PROB
                        Chance of a chain ending with a splitter.
  --chain-depth CHAIN_DEPTH
                        Comma-separated probabilities for number of FX per chain.
  --variable-density    Randomly vary density parameters for each project instead of using fixed values
  --density-range DENSITY_RANGE
                        If variable-density is enabled, controls the range (+/-) for random variation
```
### 4.4. Render audio with REAPER and save the dataset
- `save_mode` is a important argument. You can set it to 'human-readable' to get `.wav` audio files and `.yaml` metadata in each project folderl; if you set to 'training-ready', then H5 files and `.gpickle` files for `networkx` graphs would be generated. You can choose both.
```
python main.py
```



## Trouble Shooting
`DisabledDistAPIWarning: Can't reach distant API. Please start REAPER, or call reapy.config.enable_dist_api() from inside REAPER to enable distant API.
  warnings.warn(errors.DisabledDistAPIWarning())`: sometimes if leaving the container too long, jack service and REAPER would be automatically killed. Restart jack service by
```
jackd -d dummy -r 44100 -p 1024 &
# or if you have audio hardware
jackd -d alsa -d hw:0 -r 44100 -p 1024 -P &
```
and restart reaper by
```
tmux new-session -d -s reaper-session 'reaper -nosplash -nonewinst'
```
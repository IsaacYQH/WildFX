"""Build and render a deterministic WildFX bundle for a DAFx presentation."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf

from utils import PLUGIN_PRESETS_DIR
from utils.data_class import ChainDefinition, FXSetting, InputAudio, Project
from utils.graph_diagram import render_project_diagram
from utils.main_utils import mix_audio_files

SAMPLE_RATE = 44_100
PRESENTATION_MAX_PEAK = 0.98
PLUGIN_NAMES = {
    "eq": "VST3: 3 Band EQ (DISTRHO)",
    "delay": "VST3: Samurai Delay (discoDSP)",
    "reverb": "VST3: Schroeder (discoDSP)",
    "compressor": "VST3: ZamCompX2 (Damien Zammit)",
    "splitter": "JS: 3-Band Splitter",
}
GRAPH_DESCRIPTIONS = {
    "graph_1": "Each dry stem follows its own instrument-specific FX chain before the final merge.",
    "graph_2": "Drums control bass compression while rhythm and music branches form separate processed submixes.",
    "graph_3": "Nested stem processing and submixes feed a multiband graph whose middle band controls low-band compression.",
}
GRAPH_TITLES = {
    "graph_1": "Instrument-Specific Processing",
    "graph_2": "Cross-Stem Sidechain & Submixes",
    "graph_3": "Nested Multiband Mixing Graph",
}


@dataclass(frozen=True)
class DemoGraphSpec:
    """A rendered Project plus its audience-facing diagram annotations."""

    key: str
    project: Project
    chain_labels: Dict[int, str]
    edge_labels: Dict[Tuple[int, int], str]


def _load_plugin_metadata() -> Dict[str, dict]:
    """Load the five curated plugin descriptions used by the demo."""
    metadata = {}
    for path in sorted(Path(PLUGIN_PRESETS_DIR).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
        metadata[item["fx_name"]] = item
    missing = set(PLUGIN_NAMES.values()) - set(metadata)
    if missing:
        raise FileNotFoundError(f"Missing demo plugin metadata: {sorted(missing)}")
    return metadata


def _closest_preset(plugin: dict, targets: Dict[str, float]) -> int:
    """Choose a deterministic preset closest to normalized target controls."""
    names = list(plugin["valid_params"].keys())
    indices = {name: names.index(name) for name in targets}
    spans = {
        name: max(
            float(
                max(plugin["valid_params"][name]) - min(plugin["valid_params"][name])
            ),
            1e-12,
        )
        for name in targets
    }
    candidates = []
    for preset_index, preset in enumerate(plugin["presets"]):
        if any(preset[index] is None for index in indices.values()):
            continue
        distance = sum(
            ((float(preset[indices[name]]) - target) / spans[name]) ** 2
            for name, target in targets.items()
        )
        candidates.append((distance, preset_index))
    if not candidates:
        raise ValueError(f"No preset satisfies targets {targets}")
    return min(candidates)[1]


def _fx(
    metadata: Dict[str, dict],
    fx_type: str,
    targets: Dict[str, float],
    sidechain_input: Optional[int] = None,
) -> FXSetting:
    """Create an FX setting backed by a curated deterministic preset."""
    plugin = metadata[PLUGIN_NAMES[fx_type]]
    return FXSetting(
        fx_name=plugin["fx_name"],
        fx_type=fx_type,
        preset_index=_closest_preset(plugin, targets),
        n_inputs=plugin["n_inputs"],
        n_outputs=plugin["n_outputs"],
        sidechain_input=sidechain_input,
    )


def _stereo(signal: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """Pan a mono signal using equal-power gains."""
    angle = (pan + 1.0) * math.pi / 4.0
    return np.column_stack((signal * math.cos(angle), signal * math.sin(angle)))


def _note_frequency(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _synthesize_stems(output_dir: Path, duration: float, seed: int) -> List[Path]:
    """Create four short, musical stereo stems for testing and rehearsal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = int(duration * SAMPLE_RATE)
    beat_seconds = 0.5

    drums = np.zeros(samples)
    for beat in np.arange(0.0, duration, beat_seconds):
        start = int(beat * SAMPLE_RATE)
        length = min(int(0.22 * SAMPLE_RATE), samples - start)
        local_time = np.arange(length) / SAMPLE_RATE
        phase = 2 * np.pi * (70 * local_time - 38 * local_time**2)
        drums[start : start + length] += 0.42 * np.sin(phase) * np.exp(-20 * local_time)
    for beat in np.arange(beat_seconds, duration, 2 * beat_seconds):
        start = int(beat * SAMPLE_RATE)
        length = min(int(0.16 * SAMPLE_RATE), samples - start)
        local_time = np.arange(length) / SAMPLE_RATE
        noise = rng.standard_normal(length)
        drums[start : start + length] += 0.12 * noise * np.exp(-28 * local_time)
    for beat in np.arange(0.25, duration, 0.25):
        start = int(beat * SAMPLE_RATE)
        length = min(int(0.055 * SAMPLE_RATE), samples - start)
        local_time = np.arange(length) / SAMPLE_RATE
        noise = rng.standard_normal(length + 1)
        high_passed_noise = np.diff(noise)
        drums[start : start + length] += (
            0.035 * high_passed_noise * np.exp(-48 * local_time)
        )
    drums_stereo = _stereo(np.tanh(drums), 0.0)

    bass = np.zeros(samples)
    bass_notes = [36, 36, 43, 41, 36, 36, 46, 43]
    for note_index, start_time in enumerate(np.arange(0.0, duration, beat_seconds)):
        start = int(start_time * SAMPLE_RATE)
        length = min(int(beat_seconds * SAMPLE_RATE), samples - start)
        local_time = np.arange(length) / SAMPLE_RATE
        frequency = _note_frequency(bass_notes[note_index % len(bass_notes)])
        envelope = np.minimum(local_time / 0.02, 1.0) * np.exp(-1.5 * local_time)
        bass[start : start + length] += (
            0.20
            * envelope
            * (
                np.sin(2 * np.pi * frequency * local_time)
                + 0.25 * np.sin(4 * np.pi * frequency * local_time)
            )
        )
    bass_stereo = _stereo(bass, -0.05)

    chords = np.zeros(samples)
    chord_roots = [48, 53, 46, 43]
    chord_duration = 2.0
    for chord_index, start_time in enumerate(np.arange(0.0, duration, chord_duration)):
        start = int(start_time * SAMPLE_RATE)
        length = min(int(chord_duration * SAMPLE_RATE), samples - start)
        local_time = np.arange(length) / SAMPLE_RATE
        envelope = np.minimum(local_time / 0.08, 1.0) * np.minimum(
            (chord_duration - local_time) / 0.25, 1.0
        )
        root = chord_roots[chord_index % len(chord_roots)]
        for interval in (0, 4, 7):
            frequency = _note_frequency(root + interval)
            chords[start : start + length] += (
                0.055 * envelope * np.sin(2 * np.pi * frequency * local_time)
            )
    chords_stereo = _stereo(chords, 0.35)

    lead = np.zeros(samples)
    melody = [72, 74, 75, 79, 77, 75, 74, 70]
    for note_index, start_time in enumerate(np.arange(0.0, duration, beat_seconds)):
        start = int(start_time * SAMPLE_RATE)
        length = min(int(0.42 * SAMPLE_RATE), samples - start)
        local_time = np.arange(length) / SAMPLE_RATE
        frequency = _note_frequency(melody[note_index % len(melody)])
        envelope = np.minimum(local_time / 0.015, 1.0) * np.exp(-3.2 * local_time)
        vibrato_phase = 2 * np.pi * frequency * local_time + 0.018 * np.sin(
            2 * np.pi * 5.2 * local_time
        )
        lead[start : start + length] += 0.13 * envelope * np.sin(vibrato_phase)
    lead_stereo = _stereo(lead, -0.35)

    stems = {
        "01_drums.wav": drums_stereo,
        "02_bass.wav": bass_stereo,
        "03_chords.wav": chords_stereo,
        "04_lead.wav": lead_stereo,
    }
    paths = []
    for filename, audio in stems.items():
        path = output_dir / filename
        sf.write(path, audio.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
        paths.append(path)
    return paths


def _copy_stems(source_dir: Path, output_dir: Path) -> List[Path]:
    """Copy presentation stems into a stable, numbered demo directory."""
    supported = {".wav", ".flac", ".ogg", ".aif", ".aiff"}
    sources = [
        path
        for path in sorted(source_dir.iterdir())
        if path.is_file() and path.suffix.lower() in supported
    ]
    if not sources:
        raise FileNotFoundError(f"No supported audio stems found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, source in enumerate(sources, start=1):
        destination = output_dir / f"{index:02d}_{source.name}"
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _project_inputs(stems: Iterable[Path]) -> List[InputAudio]:
    return [
        InputAudio(
            audio_path=str(path.resolve()),
            audio_type=path.stem,
            input_FxChain=index,
        )
        for index, path in enumerate(stems)
    ]


def _build_graph_specs(stems: List[Path], stem_gain: float) -> List[DemoGraphSpec]:
    """Build three increasingly complex, genuinely multitrack mixing graphs."""
    metadata = _load_plugin_metadata()
    inputs = _project_inputs(stems)
    num_stems = len(stems)

    if num_stems != 4:
        raise ValueError(
            "The curated presentation graphs require exactly four stems: "
            "drums, bass, chords, and lead"
        )

    # Graph 1: every stem has a distinct chain before the first and only merge.
    graph_1_final = num_stems
    graph_1_chains = [
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.58, "Mid": 0.44, "High": 0.62, "Master": 0.44},
                ),
                _fx(
                    metadata,
                    "compressor",
                    {
                        "Attack": 0.16,
                        "Release": 0.36,
                        "Ratio": 0.58,
                        "Threshold": 0.43,
                        "Makeup": 0.34,
                        "Sidechain": 0.0,
                    },
                ),
            ],
            next_chains={graph_1_final: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.61, "Mid": 0.45, "High": 0.42, "Master": 0.45},
                ),
                _fx(
                    metadata,
                    "compressor",
                    {
                        "Attack": 0.24,
                        "Release": 0.46,
                        "Ratio": 0.52,
                        "Threshold": 0.40,
                        "Makeup": 0.39,
                        "Sidechain": 0.0,
                    },
                ),
            ],
            next_chains={graph_1_final: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.46, "Mid": 0.58, "High": 0.64, "Master": 0.45},
                ),
                _fx(metadata, "reverb", {"mix": 0.18, "Decay": 0.30}),
            ],
            next_chains={graph_1_final: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.43, "Mid": 0.59, "High": 0.68, "Master": 0.44},
                ),
                _fx(
                    metadata,
                    "delay",
                    {
                        "Dry Through": 1.0,
                        "Feedback": 0.18,
                        "Wet Level": 0.22,
                    },
                ),
            ],
            next_chains={graph_1_final: stem_gain},
        ),
        ChainDefinition(FxChain=[], next_chains={}),
    ]

    # Graph 2: drums sidechain the bass; rhythm and music form separate buses.
    graph_2_rhythm = num_stems
    graph_2_music = graph_2_rhythm + 1
    graph_2_final = graph_2_music + 1
    graph_2_chains = [
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.60, "Mid": 0.46, "High": 0.58, "Master": 0.44},
                )
            ],
            next_chains={graph_2_rhythm: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "compressor",
                    {
                        "Attack": 0.12,
                        "Release": 0.44,
                        "Ratio": 0.72,
                        "Threshold": 0.34,
                        "Makeup": 0.38,
                        "Sidechain": 1.0,
                    },
                    sidechain_input=0,
                )
            ],
            next_chains={graph_2_rhythm: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.45, "Mid": 0.61, "High": 0.66, "Master": 0.44},
                )
            ],
            next_chains={graph_2_music: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "delay",
                    {
                        "Dry Through": 1.0,
                        "Feedback": 0.24,
                        "Wet Level": 0.28,
                    },
                )
            ],
            next_chains={graph_2_music: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "compressor",
                    {
                        "Attack": 0.20,
                        "Release": 0.38,
                        "Ratio": 0.48,
                        "Threshold": 0.47,
                        "Makeup": 0.34,
                        "Sidechain": 0.0,
                    },
                )
            ],
            next_chains={graph_2_final: 0.8},
        ),
        ChainDefinition(
            FxChain=[_fx(metadata, "reverb", {"mix": 0.23, "Decay": 0.40})],
            next_chains={graph_2_final: 0.8},
        ),
        ChainDefinition(FxChain=[], next_chains={}),
    ]

    # Graph 3: two sidechains, nested submixes, and a three-band branch/merge.
    merge = num_stems
    music = merge + 1
    full_mix = music + 1
    splitter = full_mix + 1
    low = splitter + 1
    middle = splitter + 2
    high = splitter + 3
    final = splitter + 4
    graph_3_chains = [
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.59, "Mid": 0.46, "High": 0.61, "Master": 0.44},
                )
            ],
            next_chains={merge: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "compressor",
                    {
                        "Attack": 0.11,
                        "Release": 0.46,
                        "Ratio": 0.74,
                        "Threshold": 0.33,
                        "Makeup": 0.38,
                        "Sidechain": 1.0,
                    },
                    sidechain_input=0,
                )
            ],
            next_chains={merge: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.45, "Mid": 0.60, "High": 0.65, "Master": 0.44},
                )
            ],
            next_chains={music: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "delay",
                    {
                        "Dry Through": 1.0,
                        "Feedback": 0.21,
                        "Wet Level": 0.25,
                    },
                )
            ],
            next_chains={music: stem_gain},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "compressor",
                    {
                        "Attack": 0.20,
                        "Release": 0.39,
                        "Ratio": 0.50,
                        "Threshold": 0.46,
                        "Makeup": 0.35,
                        "Sidechain": 0.0,
                    },
                )
            ],
            next_chains={full_mix: 1.0},
        ),
        ChainDefinition(
            FxChain=[_fx(metadata, "reverb", {"mix": 0.21, "Decay": 0.39})],
            next_chains={full_mix: 1.0},
        ),
        ChainDefinition(FxChain=[], next_chains={splitter: 1.0}),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "splitter",
                    {"Crossover 1 (Hz)": 480.0, "Crossover 2 (Hz)": 1430.0},
                )
            ],
            next_chains={low: 1.0, middle: 1.0, high: 1.0},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "compressor",
                    {
                        "Attack": 0.20,
                        "Release": 0.42,
                        "Ratio": 0.62,
                        "Threshold": 0.32,
                        "Makeup": 0.38,
                        "Sidechain": 1.0,
                    },
                    sidechain_input=middle,
                )
            ],
            next_chains={final: 1.0},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.73, "Mid": 0.57, "High": 0.33, "Master": 0.55},
                )
            ],
            next_chains={final: 1.0},
        ),
        ChainDefinition(
            FxChain=[
                _fx(
                    metadata,
                    "eq",
                    {"Low": 0.48, "Mid": 0.52, "High": 0.66, "Master": 0.42},
                )
            ],
            next_chains={final: 1.0},
        ),
        ChainDefinition(FxChain=[], next_chains={}),
    ]

    return [
        DemoGraphSpec(
            "graph_1",
            Project(
                graph_1_chains,
                inputs,
                output_audio="graph_1.wav",
                customized=False,
            ),
            {0: "Drums", 1: "Bass", 2: "Chords", 3: "Lead", 4: "Final Mix"},
            {},
        ),
        DemoGraphSpec(
            "graph_2",
            Project(
                graph_2_chains,
                inputs,
                output_audio="graph_2.wav",
                customized=False,
            ),
            {
                0: "Drums",
                1: "Bass",
                2: "Chords",
                3: "Lead",
                graph_2_rhythm: "Rhythm Submix",
                graph_2_music: "Music Submix",
                graph_2_final: "Final Mix",
            },
            {
                (graph_2_rhythm, graph_2_final): "\u00d70.80",
                (graph_2_music, graph_2_final): "\u00d70.80",
            },
        ),
        DemoGraphSpec(
            "graph_3",
            Project(
                graph_3_chains,
                inputs,
                output_audio="graph_3.wav",
                customized=False,
            ),
            {
                0: "Drums",
                1: "Bass",
                2: "Chords",
                3: "Lead",
                merge: "Rhythm Submix",
                music: "Music Submix",
                full_mix: "Full Mix Bus",
                splitter: "Multiband Split",
                low: "Low < 480 Hz",
                middle: "Mid 480–1430 Hz",
                high: "High > 1430 Hz",
                final: "Final Mix",
            },
            {},
        ),
    ]


def _audio_metrics(path: Path) -> dict:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Non-finite samples in {path}")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    mono = np.mean(audio, axis=1, dtype=np.float64)
    analysis_samples = mono[: min(len(mono), 262_144)]
    spectrum = np.abs(np.fft.rfft(analysis_samples * np.hanning(len(analysis_samples))))
    frequencies = np.fft.rfftfreq(len(analysis_samples), 1.0 / sample_rate)
    spectral_total = float(np.sum(spectrum))
    spectral_power = spectrum**2
    power_total = float(np.sum(spectral_power))
    spectral_centroid = (
        float(np.sum(frequencies * spectrum) / spectral_total)
        if spectral_total > 0
        else 0.0
    )
    band_masks = {
        "low_below_480_hz": frequencies < 480.0,
        "mid_480_to_1430_hz": (frequencies >= 480.0) & (frequencies < 1430.0),
        "high_above_1430_hz": frequencies >= 1430.0,
    }
    band_energy_fraction = {
        name: round(
            (
                float(np.sum(spectral_power[mask]) / power_total)
                if power_total > 0
                else 0.0
            ),
            6,
        )
        for name, mask in band_masks.items()
    }
    return {
        "sample_rate": sample_rate,
        "channels": int(audio.shape[1]),
        "samples": int(audio.shape[0]),
        "duration_seconds": round(audio.shape[0] / sample_rate, 4),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "crest_factor": round(peak / rms, 4) if rms > 0 else None,
        "spectral_centroid_hz": round(spectral_centroid, 2),
        "band_energy_fraction": band_energy_fraction,
    }


def _validate_playback(playback_dir: Path) -> Dict[str, dict]:
    """Validate safety and audible differences for the numbered playback files."""
    paths = [playback_dir / "00_dry_mix.wav"] + [
        playback_dir / f"0{index}_graph_{index}.wav" for index in range(1, 4)
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    metrics = {path.name: _audio_metrics(path) for path in paths}
    for path in paths:
        if metrics[path.name]["peak"] > PRESENTATION_MAX_PEAK:
            raise ValueError(
                f"{path} peak exceeds the presentation limit "
                f"({PRESENTATION_MAX_PEAK:.2f})"
            )
    dry, dry_sr = sf.read(paths[0], always_2d=True, dtype="float32")
    if metrics[paths[0].name]["rms"] < 1e-5:
        raise ValueError("Dry mix is effectively silent")
    for path in paths[1:]:
        processed, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        if sample_rate != dry_sr or processed.shape[1] != dry.shape[1]:
            raise ValueError(f"Format mismatch in {path}")
        common_samples = min(len(dry), len(processed))
        difference_rms = float(
            np.sqrt(np.mean((processed[:common_samples] - dry[:common_samples]) ** 2))
        )
        metrics[path.name]["difference_from_dry_rms"] = round(difference_rms, 6)
        dry_flat = dry[:common_samples].reshape(-1).astype(np.float64)
        processed_flat = processed[:common_samples].reshape(-1).astype(np.float64)
        if np.std(dry_flat) > 0 and np.std(processed_flat) > 0:
            correlation = float(np.corrcoef(dry_flat, processed_flat)[0, 1])
        else:
            correlation = 0.0
        metrics[path.name]["correlation_with_dry"] = round(correlation, 6)
        tail = processed[len(dry) :]
        metrics[path.name]["tail_rms"] = round(
            float(np.sqrt(np.mean(tail**2))) if len(tail) else 0.0,
            6,
        )
        if metrics[path.name]["rms"] < 1e-5 or difference_rms < 1e-5:
            raise ValueError(f"{path} is silent or indistinguishable from the dry mix")
    for filename in ("01_graph_1.wav", "02_graph_2.wav", "03_graph_3.wav"):
        if metrics[filename]["samples"] <= len(dry):
            raise ValueError(
                f"{filename} did not preserve its time-based processing tail"
            )
        if metrics[filename]["tail_rms"] < 1e-5:
            raise ValueError(f"{filename} has no audible delay/reverb tail")
    return metrics


def _build_demo_contents(
    output_dir: Path,
    stems_dir: Optional[Path] = None,
    duration: float = 12.0,
    stem_gain: float = 0.2,
    seed: int = 42,
    render: bool = True,
) -> Path:
    """Populate an empty directory with the DAFx demo bundle."""
    if stems_dir is None and duration <= 0:
        raise ValueError("duration must be positive when generating demo stems")
    if not 0 < stem_gain <= 1:
        raise ValueError("stem_gain must be greater than 0 and at most 1")
    dry_stems_dir = output_dir / "dry_stems"
    stems = (
        _copy_stems(stems_dir, dry_stems_dir)
        if stems_dir is not None
        else _synthesize_stems(dry_stems_dir, duration, seed)
    )
    stem_metrics = {path.name: _audio_metrics(path) for path in stems}
    stem_formats = {
        (metrics["sample_rate"], metrics["channels"])
        for metrics in stem_metrics.values()
    }
    if len(stem_formats) != 1:
        raise ValueError(
            "Presentation stems must share one sample rate and channel count"
        )
    _, stem_channels = next(iter(stem_formats))
    if stem_channels != 2:
        raise ValueError("Presentation stems must be stereo")
    for filename, metrics in stem_metrics.items():
        if metrics["rms"] < 1e-5:
            raise ValueError(f"Presentation stem is effectively silent: {filename}")
        if metrics["peak"] > PRESENTATION_MAX_PEAK:
            raise ValueError(f"Presentation stem exceeds the peak limit: {filename}")

    dry_mix_path = output_dir / "dry_mix.wav"
    mix_audio_files(
        [str(path) for path in stems],
        str(dry_mix_path),
        weights=[stem_gain] * len(stems),
    )
    dry_mix_metrics = _audio_metrics(dry_mix_path)
    if dry_mix_metrics["peak"] > PRESENTATION_MAX_PEAK:
        raise ValueError(
            "Dry mix exceeds the presentation peak limit; lower --stem-gain"
        )
    graph_specs = _build_graph_specs(stems, stem_gain)
    projects = [spec.project for spec in graph_specs]
    metadata_path = output_dir / "graphs.yaml"
    Project.save_to_yaml(projects, str(metadata_path))
    diagrams_dir = output_dir / "diagrams"
    diagram_files = {}
    for index, spec in enumerate(graph_specs, start=1):
        filenames = render_project_diagram(
            project=spec.project,
            output_directory=diagrams_dir,
            filename_stem=f"0{index}_{spec.key}",
            title=f"Graph {index} · {GRAPH_TITLES[spec.key]}",
            description=GRAPH_DESCRIPTIONS[spec.key],
            chain_labels=spec.chain_labels,
            edge_labels=spec.edge_labels,
        )
        diagram_files[spec.key] = {
            file_type: f"diagrams/{filename}"
            for file_type, filename in filenames.items()
        }

    manifest = {
        "seed": seed,
        "stem_gain": stem_gain,
        "dry_stems": [path.name for path in stems],
        "dry_stem_metrics": stem_metrics,
        "graphs": GRAPH_DESCRIPTIONS,
        "graph_titles": GRAPH_TITLES,
        "diagram_files": diagram_files,
        "rendered": False,
        "metrics": {"dry_mix.wav": dry_mix_metrics},
    }
    if render:
        from main import main as render_projects

        rendered_dir = output_dir / "rendered"
        succeeded = render_projects(
            save_mode="both",
            metadata_yaml_path=str(metadata_path),
            final_output_dir=str(rendered_dir),
            batch_size=max(8, len(stems) + 4),
            project_batch_size=3,
            ram_disk_gb=1.0,
            render_tail_seconds=3.0,
        )
        if not succeeded:
            raise RuntimeError("WildFX rendering failed; see the log above")

        playback_dir = output_dir / "playback"
        playback_dir.mkdir()
        shutil.copy2(dry_mix_path, playback_dir / "00_dry_mix.wav")
        for index in range(1, 4):
            source = rendered_dir / f"project_{index - 1:08d}" / f"graph_{index}.wav"
            shutil.copy2(source, playback_dir / f"0{index}_graph_{index}.wav")
        playback_order = [f"../dry_stems/{path.name}" for path in stems]
        playback_order.extend(
            ["00_dry_mix.wav"]
            + [f"0{index}_graph_{index}.wav" for index in range(1, 4)]
        )
        playlist_path = playback_dir / "playback_order.m3u8"
        playlist_path.write_text(
            "#EXTM3U\n" + "\n".join(playback_order) + "\n",
            encoding="utf-8",
        )
        manifest["metrics"] = _validate_playback(playback_dir)
        manifest["playback_order"] = playback_order
        manifest["rendered"] = True

    manifest_path = output_dir / "manifest.json"
    manifest_temp_path = output_dir / "manifest.json.tmp"
    with manifest_temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(manifest_temp_path, manifest_path)
    return output_dir


def build_demo(
    output_dir: Path,
    stems_dir: Optional[Path] = None,
    duration: float = 12.0,
    stem_gain: float = 0.2,
    seed: int = 42,
    render: bool = True,
) -> Path:
    """Create a complete demo bundle, removing partial output on failure."""
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _build_demo_contents(
            output_dir=output_dir,
            stems_dir=stems_dir,
            duration=duration,
            stem_gain=stem_gain,
            seed=seed,
            render=render,
        )
    except Exception:
        shutil.rmtree(output_dir)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and render the deterministic WildFX DAFx demo bundle."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dafx_demo"))
    parser.add_argument("--stems-dir", type=Path)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--stem-gain", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create stems, dry mix, and graph YAML without invoking REAPER.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = build_demo(
        output_dir=arguments.output_dir,
        stems_dir=arguments.stems_dir,
        duration=arguments.duration,
        stem_gain=arguments.stem_gain,
        seed=arguments.seed,
        render=not arguments.prepare_only,
    )
    print(f"DAFx demo bundle ready at {result.resolve()}")

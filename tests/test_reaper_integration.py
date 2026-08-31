import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from presentation_demo import (
    PLUGIN_NAMES,
    _closest_preset,
    _load_plugin_metadata,
    build_demo,
)
from utils.data_class import ChainDefinition, FXSetting, InputAudio, Project

pytestmark = pytest.mark.reaper


def _tone(path: Path, frequency: float, amplitude: float) -> None:
    sample_rate = 44_100
    time_axis = np.arange(sample_rate, dtype=np.float64) / sample_rate
    mono = amplitude * np.sin(2 * np.pi * frequency * time_axis)
    sf.write(
        path,
        np.column_stack((mono, mono)).astype(np.float32),
        sample_rate,
        subtype="FLOAT",
    )


def _sidechain_project(
    control_path: Path,
    main_path: Path,
    output_name: str,
    preset_index: int,
) -> Project:
    compressor = FXSetting(
        fx_name=PLUGIN_NAMES["compressor"],
        fx_type="compressor",
        preset_index=preset_index,
        n_inputs=3,
        n_outputs=2,
        sidechain_input=0,
    )
    return Project(
        [
            # Gain zero removes the control tone from the audible mix. The
            # post-FX/pre-fader send must still deliver it to channels 3/4.
            ChainDefinition([], {2: 0.0}),
            ChainDefinition([compressor], {2: 1.0}),
            ChainDefinition([], {}),
        ],
        [
            InputAudio(str(control_path.resolve()), "control", 0),
            InputAudio(str(main_path.resolve()), "main", 1),
        ],
        output_audio=output_name,
        customized=False,
    )


def _tone_magnitude(audio: np.ndarray, sample_rate: int, frequency: float) -> float:
    mono = np.mean(audio, axis=1)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    frequencies = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
    return float(spectrum[np.argmin(np.abs(frequencies - frequency))])


def test_chain_zero_sidechain_is_pre_fader_and_does_not_leak(tmp_path: Path) -> None:
    import reapy
    from reapy import reascript_api as RPR

    from main import main

    control_path = tmp_path / "control.wav"
    silent_control_path = tmp_path / "silent_control.wav"
    main_path = tmp_path / "main.wav"
    _tone(control_path, 70.0, 0.7)
    _tone(silent_control_path, 70.0, 0.0)
    _tone(main_path, 440.0, 0.2)

    metadata = _load_plugin_metadata()[PLUGIN_NAMES["compressor"]]
    preset_index = _closest_preset(
        metadata,
        {
            "Attack": 0.10,
            "Release": 0.40,
            "Ratio": 0.80,
            "Threshold": 0.70,
            "Makeup": 0.45,
            "Sidechain": 1.0,
        },
    )
    projects = [
        _sidechain_project(control_path, main_path, "controlled.wav", preset_index),
        _sidechain_project(
            silent_control_path,
            main_path,
            "uncontrolled.wav",
            preset_index,
        ),
    ]
    metadata_path = tmp_path / "sidechain.yaml"
    Project.save_to_yaml(projects, str(metadata_path))
    output_dir = tmp_path / "output"

    with reapy.inside_reaper():
        RPR.SetEditCurPos(0.5, False, False)
        active_project = reapy.Project()
        active_project.set_info_value("RENDER_SRATE", 48_000)
        active_project.set_info_value("RENDER_NORMALIZE", 1)

    assert main(
        save_mode="human-readable",
        metadata_yaml_path=str(metadata_path),
        final_output_dir=str(output_dir),
        batch_size=8,
        project_batch_size=2,
        ram_disk_gb=0,
        render_tail_seconds=0,
    )

    controlled, sample_rate = sf.read(
        output_dir / "project_00000000" / "controlled.wav",
        always_2d=True,
        dtype="float32",
    )
    uncontrolled, _ = sf.read(
        output_dir / "project_00000001" / "uncontrolled.wav",
        always_2d=True,
        dtype="float32",
    )
    assert controlled.shape == uncontrolled.shape == (44_100, 2)
    difference_rms = float(np.sqrt(np.mean((controlled - uncontrolled) ** 2)))
    leakage_ratio = _tone_magnitude(controlled, sample_rate, 70.0) / max(
        _tone_magnitude(controlled, sample_rate, 440.0),
        1e-12,
    )
    assert difference_rms > 1e-4
    assert leakage_ratio < 0.05


def test_render_is_float_wav_and_preserves_over_full_scale(tmp_path: Path) -> None:
    from main import main

    input_path = tmp_path / "over_full_scale.wav"
    samples = np.full((44_100, 2), 1.25, dtype=np.float32)
    sf.write(input_path, samples, 44_100, subtype="FLOAT")
    project = Project(
        [ChainDefinition([], {})],
        [InputAudio(str(input_path), "test", 0)],
        output_audio="render.wav",
    )
    metadata_path = tmp_path / "float_render.yaml"
    Project.save_to_yaml([project], str(metadata_path))
    output_dir = tmp_path / "float_render_output"

    assert main(
        save_mode="human-readable",
        metadata_yaml_path=str(metadata_path),
        final_output_dir=str(output_dir),
        batch_size=2,
        project_batch_size=1,
        ram_disk_gb=0,
        render_tail_seconds=0,
    )

    output_path = output_dir / "project_00000000" / "render.wav"
    info = sf.info(output_path)
    rendered, sample_rate = sf.read(
        output_path,
        always_2d=True,
        dtype="float32",
    )
    assert info.format == "WAV"
    assert info.subtype == "FLOAT"
    assert sample_rate == 44_100
    assert rendered.shape == samples.shape
    assert float(np.max(rendered)) == pytest.approx(1.25, abs=1e-6)


def test_full_presentation_bundle_renders_and_validates(tmp_path: Path) -> None:
    result = build_demo(tmp_path / "dafx", duration=2.0, render=True)
    with (result / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["rendered"] is True
    dry_samples = manifest["metrics"]["00_dry_mix.wav"]["samples"]
    for filename in ("01_graph_1.wav", "02_graph_2.wav", "03_graph_3.wav"):
        assert manifest["metrics"][filename]["samples"] > dry_samples
        assert manifest["metrics"][filename]["tail_rms"] > 1e-5
    assert len(list((result / "playback").glob("*.wav"))) == 4
    assert len(list((result / "diagrams").glob("*.svg"))) == 3
    assert len(list((result / "diagrams").glob("*.png"))) == 3
    playlist = (
        (result / "playback" / "playback_order.m3u8")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    entries = [line for line in playlist if line and not line.startswith("#")]
    assert len(entries) == 8
    assert all((result / "playback" / entry).resolve().exists() for entry in entries)


def test_sidechain_effect_before_splitter_renders_all_bands(tmp_path: Path) -> None:
    from main import main

    control_path = tmp_path / "control.wav"
    main_path = tmp_path / "main.wav"
    _tone(control_path, 70.0, 0.7)
    _tone(main_path, 440.0, 0.2)
    metadata = _load_plugin_metadata()
    compressor_metadata = metadata[PLUGIN_NAMES["compressor"]]
    splitter_metadata = metadata[PLUGIN_NAMES["splitter"]]
    compressor = FXSetting(
        fx_name=PLUGIN_NAMES["compressor"],
        fx_type="compressor",
        preset_index=_closest_preset(
            compressor_metadata,
            {
                "Attack": 0.10,
                "Release": 0.40,
                "Ratio": 0.80,
                "Threshold": 0.70,
                "Makeup": 0.45,
                "Sidechain": 1.0,
            },
        ),
        n_inputs=compressor_metadata["n_inputs"],
        n_outputs=compressor_metadata["n_outputs"],
        sidechain_input=0,
    )
    splitter = FXSetting(
        fx_name=PLUGIN_NAMES["splitter"],
        fx_type="splitter",
        preset_index=_closest_preset(
            splitter_metadata,
            {"Crossover 1 (Hz)": 480.0, "Crossover 2 (Hz)": 1430.0},
        ),
        n_inputs=splitter_metadata["n_inputs"],
        n_outputs=splitter_metadata["n_outputs"],
    )
    project = Project(
        [
            ChainDefinition([], {5: 0.0}),
            ChainDefinition([compressor, splitter], {2: 1.0, 3: 1.0, 4: 1.0}),
            ChainDefinition([], {5: 1.0}),
            ChainDefinition([], {5: 1.0}),
            ChainDefinition([], {5: 1.0}),
            ChainDefinition([], {}),
        ],
        [
            InputAudio(str(control_path), "control", 0),
            InputAudio(str(main_path), "main", 1),
        ],
        output_audio="combined.wav",
        customized=False,
    )
    metadata_path = tmp_path / "combined.yaml"
    Project.save_to_yaml([project], str(metadata_path))
    output_dir = tmp_path / "combined_output"
    assert main(
        save_mode="human-readable",
        metadata_yaml_path=str(metadata_path),
        final_output_dir=str(output_dir),
        batch_size=8,
        project_batch_size=1,
        ram_disk_gb=0,
        render_tail_seconds=0,
    )
    output, sample_rate = sf.read(
        output_dir / "project_00000000" / "combined.wav",
        always_2d=True,
        dtype="float32",
    )
    assert sample_rate == 44_100
    assert output.shape == (44_100, 2)
    assert float(np.sqrt(np.mean(output**2))) > 1e-4

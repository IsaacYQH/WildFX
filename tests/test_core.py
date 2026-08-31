import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from gen_presets import generate_params_task, load_process_audio_task
from gen_projects import assign_fx_to_chains
from presentation_demo import _load_plugin_metadata as _load_demo_metadata
from presentation_demo import build_demo
from utils import PLUGIN_PRESETS_DIR
from utils.data_class import (
    ChainDefinition,
    FXSetting,
    InputAudio,
    Project,
    find_closest,
)
from utils.main_utils import (
    metadata_to_networkx,
    mix_audio_files,
    prepare_batch,
    process_final_output,
    process_layer_with_tracksend_awareness,
)


def _write_wave(path: Path, values: np.ndarray, sample_rate: int = 44_100) -> None:
    sf.write(path, values.astype(np.float32), sample_rate, subtype="FLOAT")


def _splitter_project(audio_path: Path) -> Project:
    splitter = FXSetting(
        fx_name="JS: 3-Band Splitter",
        fx_type="splitter",
        params=[0.2, 0.7, None, None, None],
        n_inputs=2,
        n_outputs=6,
    )
    chains = [
        ChainDefinition([splitter], {1: 0.5, 2: 0.75, 3: 1.0}),
        ChainDefinition([], {4: 1.0}),
        ChainDefinition([], {4: 1.0}),
        ChainDefinition([], {4: 1.0}),
        ChainDefinition([], {}),
    ]
    return Project(
        chains,
        [InputAudio(str(audio_path), "test", 0)],
        customized=True,
    )


def test_mix_preserves_float_precision_length_and_weights(tmp_path: Path) -> None:
    first = np.zeros((1000, 2), dtype=np.float32)
    first[:500] = 1.6
    second = np.full((1000, 2), 1e-5, dtype=np.float32)
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    output_path = tmp_path / "mix.wav"
    _write_wave(first_path, first)
    _write_wave(second_path, second)

    mix_audio_files(
        [str(first_path), str(second_path)],
        str(output_path),
        weights=[0.5, 0.25],
    )

    mixed, sample_rate = sf.read(output_path, always_2d=True, dtype="float32")
    assert sample_rate == 44_100
    assert len(mixed) == 1000
    assert sf.info(output_path).subtype == "FLOAT"
    assert mixed[0, 0] == pytest.approx(0.8000025, abs=1e-7)
    assert mixed[-1, 0] == pytest.approx(2.5e-6, abs=1e-8)


def test_three_band_splitter_uses_source_plus_three_receivers(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    _write_wave(input_path, np.zeros((100, 2)))
    project = _splitter_project(input_path)

    batches, gains, sends, unselected, splitters = (
        process_layer_with_tracksend_awareness([(0, 0)], [project], batch_size=8)
    )

    assert batches == [[(0, 0)] * 4]
    assert gains == [[1.0, 0.5, 0.75, 1.0]]
    assert dict(sends[0]) == {0: [1, 2, 3]}
    assert unselected == [[0]]
    assert splitters == {(0, 0): 3}

    prepared = prepare_batch(
        0,
        batches[0],
        [project],
        0,
        {(0, 0): str(input_path)},
        {},
        splitters,
        str(tmp_path),
    )
    batch_inputs, batch_outputs, _, chain_outputs = prepared
    assert batch_inputs == [str(input_path), None, None, None]
    assert batch_outputs[0] is None
    assert len([path for path in batch_outputs if path is not None]) == 3
    assert len(chain_outputs[(0, 0)]) == 3


def test_tail_effects_are_not_batched_with_unrelated_graphs(tmp_path: Path) -> None:
    metadata = _load_demo_metadata()

    def make_project(fx_name: str, fx_type: str) -> Project:
        plugin = metadata[fx_name]
        valid_values = list(plugin["valid_params"].values())
        params = [
            value if allowed else None
            for value, allowed in zip(plugin["presets"][0], valid_values)
        ]
        return Project(
            [
                ChainDefinition(
                    [
                        FXSetting(
                            fx_name=fx_name,
                            fx_type=fx_type,
                            params=params,
                            n_inputs=plugin["n_inputs"],
                            n_outputs=plugin["n_outputs"],
                        )
                    ],
                    {},
                )
            ],
            [InputAudio(str(tmp_path / f"{fx_type}.wav"), fx_type, 0)],
        )

    eq_project = make_project(
        "VST3: 3 Band EQ (DISTRHO)",
        "eq",
    )
    reverb_project = make_project(
        "VST3: Schroeder (discoDSP)",
        "reverb",
    )
    batches, _, _, _, _ = process_layer_with_tracksend_awareness(
        [(0, 0), (1, 0)],
        [eq_project, reverb_project],
        batch_size=8,
    )
    assert batches == [[(0, 0)], [(1, 0)]]


def test_prepare_batch_fails_instead_of_returning_partial_data(tmp_path: Path) -> None:
    project = _splitter_project(tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError):
        prepare_batch(
            0,
            [(0, 0)] * 4,
            [project],
            0,
            {(0, 0): str(tmp_path / "missing.wav")},
            {},
            {(0, 0): 3},
            str(tmp_path),
        )


def test_sidechain_consumer_can_end_with_a_three_band_splitter(
    tmp_path: Path,
) -> None:
    compressor = FXSetting(
        fx_name="VST3: ZamCompX2 (Damien Zammit)",
        fx_type="compressor",
        params=[None] * 17,
        n_inputs=3,
        n_outputs=2,
        sidechain_input=0,
    )
    splitter = FXSetting(
        fx_name="JS: 3-Band Splitter",
        fx_type="splitter",
        params=[None] * 5,
        n_inputs=2,
        n_outputs=6,
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
            InputAudio(str(tmp_path / "control.wav"), "control", 0),
            InputAudio(str(tmp_path / "main.wav"), "main", 1),
        ],
    )
    batches, gains, sends, unselected, splitters = (
        process_layer_with_tracksend_awareness(
            [(0, 0), (0, 1)],
            [project],
            batch_size=8,
        )
    )
    assert batches == [[(0, 0), (0, 1), (0, 1), (0, 1), (0, 1)]]
    assert gains == [[0.0, 1.0, 1.0, 1.0, 1.0]]
    assert dict(sends[0]) == {0: [1], 1: [2, 3, 4]}
    assert unselected == [[1]]
    assert splitters == {(0, 1): 3}


def test_sidechain_from_chain_zero_is_kept_in_metadata_graph(tmp_path: Path) -> None:
    audio_path = tmp_path / "input.wav"
    _write_wave(audio_path, np.zeros((100, 2)))
    compressor = FXSetting(
        fx_name="VST3: ZamCompX2 (Damien Zammit)",
        fx_type="compressor",
        params=[None] * 17,
        n_inputs=3,
        n_outputs=2,
        sidechain_input=0,
    )
    project = Project(
        [
            ChainDefinition([], {2: 1.0}),
            ChainDefinition([compressor], {2: 1.0}),
            ChainDefinition([], {}),
        ],
        [
            InputAudio(str(audio_path), "control", 0),
            InputAudio(str(audio_path), "main", 1),
        ],
    )

    graph = metadata_to_networkx(
        project,
        {"VST3: ZamCompX2 (Damien Zammit)": [f"param_{index}" for index in range(17)]},
    )

    assert isinstance(graph, nx.DiGraph)
    assert graph.nodes["fx_0_0"]["type"] == "passthrough"
    assert graph.nodes["fx_2_0"]["type"] == "passthrough"
    assert graph.edges["fx_0_0", "fx_1_0"]["label"] == "control"
    assert set(graph.predecessors("fx_2_0")) == {"fx_0_0", "fx_1_0"}


def test_zero_is_a_valid_closest_parameter() -> None:
    assert find_closest([-1.0, 0.0, 1.0], 0.0) == 0.0


def test_parameter_generation_is_reproducible_across_worker_order() -> None:
    valid = {"A": [0.0, 0.5, 1.0], "Ignored": []}
    forward = [generate_params_task((index, valid, 42)) for index in range(20)]
    reverse = [
        generate_params_task((index, valid, 42)) for index in reversed(range(20))
    ]
    assert forward == list(reversed(reverse))
    assert forward == [generate_params_task((index, valid, 42)) for index in range(20)]


def test_preset_clustering_rejects_silent_and_nonfinite_audio(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.wav"
    silent_path = tmp_path / "silent.wav"
    nonfinite_path = tmp_path / "nonfinite.wav"
    samples = np.linspace(-0.25, 0.25, 256, dtype=np.float32)
    _write_wave(valid_path, np.column_stack((samples, samples)))
    _write_wave(silent_path, np.zeros((256, 2), dtype=np.float32))
    invalid = np.zeros((256, 2), dtype=np.float32)
    invalid[10, 0] = np.nan
    _write_wave(nonfinite_path, invalid)

    result = load_process_audio_task((7, str(valid_path), 44_100))
    assert result is not None
    assert result[0] == 7
    assert np.max(np.abs(result[1])) == pytest.approx(1.0)
    assert load_process_audio_task((0, str(silent_path), 44_100)) is None
    assert load_process_audio_task((0, str(nonfinite_path), 44_100)) is None


def test_assign_fx_keeps_the_compatible_splitter() -> None:
    base_plugin = {
        "fx_name": "EQ",
        "fx_type": "eq",
        "n_inputs": 2,
        "n_outputs": 2,
        "preset_count": 1,
        "presets": [[0.5]],
        "param_names": ["Gain"],
        "supports_sidechain": False,
        "is_splitter": False,
    }
    two_band = {
        **base_plugin,
        "fx_name": "two-band",
        "fx_type": "splitter",
        "n_outputs": 4,
        "is_splitter": True,
    }
    three_band = {
        **base_plugin,
        "fx_name": "three-band",
        "fx_type": "splitter",
        "n_outputs": 6,
        "is_splitter": True,
    }
    connections = {0: {1, 2, 3}, 1: {4}, 2: {4}, 3: {4}, 4: set()}
    chains = assign_fx_to_chains(
        connections=connections,
        chains_needing_splitters=[0],
        available_plugins_by_types={
            "eq": [base_plugin],
            "splitter": [two_band, three_band],
        },
        available_bands={1: [], 2: [two_band], 3: [three_band]},
        chain_layers={0: 0, 1: 1, 2: 1, 3: 1, 4: 2},
        chain_depth_distribution=[0.0, 1.0],
        sidechain_probability=0.0,
        rng=random.Random(7),
    )
    assert chains[0].FxChain[-1].fx_name == "three-band"


def test_zamcomp_meter_parameters_are_never_written(tmp_path: Path) -> None:
    path = Path(PLUGIN_PRESETS_DIR) / "ZamCompX2_Damien_Zammit.json"
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    names = list(metadata["valid_params"])
    gain_reduction = names.index("Gain Reduction")
    output_level = names.index("Output Level")
    assert metadata["valid_params"]["Gain Reduction"] == []
    assert metadata["valid_params"]["Output Level"] == []
    project = Project(
        [
            ChainDefinition(
                [
                    FXSetting(
                        "VST3: ZamCompX2 (Damien Zammit)",
                        "compressor",
                        preset_index=0,
                        n_inputs=3,
                        n_outputs=2,
                    )
                ],
                {},
            )
        ],
        [InputAudio(str(tmp_path / "input.wav"), "test", 0)],
        customized=False,
    )
    yaml_path = tmp_path / "project.yaml"
    Project.save_to_yaml([project], str(yaml_path))
    loaded = Project.load_from_yaml(str(yaml_path), num_cores=1)[0]
    params = loaded.FxChains[0].FxChain[0].params
    assert params[gain_reduction] is None
    assert params[output_level] is None


def test_presentation_bundle_prepares_deterministically(tmp_path: Path) -> None:
    first = build_demo(tmp_path / "first", duration=0.5, render=False)
    second = build_demo(tmp_path / "second", duration=0.5, render=False)
    first_mix, _ = sf.read(first / "dry_mix.wav", dtype="float32")
    second_mix, _ = sf.read(second / "dry_mix.wav", dtype="float32")
    assert np.array_equal(first_mix, second_mix)
    projects = Project.load_from_yaml(str(first / "graphs.yaml"), num_cores=1)
    assert len(projects) == 3
    assert [len(project.FxChains) for project in projects] == [5, 7, 12]
    sidechain_counts = [
        sum(
            fx.sidechain_input is not None
            for chain in project.FxChains
            for fx in chain.FxChain
        )
        for project in projects
    ]
    assert sidechain_counts == [0, 1, 2]
    assert (
        sum(
            fx.fx_type == "splitter"
            for chain in projects[2].FxChains
            for fx in chain.FxChain
        )
        == 1
    )
    assert {next(iter(chain.next_chains)) for chain in projects[0].FxChains[:4]} == {4}
    assert projects[1].FxChains[4].next_chains == {6: 0.8}
    assert projects[1].FxChains[5].next_chains == {6: 0.8}
    assert len(list((first / "dry_stems").glob("*.wav"))) == 4
    assert len(list((first / "diagrams").glob("*.svg"))) == 3
    assert len(list((first / "diagrams").glob("*.png"))) == 3
    for png_path in sorted((first / "diagrams").glob("*.png")):
        with Image.open(png_path) as image:
            assert image.size == (1920, 1080)
    assert "SIDECHAIN" not in (first / "diagrams/01_graph_1.svg").read_text()
    graph_2_svg = (first / "diagrams/02_graph_2.svg").read_text()
    assert "SIDECHAIN" in graph_2_svg
    assert graph_2_svg.count("\u00d70.80") == 2
    assert (first / "diagrams/03_graph_3.svg").read_text().count("SIDECHAIN") == 2
    with (first / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert set(manifest["diagram_files"]) == {"graph_1", "graph_2", "graph_3"}
    band_energy = manifest["metrics"]["dry_mix.wav"]["band_energy_fraction"]
    assert all(fraction > 0.001 for fraction in band_energy.values())


def test_failed_presentation_build_removes_partial_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "failed_demo"
    with pytest.raises(ValueError):
        build_demo(output_dir, duration=0, render=False)
    assert not output_dir.exists()


def test_both_mode_exports_float_h5_graph_wav_and_named_stem(tmp_path: Path) -> None:
    import h5py

    stem_path = tmp_path / "stem.wav"
    render_path = tmp_path / "render.wav"
    _write_wave(stem_path, np.full((200, 2), 0.125))
    _write_wave(render_path, np.full((200, 2), 0.25))
    project = Project(
        [ChainDefinition([], {})],
        [InputAudio(str(stem_path), "stem", 0)],
        output_audio="example.wav",
    )

    assert process_final_output(
        0,
        0,
        [project],
        {(0, 0): str(render_path)},
        {(0, 0)},
        {},
        str(tmp_path / "output"),
        "both",
        4,
    )
    project_dir = tmp_path / "output" / "project_00000000"
    assert (project_dir / "example.wav").exists()
    assert (project_dir / "mixing_graph.pickle").exists()
    assert (project_dir / "stems" / "input_audio_0.wav").exists()
    with h5py.File(project_dir / "audio_data.h5", "r") as handle:
        assert handle["output"].shape == (2, 200)
        assert handle["stems"]["input_audio_0"].shape == (2, 200)


def test_human_export_rejects_nonfinite_render(tmp_path: Path) -> None:
    stem_path = tmp_path / "stem.wav"
    render_path = tmp_path / "render.wav"
    _write_wave(stem_path, np.full((20, 2), 0.125))
    invalid = np.zeros((20, 2), dtype=np.float32)
    invalid[4, 0] = np.nan
    _write_wave(render_path, invalid)
    project = Project(
        [ChainDefinition([], {})],
        [InputAudio(str(stem_path), "stem", 0)],
        output_audio="example.wav",
    )
    assert not process_final_output(
        0,
        0,
        [project],
        {(0, 0): str(render_path)},
        {(0, 0)},
        {},
        str(tmp_path / "output"),
        "human-readable",
        4,
    )
    assert not (tmp_path / "output" / "project_00000000").exists()
    assert not list((tmp_path / "output").glob(".project_00000000.*"))

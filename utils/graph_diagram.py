"""Render presentation-ready diagrams from WildFX project definitions."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from utils.data_class import Project

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
NODE_WIDTH = 210
NODE_HEIGHT = 76

NODE_COLORS = {
    "input": "#E2E8F0",
    "eq": "#38BDF8",
    "compressor": "#FB923C",
    "delay": "#A78BFA",
    "reverb": "#2DD4BF",
    "splitter": "#4ADE80",
    "merge": "#94A3B8",
    "route": "#CBD5E1",
    "output": "#334155",
}
TEXT_COLORS = {"output": "#FFFFFF"}
AUDIO_EDGE_COLOR = "#475569"
CONTROL_EDGE_COLOR = "#A21CAF"


@dataclass
class DiagramNode:
    """A single node in a rendered mixing-graph diagram."""

    node_id: str
    title: str
    subtitle: str
    kind: str
    order: float
    level: int = 0
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class DiagramEdge:
    """An audio or sidechain connection between two diagram nodes."""

    source: str
    target: str
    kind: str = "audio"
    label: str = ""


def _effect_title(fx_type: str) -> str:
    """Return a compact, audience-facing effect name."""
    return {
        "eq": "3-Band EQ",
        "compressor": "ZamCompX2",
        "delay": "Samurai Delay",
        "reverb": "Schroeder Reverb",
        "splitter": "3-Band Splitter",
    }.get(fx_type, fx_type.replace("_", " ").title())


def _stem_title(audio_type: str) -> str:
    """Convert a metadata stem label into a short diagram title."""
    cleaned = audio_type
    while cleaned and (cleaned[0].isdigit() or cleaned[0] in "_- "):
        cleaned = cleaned[1:]
    return (cleaned or audio_type).replace("_", " ").title()


def _build_diagram_graph(
    project: Project,
    chain_labels: Mapping[int, str],
    edge_labels: Mapping[Tuple[int, int], str],
) -> Tuple[Dict[str, DiagramNode], List[DiagramEdge]]:
    """Convert one Project into explicit stem, FX, merge, and output nodes."""
    nodes: Dict[str, DiagramNode] = {}
    edges: List[DiagramEdge] = []
    chain_entries: Dict[int, str] = {}
    chain_exits: Dict[int, str] = {}
    predecessor_count = {index: 0 for index in range(len(project.FxChains))}
    input_order = {
        input_audio.input_FxChain: float(index)
        for index, input_audio in enumerate(project.input_audios)
    }

    for chain in project.FxChains:
        for target in chain.next_chains:
            predecessor_count[target] += 1

    for chain_index, chain in enumerate(project.FxChains):
        chain_order = input_order.get(chain_index, float(chain_index))
        previous_node: Optional[str] = None

        if not chain.next_chains and not chain.FxChain:
            output_node = f"output-{chain_index}"
            nodes[output_node] = DiagramNode(
                output_node,
                chain_labels.get(chain_index, "Final Mix"),
                project.output_audio or "output.wav",
                "output",
                chain_order,
            )
            chain_entries[chain_index] = output_node
            chain_exits[chain_index] = output_node
            continue

        if predecessor_count[chain_index] > 1:
            merge_node = f"merge-{chain_index}"
            nodes[merge_node] = DiagramNode(
                merge_node,
                chain_labels.get(chain_index, "Submix"),
                "merge",
                "merge",
                chain_order,
            )
            chain_entries[chain_index] = merge_node
            previous_node = merge_node

        for fx_index, fx in enumerate(chain.FxChain):
            fx_node = f"fx-{chain_index}-{fx_index}"
            nodes[fx_node] = DiagramNode(
                fx_node,
                _effect_title(fx.fx_type),
                chain_labels.get(chain_index, fx.fx_type.title()),
                fx.fx_type,
                chain_order,
            )
            if chain_index not in chain_entries:
                chain_entries[chain_index] = fx_node
            if previous_node is not None:
                edges.append(DiagramEdge(previous_node, fx_node))
            previous_node = fx_node

        if previous_node is None:
            route_node = f"route-{chain_index}"
            nodes[route_node] = DiagramNode(
                route_node,
                chain_labels.get(chain_index, "Route"),
                "audio bus",
                "route",
                chain_order,
            )
            chain_entries[chain_index] = route_node
            previous_node = route_node
        chain_exits[chain_index] = previous_node

    for input_index, input_audio in enumerate(project.input_audios):
        input_node = f"input-{input_index}"
        nodes[input_node] = DiagramNode(
            input_node,
            _stem_title(input_audio.audio_type),
            "dry stem",
            "input",
            float(input_index),
        )
        edges.append(DiagramEdge(input_node, chain_entries[input_audio.input_FxChain]))

    for chain_index, chain in enumerate(project.FxChains):
        targets = list(chain.next_chains.items())
        for target_index, _gain in targets:
            label = edge_labels.get((chain_index, target_index), "")
            edges.append(
                DiagramEdge(
                    chain_exits[chain_index],
                    chain_entries[target_index],
                    label=label,
                )
            )

        for fx_index, fx in enumerate(chain.FxChain):
            if fx.sidechain_input is not None:
                edges.append(
                    DiagramEdge(
                        chain_exits[fx.sidechain_input],
                        f"fx-{chain_index}-{fx_index}",
                        kind="control",
                        label="SIDECHAIN",
                    )
                )

    return nodes, edges


def _layout_graph(nodes: Dict[str, DiagramNode], edges: Iterable[DiagramEdge]) -> None:
    """Assign a deterministic left-to-right layered layout."""
    audio_edges = [edge for edge in edges if edge.kind == "audio"]
    incoming = {node_id: 0 for node_id in nodes}
    outgoing: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
    for edge in audio_edges:
        incoming[edge.target] += 1
        outgoing[edge.source].append(edge.target)

    ready = sorted(
        (node_id for node_id, degree in incoming.items() if degree == 0),
        key=lambda node_id: (nodes[node_id].order, node_id),
    )
    processed = 0
    while ready:
        node_id = ready.pop(0)
        processed += 1
        for target in outgoing[node_id]:
            nodes[target].level = max(nodes[target].level, nodes[node_id].level + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
                ready.sort(key=lambda item: (nodes[item].order, item))
    if processed != len(nodes):
        raise ValueError("Presentation diagram audio graph contains a cycle")

    max_level = max(node.level for node in nodes.values())
    left = 120.0
    right = CANVAS_WIDTH - 120.0
    for level in range(max_level + 1):
        level_nodes = sorted(
            (node for node in nodes.values() if node.level == level),
            key=lambda node: (node.order, node.node_id),
        )
        if not level_nodes:
            continue
        x = left if max_level == 0 else left + (right - left) * level / max_level
        if len(level_nodes) == 1:
            y_positions = [555.0]
        else:
            maximum_step = 190.0
            step = min(maximum_step, 660.0 / (len(level_nodes) - 1))
            first_y = 555.0 - step * (len(level_nodes) - 1) / 2.0
            y_positions = [first_y + step * index for index in range(len(level_nodes))]
        for node, y in zip(level_nodes, y_positions):
            node.x = x
            node.y = y


def _edge_geometry(
    source: DiagramNode, target: DiagramNode, kind: str
) -> Tuple[
    Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]
]:
    """Return cubic Bézier geometry that terminates at node borders."""
    if kind == "control" and abs(source.x - target.x) < 1.0:
        direction = 1.0 if target.y > source.y else -1.0
        start = (source.x, source.y + direction * NODE_HEIGHT / 2.0)
        end = (target.x, target.y - direction * NODE_HEIGHT / 2.0)
        offset = 150.0
        control_1 = (source.x + offset, start[1] + direction * 25.0)
        control_2 = (target.x + offset, end[1] - direction * 25.0)
        return start, control_1, control_2, end

    direction = 1.0 if target.x >= source.x else -1.0
    start = (source.x + direction * NODE_WIDTH / 2.0, source.y)
    end = (target.x - direction * NODE_WIDTH / 2.0, target.y)
    span = end[0] - start[0]
    control_1 = (start[0] + span * 0.42, start[1])
    control_2 = (end[0] - span * 0.42, end[1])
    return start, control_1, control_2, end


def _cubic_points(
    geometry: Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ],
    count: int = 40,
) -> List[Tuple[float, float]]:
    """Sample a cubic Bézier for Pillow rendering."""
    p0, p1, p2, p3 = geometry
    points = []
    for index in range(count + 1):
        t = index / count
        inverse = 1.0 - t
        points.append(
            (
                inverse**3 * p0[0]
                + 3 * inverse**2 * t * p1[0]
                + 3 * inverse * t**2 * p2[0]
                + t**3 * p3[0],
                inverse**3 * p0[1]
                + 3 * inverse**2 * t * p1[1]
                + 3 * inverse * t**2 * p2[1]
                + t**3 * p3[1],
            )
        )
    return points


def _svg_document(
    nodes: Mapping[str, DiagramNode],
    edges: Iterable[DiagramEdge],
    title: str,
    description: str,
) -> str:
    """Create the complete editable SVG document."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" '
            f'height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">'
        ),
        "<defs>",
        (
            '<marker id="audio-arrow" markerWidth="12" markerHeight="12" refX="10" '
            'refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 Z" '
            f'fill="{AUDIO_EDGE_COLOR}"/></marker>'
        ),
        (
            '<marker id="control-arrow" markerWidth="12" markerHeight="12" refX="10" '
            'refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 Z" '
            f'fill="{CONTROL_EDGE_COLOR}"/></marker>'
        ),
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="4" stdDeviation="6" flood-opacity="0.16"/></filter>',
        "</defs>",
        '<rect width="1920" height="1080" fill="#F8FAFC"/>',
        (
            '<text x="90" y="78" font-family="Arial, sans-serif" font-size="44" '
            f'font-weight="700" fill="#0F172A">{html.escape(title)}</text>'
        ),
        (
            '<text x="90" y="118" font-family="Arial, sans-serif" font-size="22" '
            f'fill="#475569">{html.escape(description)}</text>'
        ),
    ]

    for edge in edges:
        source = nodes[edge.source]
        target = nodes[edge.target]
        start, control_1, control_2, end = _edge_geometry(source, target, edge.kind)
        color = CONTROL_EDGE_COLOR if edge.kind == "control" else AUDIO_EDGE_COLOR
        marker = "control-arrow" if edge.kind == "control" else "audio-arrow"
        dash = ' stroke-dasharray="12 9"' if edge.kind == "control" else ""
        lines.append(
            (
                f'<path d="M {start[0]:.1f} {start[1]:.1f} C {control_1[0]:.1f} '
                f"{control_1[1]:.1f}, {control_2[0]:.1f} {control_2[1]:.1f}, "
                f'{end[0]:.1f} {end[1]:.1f}" fill="none" stroke="{color}" '
                f'stroke-width="4"{dash} marker-end="url(#{marker})"/>'
            )
        )
        if edge.label:
            points = _cubic_points((start, control_1, control_2, end), count=10)
            label_x, label_y = points[len(points) // 2]
            lines.append(
                (
                    f'<text x="{label_x:.1f}" y="{label_y - 10:.1f}" '
                    'text-anchor="middle" font-family="Arial, sans-serif" '
                    f'font-size="17" font-weight="700" fill="{color}" stroke="#F8FAFC" '
                    f'stroke-width="7" paint-order="stroke">{html.escape(edge.label)}</text>'
                )
            )

    for node in nodes.values():
        fill = NODE_COLORS.get(node.kind, "#CBD5E1")
        text_color = TEXT_COLORS.get(node.kind, "#0F172A")
        left = node.x - NODE_WIDTH / 2.0
        top = node.y - NODE_HEIGHT / 2.0
        lines.extend(
            [
                (
                    f'<rect x="{left:.1f}" y="{top:.1f}" width="{NODE_WIDTH}" '
                    f'height="{NODE_HEIGHT}" rx="16" fill="{fill}" stroke="#FFFFFF" '
                    'stroke-width="3" filter="url(#shadow)"/>'
                ),
                (
                    f'<text x="{node.x:.1f}" y="{node.y - 5:.1f}" text-anchor="middle" '
                    'font-family="Arial, sans-serif" font-size="23" font-weight="700" '
                    f'fill="{text_color}">{html.escape(node.title)}</text>'
                ),
                (
                    f'<text x="{node.x:.1f}" y="{node.y + 22:.1f}" text-anchor="middle" '
                    f'font-family="Arial, sans-serif" font-size="15" fill="{text_color}" '
                    f'opacity="0.82">{html.escape(node.subtitle)}</text>'
                ),
            ]
        )

    lines.extend(
        [
            '<line x1="90" y1="1005" x2="155" y2="1005" '
            f'stroke="{AUDIO_EDGE_COLOR}" stroke-width="4" marker-end="url(#audio-arrow)"/>',
            '<text x="175" y="1012" font-family="Arial, sans-serif" font-size="19" '
            'fill="#334155">audio signal</text>',
            '<line x1="350" y1="1005" x2="415" y2="1005" '
            f'stroke="{CONTROL_EDGE_COLOR}" stroke-width="4" stroke-dasharray="12 9" '
            'marker-end="url(#control-arrow)"/>',
            '<text x="435" y="1012" font-family="Arial, sans-serif" font-size="19" '
            'fill="#334155">sidechain control</text>',
            '<text x="1830" y="1012" text-anchor="end" font-family="Arial, sans-serif" '
            'font-size="18" font-weight="700" fill="#64748B">WildFX · DAFx presentation</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_font(size: int, bold: bool = False):
    """Load a common Linux/macOS font for deterministic PNG output."""
    from PIL import ImageFont

    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_arrow(draw, points: List[Tuple[float, float]], color: str) -> None:
    """Draw an arrowhead aligned to the final curve segment."""
    end_x, end_y = points[-1]
    prev_x, prev_y = points[-2]
    angle = math.atan2(end_y - prev_y, end_x - prev_x)
    length = 15.0
    spread = 0.55
    arrow = [
        (end_x, end_y),
        (
            end_x - length * math.cos(angle - spread),
            end_y - length * math.sin(angle - spread),
        ),
        (
            end_x - length * math.cos(angle + spread),
            end_y - length * math.sin(angle + spread),
        ),
    ]
    draw.polygon(arrow, fill=color)


def _png_image(
    nodes: Mapping[str, DiagramNode],
    edges: Iterable[DiagramEdge],
    title: str,
    description: str,
):
    """Render a 1920×1080 PNG using the same geometry as the SVG."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#F8FAFC")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(44, bold=True)
    description_font = _load_font(22)
    node_title_font = _load_font(23, bold=True)
    node_subtitle_font = _load_font(15)
    edge_font = _load_font(17, bold=True)
    legend_font = _load_font(19)
    footer_font = _load_font(18, bold=True)

    draw.text((90, 43), title, fill="#0F172A", font=title_font)
    draw.text((90, 94), description, fill="#475569", font=description_font)

    for edge in edges:
        geometry = _edge_geometry(nodes[edge.source], nodes[edge.target], edge.kind)
        points = _cubic_points(geometry)
        color = CONTROL_EDGE_COLOR if edge.kind == "control" else AUDIO_EDGE_COLOR
        if edge.kind == "control":
            for index in range(0, len(points) - 1, 2):
                draw.line([points[index], points[index + 1]], fill=color, width=4)
        else:
            draw.line(points, fill=color, width=4, joint="curve")
        _draw_arrow(draw, points, color)
        if edge.label:
            label_x, label_y = points[len(points) // 2]
            box = draw.textbbox((0, 0), edge.label, font=edge_font)
            width = box[2] - box[0]
            height = box[3] - box[1]
            draw.rounded_rectangle(
                (
                    label_x - width / 2 - 7,
                    label_y - height - 16,
                    label_x + width / 2 + 7,
                    label_y + 2,
                ),
                radius=5,
                fill="#F8FAFC",
            )
            draw.text(
                (label_x - width / 2, label_y - height - 11),
                edge.label,
                fill=color,
                font=edge_font,
            )

    for node in nodes.values():
        fill = NODE_COLORS.get(node.kind, "#CBD5E1")
        text_color = TEXT_COLORS.get(node.kind, "#0F172A")
        bounds = (
            node.x - NODE_WIDTH / 2,
            node.y - NODE_HEIGHT / 2,
            node.x + NODE_WIDTH / 2,
            node.y + NODE_HEIGHT / 2,
        )
        draw.rounded_rectangle(
            bounds,
            radius=16,
            fill=fill,
            outline="#FFFFFF",
            width=3,
        )
        title_box = draw.textbbox((0, 0), node.title, font=node_title_font)
        subtitle_box = draw.textbbox((0, 0), node.subtitle, font=node_subtitle_font)
        draw.text(
            (node.x - (title_box[2] - title_box[0]) / 2, node.y - 29),
            node.title,
            fill=text_color,
            font=node_title_font,
        )
        draw.text(
            (node.x - (subtitle_box[2] - subtitle_box[0]) / 2, node.y + 7),
            node.subtitle,
            fill=text_color,
            font=node_subtitle_font,
        )

    draw.line((90, 1005, 155, 1005), fill=AUDIO_EDGE_COLOR, width=4)
    _draw_arrow(draw, [(90, 1005), (155, 1005)], AUDIO_EDGE_COLOR)
    draw.text((175, 993), "audio signal", fill="#334155", font=legend_font)
    for x in range(350, 416, 18):
        draw.line((x, 1005, min(x + 10, 415), 1005), fill=CONTROL_EDGE_COLOR, width=4)
    _draw_arrow(draw, [(350, 1005), (415, 1005)], CONTROL_EDGE_COLOR)
    draw.text((435, 993), "sidechain control", fill="#334155", font=legend_font)
    footer = "WildFX · DAFx presentation"
    footer_box = draw.textbbox((0, 0), footer, font=footer_font)
    draw.text(
        (1830 - (footer_box[2] - footer_box[0]), 993),
        footer,
        fill="#64748B",
        font=footer_font,
    )
    return image


def render_project_diagram(
    project: Project,
    output_directory: Path,
    filename_stem: str,
    title: str,
    description: str,
    chain_labels: Mapping[int, str],
    edge_labels: Optional[Mapping[Tuple[int, int], str]] = None,
) -> Dict[str, str]:
    """Render editable SVG and 1920×1080 PNG files for one Project.

    Args:
        project: The exact WildFX project used for audio rendering.
        output_directory: Directory receiving both diagram formats.
        filename_stem: Shared filename without an extension.
        title: Large presentation title.
        description: One-line explanation under the title.
        chain_labels: Audience-facing labels for graph chain indices.
        edge_labels: Optional labels for selected chain-to-chain connections.

    Returns:
        Relative diagram filenames keyed by ``svg`` and ``png``.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    nodes, edges = _build_diagram_graph(
        project,
        chain_labels,
        edge_labels or {},
    )
    _layout_graph(nodes, edges)
    svg_path = output_directory / f"{filename_stem}.svg"
    png_path = output_directory / f"{filename_stem}.png"
    svg_path.write_text(
        _svg_document(nodes, edges, title, description),
        encoding="utf-8",
    )
    _png_image(nodes, edges, title, description).save(png_path, format="PNG")
    return {"svg": svg_path.name, "png": png_path.name}

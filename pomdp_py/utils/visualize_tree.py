"""Tree visualization for pomdp-py MCTS search trees.

Supports POUCT, POMCP, BA-POUCT, and BAMCP planners — any algorithm that
builds a VNode/QNode tree accessible via ``planner.tree`` or ``agent.tree``.

Example usage::

    from pomdp_py.utils.visualize_tree import visualize_tree, save_tree
    import matplotlib.pyplot as plt

    action = planner.plan(agent)

    # Interactive display
    visualize_tree(agent.tree, max_depth=4)
    plt.show()

    # Save as SVG (zoomable — recommended for large trees)
    save_tree(agent.tree, "tree.svg", max_depth=5, title="RockSample POUCT")

    # Save as PNG
    save_tree(agent.tree, "tree.png", max_depth=3, dpi=150)

Tips for large trees:
    - Use ``min_visits`` to prune low-confidence branches, e.g.
      ``min_visits=root.num_visits // 100`` keeps the top 1% of paths.
    - SVG output lets you zoom in arbitrarily in any browser or vector editor.
    - ``x_gap`` controls horizontal breathing room between leaf nodes.
    - ``y_gap`` controls vertical spacing between tree levels (in inches).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Node type detection
# ---------------------------------------------------------------------------

def _is_qnode(node):
    """True if node is a QNode — detected by absence of ``argmax`` (VNode-only)."""
    return not hasattr(node, "argmax")


# ---------------------------------------------------------------------------
# Visible-children collection with filters
# ---------------------------------------------------------------------------

def _collect_visible_children(node, depth, max_depth, min_visits):
    """Return list of (key, child) pairs that pass the depth/visit filters."""
    if max_depth is not None and depth >= max_depth:
        return []
    children = [
        (key, child)
        for key, child in node.children.items()
        if child.num_visits >= min_visits
    ]
    children.sort(key=lambda kc: kc[1].num_visits, reverse=True)
    return children


# ---------------------------------------------------------------------------
# Layout: positions are in units of (x_gap, y_gap) inches
# ---------------------------------------------------------------------------

def _assign_positions(root, max_depth, min_visits):
    """
    Assign integer-unit (x, y) layout coordinates to every visible node.
    x is in leaf-slot units, y is in depth units (negative = downward).

    Returns
    -------
    positions : dict  id(node) -> (x, y)
    node_info : dict  id(node) -> (node, depth, key_from_parent)
    """
    positions = {}
    node_info = {}
    x_counter = [0.0]

    def _layout(node, depth, key_from_parent):
        children = _collect_visible_children(node, depth, max_depth, min_visits)
        if not children:
            x = x_counter[0]
            x_counter[0] += 1.0
        else:
            child_xs = []
            for key, child in children:
                _layout(child, depth + 1, key)
                child_xs.append(positions[id(child)][0])
            x = (child_xs[0] + child_xs[-1]) / 2.0

        positions[id(node)] = (x, -depth)
        node_info[id(node)] = (node, depth, key_from_parent)

    _layout(root, 0, None)
    return positions, node_info


def _build_parent_map(root, max_depth, min_visits):
    parent_map = {}

    def _walk(node, depth):
        for _, child in _collect_visible_children(node, depth, max_depth, min_visits):
            parent_map[id(child)] = id(node)
            _walk(child, depth + 1)

    _walk(root, 0)
    return parent_map


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_VNODE_FACE = "#5B9BD5"
_VNODE_EDGE = "#2E75B6"
_QNODE_FACE = "#F4A261"
_QNODE_EDGE = "#C96B1A"
_TEXT_COLOR = "#1a1a1a"
_BG_COLOR   = "#F7F9FC"


def _edge_color(frac):
    lo = np.array([0xAA, 0xBD, 0xD4]) / 255.0
    hi = np.array([0x2E, 0x75, 0xB6]) / 255.0
    return lo + frac * (hi - lo)


# ---------------------------------------------------------------------------
# Bezier S-curve edges (drawn in data coordinates)
# ---------------------------------------------------------------------------

def _bezier_curve(x0, y0, x1, y1, n=40):
    """Cubic Bezier S-curve between two data-coordinate points."""
    dy = abs(y1 - y0)
    tension = 0.5
    cp1x, cp1y = x0, y0 - dy * tension
    cp2x, cp2y = x1, y1 + dy * tension
    t = np.linspace(0, 1, n)
    xs = (1-t)**3*x0 + 3*(1-t)**2*t*cp1x + 3*(1-t)*t**2*cp2x + t**3*x1
    ys = (1-t)**3*y0 + 3*(1-t)**2*t*cp1y + 3*(1-t)*t**2*cp2y + t**3*y1
    return xs, ys


def _draw_edges(ax, positions, node_info, parent_map, root_visits):
    max_lw, min_lw = 4.0, 0.5
    root_visits = max(root_visits, 1)

    for nid, (node, depth, _) in node_info.items():
        if nid not in parent_map:
            continue
        pid = parent_map[nid]
        x0, y0 = positions[pid]
        x1, y1 = positions[nid]
        frac = node.num_visits / root_visits
        lw = min_lw + (max_lw - min_lw) * frac
        color = _edge_color(frac)
        alpha = 0.4 + 0.6 * frac

        xs, ys = _bezier_curve(x0, y0, x1, y1)
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", zorder=1, transform=ax.transData)


# ---------------------------------------------------------------------------
# Node drawing — markers in display space, labels in offset points
# ---------------------------------------------------------------------------

_MAX_LABEL_CHARS = 14


def _truncate(s, max_chars=_MAX_LABEL_CHARS):
    s = str(s)
    return s if len(s) <= max_chars else s[:max_chars - 1] + "…"


def _key_str(key):
    if key is None:
        return "root"
    s = str(key)
    if s == "None" or s == "Observation(None)":
        return "∅"
    return _truncate(s)


def _node_label(node, key_from_parent):
    key = _key_str(key_from_parent)
    if _is_qnode(node):
        key_line   = f"[{key}]"
        stats_line = f"N:{node.num_visits}\nQ:{node.value:.3f}"
    else:
        try:
            v_str = f"{node.value:.3f}"
        except (ValueError, ZeroDivisionError):
            v_str = "—"
        key_line   = key
        stats_line = f"N:{node.num_visits}\nV:{v_str}"
    return f"{key_line}\n{stats_line}"


def _draw_nodes(ax, positions, node_info, markersize, fontsize):
    """
    Draw nodes as fixed-size markers (display coordinates) so they always
    appear as true circles/squares regardless of axis aspect ratio.
    Labels are anchored via annotate with offset in points.
    """
    label_offset_pts = -(markersize / 2 + 4)  # points below node centre

    for nid, (node, depth, key) in node_info.items():
        x, y = positions[nid]
        label = _node_label(node, key)

        if _is_qnode(node):
            ax.plot(x, y,
                    marker="s",
                    markersize=markersize,
                    markerfacecolor=_QNODE_FACE,
                    markeredgecolor=_QNODE_EDGE,
                    markeredgewidth=1.5,
                    zorder=3,
                    linestyle="none")
        else:
            ax.plot(x, y,
                    marker="o",
                    markersize=markersize,
                    markerfacecolor=_VNODE_FACE,
                    markeredgecolor=_VNODE_EDGE,
                    markeredgewidth=1.5,
                    zorder=3,
                    linestyle="none")

        ax.annotate(
            label,
            xy=(x, y),
            xycoords="data",
            xytext=(0, label_offset_pts),
            textcoords="offset points",
            ha="center", va="top",
            fontsize=fontsize,
            color=_TEXT_COLOR,
            linespacing=1.5,
            zorder=4,
            annotation_clip=False,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_tree(
    root,
    max_depth=None,
    min_visits=0,
    markersize=18,
    x_gap=1.2,
    y_gap=1.2,
    fontsize=7,
    figsize=None,
    ax=None,
    title=None,
):
    """Visualize a pomdp-py MCTS search tree with matplotlib.

    Nodes are drawn as fixed-size markers in display (point) coordinates so
    they always appear as true circles and squares regardless of how wide or
    tall the tree is. Edges are drawn in data coordinates as Bezier S-curves
    with thickness proportional to visit count.

    Parameters
    ----------
    root : VNode or RootVNode
        The root of the search tree (e.g. ``agent.tree`` or ``planner.tree``).
    max_depth : int, optional
        Maximum depth to render. Deeper subtrees are collapsed.
    min_visits : int, optional
        Hide nodes with fewer visits than this (default 0).
        Heuristic: ``root.num_visits // 100`` keeps the top 1% of paths.
    markersize : float, optional
        Node diameter in points (default 18). Nodes are always true circles.
    x_gap : float, optional
        Horizontal spacing between leaf nodes in inches (default 1.2).
    y_gap : float, optional
        Vertical spacing between tree levels in inches (default 1.2).
    fontsize : float, optional
        Font size for node labels (default 7).
    figsize : tuple or None, optional
        Figure size in inches. If None, auto-sized from tree extents.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If None, a new figure is created.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not root.children:
        raise ValueError("Tree root has no children — was the planner run?")

    positions, node_info = _assign_positions(root, max_depth, min_visits)
    parent_map = _build_parent_map(root, max_depth, min_visits)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    n_leaves = sum(1 for p in positions.values() if p[0] == p[0])  # all nodes
    x_span = max(xs) - min(xs)
    y_depth = max(ys) - min(ys)  # number of depth levels

    if ax is None:
        if figsize is None:
            # Width scales with number of leaf slots, height with depth.
            # x_gap and y_gap are in inches per unit.
            w = max((x_span + 2) * x_gap, 8)
            h = max((y_depth + 1) * y_gap + 1.0, 5)
            figsize = (min(w, 80), min(h, 120))
        fig, ax = plt.subplots(figsize=figsize, facecolor=_BG_COLOR)
    else:
        fig = ax.get_figure()
        fig.patch.set_facecolor(_BG_COLOR)

    ax.set_facecolor(_BG_COLOR)
    ax.axis("off")
    # No set_aspect — x and y are scaled independently so figsize controls spacing
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 0.5)

    _draw_edges(ax, positions, node_info, parent_map, root_visits=root.num_visits)
    _draw_nodes(ax, positions, node_info, markersize, fontsize)

    vnode_handle = Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=_VNODE_FACE, markeredgecolor=_VNODE_EDGE,
                           markersize=10, label="VNode (belief)", markeredgewidth=1.5)
    qnode_handle = Line2D([0], [0], marker="s", color="w",
                           markerfacecolor=_QNODE_FACE, markeredgecolor=_QNODE_EDGE,
                           markersize=10, label="QNode (action)", markeredgewidth=1.5)
    ax.legend(handles=[vnode_handle, qnode_handle], loc="upper right",
              fontsize=8, framealpha=0.9, frameon=True,
              facecolor="white", edgecolor="#cccccc")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color="#333333", pad=10)

    fig.tight_layout()
    return ax


def save_tree(root, path, dpi=150, **kwargs):
    """Render and save the tree to a file.

    SVG is recommended for large trees — fully zoomable in any browser or
    vector editor with no loss of detail.

    Parameters
    ----------
    root : VNode or RootVNode
        The root of the search tree.
    path : str
        Output path. Format inferred from extension (``.svg``, ``.png``, ``.pdf``).
    dpi : int, optional
        Resolution for raster formats (default 150). Ignored for SVG/PDF.
    **kwargs
        Forwarded to :func:`visualize_tree`.

    Examples
    --------
    >>> save_tree(agent.tree, "tree.svg", min_visits=50, title="BAMCP")
    >>> save_tree(agent.tree, "tree.png", max_depth=3, dpi=200)
    """
    ax = visualize_tree(root, **kwargs)
    ax.get_figure().savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(ax.get_figure())

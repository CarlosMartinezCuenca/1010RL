import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tenten.agents.registry import NAMES  # noqa: E402

# Paleta de referencia validada y tinta neutra para las baselines
COLORS = {
    "tabular": "#2a78d6",
    "dqn": "#eb6834",
    "ntuple": "#1baf7a",
    "ntuple_search": "#eda100",
}
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=28)
    ax.set_xlabel("Pasos de entrenamiento", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def spread(ys, gap):
    """Separa verticalmente etiquetas que quedarían una encima de otra."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


def plot_curves(histories, metric, title, ylabel, path, references, total_steps):
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    style(ax, title, ylabel)

    for value in references.values():
        ax.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    for model, history in histories.items():
        ax.plot(
            [h["step"] for h in history],
            [h[metric] for h in history],
            color=COLORS[model],
            linewidth=2,
            label=NAMES[model],
        )

    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.set_xlim(0, total_steps)
    ax.set_ylim(bottom=0)
    gap = 0.045 * ax.get_ylim()[1]

    # Baselines en una sola etiqueta sobre la más alta; modelos al final de su línea
    text = " · ".join(f"{label} {value:.0f}" for label, value in references.items())
    ax.annotate(
        text,
        (total_steps * 0.01, max(references.values()) + gap * 0.25),
        va="bottom",
        color=INK_2,
        fontsize=9,
    )
    ends = {m: h[-1][metric] for m, h in histories.items()}
    for (model, value), y in zip(ends.items(), spread(list(ends.values()), gap)):
        ax.annotate(
            NAMES[model],
            (total_steps, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
        )

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        frameon=False,
        labelcolor=INK,
        ncol=len(histories),
        borderaxespad=0.2,
    )
    fig.subplots_adjust(right=0.82, top=0.85)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_final(scores, games, path):
    order = sorted(scores, key=scores.get)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(order) + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(
        f"Puntuación media en {games} partidas nuevas (mejor modelo de cada uno)",
        loc="left",
        color=INK,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    colors = [COLORS.get(name, MUTED) for name in order]
    bars = ax.barh(
        [NAMES[n] for n in order], [scores[n] for n in order], height=0.4, color=colors
    )
    for bar, name in zip(bars, order):
        ax.annotate(
            f"{scores[name]:.0f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=10,
        )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.set_xlabel("Puntos", color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

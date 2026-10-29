"""Figures 4 and 6: the forward model as a block diagram, naive then with freedom.

Both share one layout so they read as a before/after pair: the full model only
adds the blocks outlined in the lighter blue. Fixed physics is a grey outline;
anything gradient descent updates is filled.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from figures import style
from figures.common import OUT

NAME = "architecture"
ROW, BELOW, ABOVE, TOP = 2.0, 0.1, 3.7, 4.75
OUT_X = 13.35


def compute() -> None:
    """Pure schematic: nothing to compute."""


def draw(full: bool) -> None:
    fig, ax = plt.subplots(figsize=(style.COLUMN_WIDTH_IN, 2.9))
    ax.set_xlim(0, 14.4)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(
        x: float, y: float, text: str, edge: str = style.RING, w: float = 1.9
    ) -> None:
        learned = edge != style.RING
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - 0.5),
                w,
                1.0,
                boxstyle="round,pad=0,rounding_size=0.18",
                facecolor=style.RAMP[6] if learned else style.PAPER,
                edgecolor=edge,
                lw=1.0 if learned else 0.8,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5)

    def arrow(start, end, color: str = style.RING, dashed: bool = False) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>,head_length=0.35,head_width=0.18",
                "color": color,
                "lw": 0.8,
                "ls": (0, (3, 2)) if dashed else "-",
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    # The gradient bus runs just above the highest learned block, so the naive
    # diagram crops to its content at the same scale as the full one.
    top = TOP if full else ROW + 1.25
    # Axes end just above the bus: their empty title slots would otherwise stretch
    # the tight crop. Fixed x range and width keep both diagrams at one scale.
    ax.set_ylim(-0.6, top + 0.55)

    def gradient(x: float, y: float) -> None:
        """Dashed drop from the gradient bus into a learned block."""
        arrow((x, top), (x, y + 0.5), color=style.SIGNAL, dashed=True)

    box(1.1, ROW, "sample\n$S(\\mathbf{k})$", style.SIGNAL)
    box(3.5, ROW, "illuminate\n$S(\\mathbf{k}-\\mathbf{k}_j)$")
    box(
        5.9,
        ROW,
        "lens\n$\\times\\,M(\\mathbf{k})$" if full else "ideal lens\n$\\times\\,$disc",
    )
    box(8.3, ROW, "sensor\n$|\\mathcal{F}^{-1}\\{\\cdot\\}|^2$")
    box(3.5, BELOW, "LED angles $\\mathbf{k}_j$", w=2.4)
    arrow((3.5, BELOW + 0.5), (3.5, ROW - 0.5))
    for a, b in ((2.05, 2.55), (4.45, 4.95), (6.85, 7.35)):
        arrow((a, ROW), (b, ROW))

    if full:
        box(5.9, ABOVE, "Zernike coeffs\n+ radius", style.ADDED, w=2.4)
        arrow((5.9, ABOVE - 0.5), (5.9, ROW + 0.5), color=style.ADDED)
        box(9.95, ROW, "$\\times\\,g_j$", style.ADDED, w=0.9)
        box(11.45, ROW, "$+\\,b_j$", style.ADDED, w=0.9)
        arrow((9.25, ROW), (9.5, ROW))
        arrow((10.4, ROW), (11.0, ROW))
        arrow((11.9, ROW), (OUT_X - 0.75, ROW))
        for x, label in ((9.95, "LED gain"), (11.45, "stray light")):
            ax.text(
                x,
                ROW - 0.62,
                label,
                ha="center",
                va="top",
                fontsize=7,
                color=style.INK_MUTED,
            )
        ax.text(
            7.3,
            ABOVE,
            "added freedom",
            ha="left",
            va="center",
            fontsize=8,
            color=style.ADDED,
        )
    else:
        arrow((9.25, ROW), (OUT_X - 0.75, ROW))
    box(OUT_X, ROW, "predicted\n$\\hat{y}_j$", w=1.5)

    box(OUT_X, BELOW, "loss", w=1.5)
    box(10.45, BELOW, "measured $y_j$", w=2.2)
    arrow((OUT_X, ROW - 0.5), (OUT_X, BELOW + 0.5))
    arrow((11.55, BELOW), (OUT_X - 0.75, BELOW))

    # Gradients flow back from the loss to every learned block.
    ax.plot(
        [OUT_X + 0.75, OUT_X + 1.0, OUT_X + 1.0, 1.1],
        [BELOW, BELOW, top, top],
        color=style.SIGNAL,
        ls=(0, (3, 2)),
        lw=0.8,
    )
    gradient(1.1, ROW)
    ax.text(
        3.5,
        top + 0.1,
        "gradient descent",
        ha="center",
        va="bottom",
        fontsize=8,
        color=style.SIGNAL,
    )
    if full:
        gradient(5.9, ABOVE)
        gradient(9.95, ROW)
        gradient(11.45, ROW)

    name = "full" if full else "naive"
    (OUT / NAME).mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / NAME / f"{name}.png")
    plt.close(fig)


def render() -> None:
    style.apply()
    draw(full=False)
    draw(full=True)

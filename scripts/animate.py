#!/usr/bin/env python3
"""
Save TWO GIFs animating tussock growth through time with true Radius (cm) in data coordinates.

CSV columns required:
TimeStep, Radius, X, Y, Z, Status
Status: 1=alive (green), 0=dead (brown)

This renders each tiller as a filled disk in the XY plane at height Z with radius = Radius (cm).

Outputs:
1) Original view (keeps matplotlib's default 3D view unless you change it)
2) "Down the Y axis" view (camera looks along Y so you mainly see X vs Z)
"""

import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def compute_limits(values, pad_frac=0.08):
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        pad = 1.0
    else:
        pad = (vmax - vmin) * pad_frac
    return vmin - pad, vmax + pad


def make_disk_polygon(x, y, z, r, n=40):
    """Return a single polygon (list of (x,y,z)) approximating a circle in XY at height z."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    xs = x + r * np.cos(theta)
    ys = y + r * np.sin(theta)
    zs = np.full_like(xs, z)
    return list(zip(xs, ys, zs))

def make_disk_polygon_xz(x, y, z, r, n=40):
    """Circle in XZ plane at fixed y (good for viewing along Y)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    xs = x + r * np.cos(theta)
    zs = z + r * np.sin(theta)
    ys = np.full_like(xs, y)
    return list(zip(xs, ys, zs))


def render_gif(df, timesteps, xlim, ylim, zlim, args, out_path, view_mode):
    """
    view_mode:
      - "default": keep the angle as-is (matplotlib's default for 3D axes)
      - "down_y": look straight down the Y axis (so you mostly see X and Z)
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # Make 1 cm look like 1 cm across axes
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))

    # Camera / view
    if view_mode == "down_y":
        # Look along the Y axis so the plot reads like X (horizontal) vs Z (vertical).
        # elev=0 gives a "straight on" feel; azim=90 points the camera down +Y toward the origin.
        ax.view_init(elev=0, azim=90)

        # If your matplotlib supports orthographic projection, use it to reduce perspective distortion.
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type("ortho")

    title = ax.set_title("")

    coll = Poly3DCollection([], edgecolor="none", alpha=args.alpha)
    ax.add_collection3d(coll)

    def frame_df(t):
        return df[df["TimeStep"] <= t] if args.show_trail else df[df["TimeStep"] == t]

    def update(frame_idx):
        t = timesteps[frame_idx]
        d = frame_df(t)

        polys = []
        facecolors = []

        for row in d.itertuples(index=False):
            x = float(getattr(row, "X"))
            y = float(getattr(row, "Y"))
            z = float(getattr(row, "Z"))
            r = float(getattr(row, "Radius"))
            status = int(getattr(row, "Status"))

            if r <= 0:
                continue

            if view_mode == "down_y":
                polys.append(make_disk_polygon_xz(x, y, z, r, n=args.circle_points))
            else:
                polys.append(make_disk_polygon(x, y, z, r, n=args.circle_points))
            facecolors.append("green" if status == 1 else "saddlebrown")

        coll.set_verts(polys)
        coll.set_facecolor(facecolors)

        n_alive = int((d["Status"].values == 1).sum())
        n_total = int(d.shape[0])
        mode = "trail (≤ year)" if args.show_trail else "snapshot (year)"
        view_label = "default view" if view_mode == "default" else "down Y view"
        title.set_text(f"Year: {t} | {mode} | {view_label} | alive: {n_alive}/{n_total}")

        return coll, title

    anim = FuncAnimation(fig, update, frames=len(timesteps), interval=args.interval, blit=False)
    fps = max(1, int(round(1000.0 / args.interval)))
    anim.save(out_path, writer="pillow", fps=fps, dpi=args.dpi)

    plt.close(fig)
    print(f"Saved GIF to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to CSV")
    ap.add_argument("--out", default="tussock.gif",
                    help="Output GIF filename for the default view (default: tussock.gif)")
    ap.add_argument("--out_down_y", default=None,
                    help="Output GIF filename for the 'down Y axis' view "
                         "(default: auto from --out)")
    ap.add_argument("--interval", type=int, default=100, help="Frame interval in ms (default: 100)")
    ap.add_argument("--dpi", type=int, default=150, help="DPI for saved GIF (default: 150)")
    ap.add_argument("--show_trail", action="store_true",
                    help="If set, include all tillers up to current year (≤ TimeStep). Otherwise snapshot only.")
    ap.add_argument("--circle_points", type=int, default=50,
                    help="Vertices per disk (default: 50). Higher = smoother, slower.")
    ap.add_argument("--alpha", type=float, default=0.85, help="Disk transparency (default: 0.85)")
    args = ap.parse_args()

    if args.out_down_y is None:
        if args.out.lower().endswith(".gif"):
            args.out_down_y = args.out[:-4] + "_down_y.gif"
        else:
            args.out_down_y = args.out + "_down_y.gif"

    df = pd.read_csv(args.csv_path)

    required = {"TimeStep", "Radius", "X", "Y", "Z", "Status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for c in ["TimeStep", "Radius", "X", "Y", "Z", "Status"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(required)).copy()
    df["TimeStep"] = df["TimeStep"].astype(int)
    df["Status"] = df["Status"].astype(int)

    timesteps = np.sort(df["TimeStep"].unique())
    if timesteps.size == 0:
        raise ValueError("No valid TimeStep rows found after cleaning.")
    
    xlim = (-30.0, 30.0)
    ylim = (-30.0, 30.0)
    zlim = (0.0, 50.0)

    # 1) Default view (keeps the original angle behavior)
    render_gif(df, timesteps, xlim, ylim, zlim, args, args.out, view_mode="default")

    # 2) Down the Y axis view
    render_gif(df, timesteps, xlim, ylim, zlim, args, args.out_down_y, view_mode="down_y")


if __name__ == "__main__":
    main()

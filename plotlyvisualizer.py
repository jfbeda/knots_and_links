import os
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from knots_and_links import *


class PlotlyVisualizer:

    def _validate_coords(self, coords):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be a 2D array with 3 columns (x,y,z).")
        if coords.size == 0:
            raise ValueError("coords is empty; nothing to plot.")
        if not np.isfinite(coords).all():
            raise ValueError("coords contain NaN/Inf.")

    def make_link_figure(
        self, link, *,
        closed=True,
        line_width=8,
        marker_size=3,
        colors=("#96C546", "#960E67"),
        width=900, height=900,
        orthographic=True,
        show_axis_triad=True     # NEW
    ):
        if not link.subknots:
            raise ValueError("Link has no subknots to display.")

        fig = go.Figure()
        all_coords = []

        items = list(link.subknots.items())


        for idx, (name, knot) in enumerate(items):
            coords = np.asarray(knot.coords, dtype=float)
            self._validate_coords(coords)

            if closed:
                coords = np.vstack([coords, coords[0]])

            all_coords.append(coords)

            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="lines+markers",
                    line=dict(width=line_width, color=color),
                    marker=dict(size=marker_size, color=color),
                    opacity=1.0,
                    showlegend=False,
                )
            )

        all_coords = np.vstack(all_coords)

        scene = dict(
            aspectmode="data",
            xaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
        )

        unnormalized_eye = np.array([0.4,0.4,1])
        length = np.sqrt(3)
        normalized_eye = length * unnormalized_eye/np.linalg.norm(unnormalized_eye)
        camera = dict(
            eye=dict(x= normalized_eye[0], y=normalized_eye[1], z=normalized_eye[2]),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0),
            projection=dict(type="orthographic" if orthographic else "perspective"),
        )

        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            scene=scene,
            scene_camera=camera,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        if show_axis_triad:
            self._add_axis_triad(fig, all_coords)


        return fig

    
    def _add_axis_triad(self, fig, all_coords, *, size_frac=0.4, offset_frac=0.1):
        """
        Add a small XYZ axis triad to the figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
        all_coords : (N,3) array of all points in the scene
        size_frac : length of axis as fraction of bounding box size
        offset_frac : offset from bounding box corner
        """
        mins = all_coords.min(axis=0)
        maxs = all_coords.max(axis=0)
        span = maxs - mins
        L = size_frac * span.max()

        # anchor point near bottom-back-left corner
        origin = np.array([
            mins[0] + offset_frac * span[0],  # left   (+x moves right)
            mins[1] + offset_frac * span[1],  # back   (+y moves forward)
            mins[2] + offset_frac * span[2],  # bottom (+z moves up)
        ])

        axes = [
            ("X", np.array([L, 0, 0]), "#000000"),
            ("Y", np.array([0, L, 0]), "#000000"),
            ("Z", np.array([0, 0, L]), "#000000"),
        ]

        for label, direction, color in axes:
            pts = np.vstack([origin, origin + direction])

            # axis line
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    line=dict(color=color, width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # label at axis tip
            # tip = origin + direction
            # fig.add_trace(
            #     go.Scatter3d(
            #         x=[tip[0]],
            #         y=[tip[1]],
            #         z=[tip[2]],
            #         mode="text",
            #         text=[label],
            #         textfont=dict(size=14, color=color),
            #         showlegend=False,
            #         hoverinfo="skip",
            #     )
            # )



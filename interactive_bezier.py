
import numpy as np
import matplotlib.pyplot as plt
from read_svg_cubics import read_svg_cubics


"""
Optimized cubic bezier curve function
Each row of C contains 4 indices into P, because C is cubic

Return: 2 element array with x,y coordinates
"""
def cubic_bezier(points, t) -> np.ndarray:
    p0, p1, p2, p3 = points
    return (
        (1-t)**3 * p0
        + 3*(1-t)**2*t * p1
        + 3*(1-t)*t**2 * p2
        + t**3 * p3
    )


class InteractiveBezier:
    def __init__(self, svg_path):
        # N = Control points
        # M = Curves
        # K = Bezier objects

        # P: (N,2) float control-point positions
        # C: (M,4) int indices into P (0-based)
        # I: (M,) int object indices (0-based)
        # F,S: (K,3) float RGB colors in 0..1, NaN means none/unknown
        # W: (K,) float stroke widths, NaN means unspecified
        # D: (K,) int visibility flags
        # viewBox: (4,) float SVG viewBox, or an empty array
        self.P, self.C, self.I, self.F, self.S, self.W, self.D, self.viewBox = read_svg_cubics(svg_path)

        self.fig, self.ax = plt.subplots() #Figure (window), and Axis (plotting area)

        self.selected_array = None
        self.nearby_array = []

        self.prev_event_x = 0
        self.prev_event_y = 0

    """
    Draw plots and gui
    """
    def draw(self):
        self.ax.clear()

        t_values = np.linspace(0, 1, 100)

        for indices in self.C:
            points = self.P[indices]
            curve = np.array([cubic_bezier(points, t) for t in t_values])

            self.ax.plot(curve[:, 0], curve[:, 1], "b-")                 # Draw the curve, connecting linearly between t values
            self.ax.plot(points[:, 0], points[:, 1], "k--", alpha=0.3)   # Draw the control polygon's handles

        self.ax.scatter(self.P[:, 0], self.P[:, 1], color="red", picker=5)         # Draw the control point dots
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()
        self.ax.set_title("Drag red control points")

        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        self.prev_event_x = event.xdata
        self.prev_event_y = event.ydata

        self.selected_array = None
        self.nearby_array = []

        if len(self.P) == 0:
            return

        # Find nearest control point in screen pixels
        xy = self.ax.transData.transform(self.P)                   # Fetches all control point coordinates
        distances = np.linalg.norm(                                # Subtracts mouse click pos from control point coordinates
            xy - np.array([event.x, event.y]), axis=1
        )

        closest = np.argmin(distances)                   # Gets the smallest distance

        if distances[closest] < 12:
            # Select all points at the same coordinate location
            self.selected_array = np.where(
                np.linalg.norm(self.P - self.P[closest], axis=1) < 1e-5
            )[0]

            # Get the nearby points of the selected point
            if len(self.selected_array) >= 2:
                for selected in self.selected_array:

                    # Find the curve containing the selected point
                    curve_idx = np.where(self.C == selected)[0][0]
                    curve = self.C[curve_idx]

                    # Find the selected point's position within the curve
                    position = np.where(curve == selected)[0][0]

                    # Find the adjacent control handle depending on which end the selected point is at
                    if position == 0:
                        self.nearby_array.append(curve[1])
                    elif position == 3:
                        self.nearby_array.append(curve[2])

    def on_move(self, event):
        if self.selected_array is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Calculate how far the mouse moved
        event_x_delta = event.xdata - self.prev_event_x
        event_y_delta = event.ydata - self.prev_event_y

        # Move selected anchor points
        for selected in self.selected_array:
            self.P[selected] = [event.xdata, event.ydata]

        # Move adjacent handles by the same amount
        for nearby in self.nearby_array:
            self.P[nearby] += [event_x_delta, event_y_delta]

        # Save current position for next mouse movement
        self.prev_event_x = event.xdata
        self.prev_event_y = event.ydata

        self.draw()

    def on_release(self, event):
        self.selected_array = None
        self.nearby_array = []

    def run(self):
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        self.draw()
        plt.show()

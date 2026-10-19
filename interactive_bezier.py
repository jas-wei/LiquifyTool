import numpy as np
import matplotlib.pyplot as plt
from read_svg_cubics import read_svg_cubics

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
P, C, I, F, S, W, D, viewBox = read_svg_cubics("example.svg")

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


fig, ax = plt.subplots() #Figure (window), and Axis (plotting area)
selected_array = None
nearby_array = []

prev_event_x = 0
prev_event_y = 0

"""
Draw plots and gui
"""
def draw():
    ax.clear()

    t_values = np.linspace(0, 1, 100)

    for indices in C:
        points = P[indices]
        curve = np.array([cubic_bezier(points, t) for t in t_values])

        ax.plot(curve[:, 0], curve[:, 1], "b-")                 # Draw the curve, connecting linearly between t values
        ax.plot(points[:, 0], points[:, 1], "k--", alpha=0.3)   # Draw the control polygon's handles

    ax.scatter(P[:, 0], P[:, 1], color="red", picker=5)         # Draw the control point dots
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Drag red control points")

    fig.canvas.draw_idle()

def on_press(event):
    global selected_array, nearby_array
    global prev_event_x, prev_event_y

    if event.inaxes != ax or event.xdata is None:
        return

    prev_event_x = event.xdata
    prev_event_y = event.ydata

    selected_array = None
    nearby_array = []

    # Find nearest control point in screen pixels
    xy = ax.transData.transform(P)                   # Fetches all control point coordinates
    distances = np.linalg.norm(                      # Subtracts mouse click pos from control point coordinates
        xy - np.array([event.x, event.y]), axis=1    
    )

    if len(P) == 0:
        return
    
    closest = np.argmin(distances)                   # Gets the smallest distance

    if distances[closest] < 12:
        # Select all points at the same coordinate location
        selected_array = np.where(
            np.linalg.norm(P - P[closest], axis=1) < 1e-5
        )[0]

        # Get the nearby points of the selected point
        if len(selected_array) >= 2:
            for selected in selected_array:

                # Find the curve containing the selected point
                curve_idx = np.where(C == selected)[0][0]
                curve = C[curve_idx]

                # Find the selected point's position within the curve
                position = np.where(curve == selected)[0][0]

                # Find the adjacent control handle depending on which end the selected point is at
                if position == 0:
                    nearby_array.append(curve[1])
                elif position == 3:
                    nearby_array.append(curve[2])


def on_move(event):
    global prev_event_x, prev_event_y

    if selected_array is None or event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return

    # Calculate how far the mouse moved
    event_x_delta = event.xdata - prev_event_x
    event_y_delta = event.ydata - prev_event_y

    # Move selected anchor points
    for selected in selected_array:
        P[selected] = [event.xdata, event.ydata]

    # Move adjacent handles by the same amount
    for nearby in nearby_array:
        P[nearby] += [event_x_delta, event_y_delta]

    # Save current position for next mouse movement
    prev_event_x = event.xdata
    prev_event_y = event.ydata

    draw()

def on_release(event):
    global selected_array, nearby_array

    selected_array = None
    nearby_array = []

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_release_event", on_release)

draw()
plt.show()

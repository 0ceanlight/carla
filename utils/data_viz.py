
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcursors

LINE_WIDTH = 2.5
mpl.rcParams['font.size'] *= 2.0

def get_pose_plot(pose_sets, colors, labels, min_x=None, max_x=None, min_y=None, max_y=None, title=None, ylabel=None):
    """
    Plot multiple sequences of float values as 2D scatter plots.

    Each set of values in `pose_sets` is plotted such that each element is a point
    (x, y) = (i, value), where i is the index of the value in the list.

    Parameters:
    - pose_sets (list of list of float): A list where each inner list is a sequence
      of float values to be plotted.
    - colors (list of str): A list of color names or codes for each set.
    - labels (list of str): A list of labels for each dataset to be used in the legend.

    Returns:
    - plt: The matplotlib plot object.

    Features:
    - Each dataset is displayed in its respective color.
    - A legend (key) is shown using the provided labels.
    - Hovering over any point displays its (x, y) coordinates, with annotation colored
      to match the data point.
    """

    assert len(pose_sets) == len(colors) == len(labels), \
        "Input arrays must be of the same length."

    plt.figure(figsize=(10, 6))

    for pose_set, color, label in zip(pose_sets, colors, labels):
        x = list(range(len(pose_set)))
        y = pose_set
        # scatter = plt.scatter(x, y, color=color, label=label)
        # plot = plt.plot(x, y, color=color, linewidth=LINE_WIDTH, alpha=0.7)  # adjust line
        plot = plt.plot(x, y, color=color, linewidth=LINE_WIDTH, alpha=1.0)  # adjust line

        # cursor = mplcursors.cursor(scatter, hover=True)
        cursor = mplcursors.cursor(plot, hover=True)
        cursor.connect("add", lambda sel, c=color: (
            sel.annotation.set_text(f"x={int(sel.target[0])}, y={sel.target[1]:.3f}"),
            sel.annotation.get_bbox_patch().set(fc=c, alpha=0.8)
        ))

    if min_x is not None:
        plt.xlim(left=min_x)
    if max_x is not None:
        plt.xlim(right=max_x)
    if min_y is not None:
        plt.ylim(bottom=min_y)
    if max_y is not None:
        plt.ylim(top=max_y)

    if title is not None:
        plt.title(title)
    else:
        plt.title("Pose Absolute Error and Registration Fitness, RMSE")
    plt.xlabel("Frame Index")
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("Distance from ground truth (m) / Fitness / RMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


def get_split_pose_plot(
    top_pose_sets, top_colors, top_labels,
    bottom_pose_sets, bottom_colors, bottom_labels,
    min_x=None, max_x=None,
    top_min_y=None, top_max_y=None,
    bottom_min_y=None, bottom_max_y=None,
    title=None
):
    """
    Plot two vertically stacked 2D graphs:
    1. Top graph: positional errors
    2. Bottom graph: registration metrics (fitness, RMSE)

    Hovering over any point displays its (x, y) coordinates.

    Parameters:
    - top_pose_sets, bottom_pose_sets: list of float lists to be plotted
    - top_colors, bottom_colors: corresponding color for each line
    - top_labels, bottom_labels: legend labels
    - min_x, max_x, top_min_y, top_max_y, bottom_min_y, bottom_max_y: axis limits
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot top graph (Positional errors)
    for data, color, label in zip(top_pose_sets, top_colors, top_labels):
        x = list(range(len(data)))
        # line, = ax1.plot(x, data, color=color, label=label, linewidth=LINE_WIDTH, alpha=0.7)
        line, = ax1.plot(x, data, color=color, label=label, linewidth=LINE_WIDTH, alpha=1.0)
        cursor = mplcursors.cursor(line, hover=True)
        cursor.connect("add", lambda sel, c=color: (
            sel.annotation.set_text(f"x={int(sel.target[0])}, y={sel.target[1]:.3f}"),
            sel.annotation.get_bbox_patch().set(fc=c, alpha=0.8)
        ))

    ax1.set_ylabel("Positional Error (m)")
    ax1.set_ylim(top=top_max_y if top_max_y is not None else None,
                 bottom=top_min_y if top_min_y is not None else None)
    ax1.grid(True)
    ax1.legend()
    if title is not None:
        ax1.set_title(title)
    else:
        ax1.set_title("Pose Absolute Error and Registration Fitness, RMSE")

    # Plot bottom graph (Fitness and RMSE)
    for data, color, label in zip(bottom_pose_sets, bottom_colors, bottom_labels):
        x = list(range(len(data)))
        # line, = ax2.plot(x, data, color=color, label=label, linewidth=LINE_WIDTH, alpha=0.7)
        line, = ax2.plot(x, data, color=color, label=label, linewidth=LINE_WIDTH, alpha=1.0)
        cursor = mplcursors.cursor(line, hover=True)
        cursor.connect("add", lambda sel, c=color: (
            sel.annotation.set_text(f"x={int(sel.target[0])}, y={sel.target[1]:.3f}"),
            sel.annotation.get_bbox_patch().set(fc=c, alpha=0.8)
        ))

    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Fitness / RMSE")
    ax2.set_ylim(top=bottom_max_y if bottom_max_y is not None else None,
                 bottom=bottom_min_y if bottom_min_y is not None else None)
    ax2.grid(True)
    ax2.legend()

    if min_x is not None or max_x is not None:
        ax1.set_xlim(left=min_x if min_x is not None else None,
                     right=max_x if max_x is not None else None)

    plt.tight_layout()
    return plt

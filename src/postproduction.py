from matplotlib.patches import Rectangle


def highlight_x_interval(ax, x_begin, x_end, y_begin, y_end, color):
    ax.add_patch(Rectangle((x_begin, y_begin), x_end, y_end-y_begin, color=color))



import matplotlib.pyplot as plt


def show_line_plot(x_series, title, x_label, y_label, y_series=False, plot_arr=False):
    """
    Creates a simple line graph

    Args:
        x_series: Array-like data structure
        title: <str> graph title
        x_label: <str> graph x-axis label
        y_label: <str> graph y-axis label
        y_series: Array-like data structure
        plot_arr: <bool> controls whether the y_series parameter will be used

    Returns: <Matplotlib figure> the plotted figure

    """
    fig = plt.subplots()

    if plot_arr:
        plt.plot(x_series)
    else:
        if not y_series:
            raise Exception('If plot_arr is False, y_series must be provided')
        plt.plot(x_series, y_series)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()

    return fig

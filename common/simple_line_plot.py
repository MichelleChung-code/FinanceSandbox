import matplotlib.pyplot as plt


def show_line_plot(x_series, title, x_label, y_label, y_series=False, plot_arr=False):
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

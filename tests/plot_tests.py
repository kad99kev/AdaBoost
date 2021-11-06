import plotly.figure_factory as ff


def plot_history(data, name):
    """
    Plots the histories of the test runs.

    Arguments:
        data: History data.
        column_names: Names of the columns.
    """
    fig = ff.create_table(data)
    fig.write_image(f"images/tests/{name}.png")

"""
Written by:
Name: Kevlyn Kadamala
Student ID: 21236191
Class: MSc AI
"""

import re
import plotly.graph_objects as go
from igraph import Graph


def _remove_duplicates(sequence):
    """
    Removes duplicates while maintaining order.

    Arguments:
        sequence: List from which duplicates need to be removed.
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def _get_colours(vertices, classes):
    """
    Gives a list of the class colours for each vertex.
    """
    colours = []
    for v in vertices:
        class_ = re.findall("Class: -?[0-9]", v)
        colour = "#6175c1" if int(class_[0].split(" ")[1]) == classes[0] else "#c16161"
        colours.append(colour)
    return colours


def plot_tree(nodes, root, classes):
    """
    Plot a decision tree.
    Reference: https://plotly.com/python/tree-plots/

    Arguments:
        nodes: Nodes in the tree, list of tuples in the form of (parent, child).
        root: The root node of the tree.
        classes: Classes in the Decision Tree.
    """

    # Get individual vertices from a list of nodes.
    # Need to maintain order.
    vertices = []
    for node in nodes:
        vertices.extend(node)
    vertices = _remove_duplicates(vertices)
    colours = _get_colours(vertices, classes)

    v_label = list(map(str, range(len(vertices))))

    labels = list(map(str, vertices))

    # Create graph object.
    G = Graph()
    # Add vertices to the graph.
    G.add_vertices(vertices)
    # Add edges to the graph.
    G.add_edges(nodes)

    # Get layout of the tree using the Reingold Tilford layout.
    # Reference: https://igraph.org/python/tutorial/0.9.8/tutorial.html#layout-algorithms
    lay = G.layout("tree", root=vertices.index(root))
    position = {k: lay[k] for k in range(len(vertices))}
    Y = [lay[k][1] for k in range(len(vertices))]
    M = max(Y)
    E = [e.tuple for e in G.es]  # list of edges
    L = len(position)

    # Node positions
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]

    # Edge positions
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Xe,
            y=Ye,
            mode="lines",
            line=dict(color="rgb(210,210,210)", width=1),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=Xn,
            y=Yn,
            mode="markers",
            name="bla",
            marker=dict(
                symbol="square",
                size=100,
                color=colours,
                line=dict(color="rgb(50,50,50)", width=1),
            ),
            text=labels,
            hoverinfo="text",
            opacity=0.8,
        )
    )

    def make_annotations(pos, text, font_size=10, font_color="rgb(250,250,250)"):
        """
        Annotates the node.

        Arguments:
            pos: Co-ordinates for the markers.
            text: Text for each node.
            font_size: Font size for the text.
            font_color: Font colour of the text.
        """
        L = len(pos)
        if len(text) != L:
            raise ValueError("The lists pos and text must have the same len")
        annotations = []
        for k in range(L):
            text = labels[k].replace("\n", "<br>")
            annotations.append(
                dict(
                    text=text,  # or replace labels with a different list for the text within the circle
                    x=pos[k][0],
                    y=2 * M - position[k][1],
                    xref="x1",
                    yref="y1",
                    font=dict(color=font_color, size=font_size),
                    showarrow=False,
                )
            )
        return annotations

    axis = dict(
        showline=False,  # Hide axis line, grid, ticklabels and title.
        zeroline=False,
        showgrid=False,
        showticklabels=False,
    )

    fig.update_layout(
        title="Decision Tree",
        annotations=make_annotations(position, v_label),
        font_size=12,
        showlegend=False,
        xaxis=axis,
        yaxis=axis,
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode="closest",
        plot_bgcolor="rgb(248,248,248)",
    )
    fig.show()
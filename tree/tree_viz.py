import igraph
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go

# Ref: https://plotly.com/python/tree-plots/


def remove_duplicates(sequence):
    # Removes duplicates while maintaining order
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def plot_tree(nodes, root):
    vertices = []
    for node in nodes:
        vertices.extend(node)
    vertices = remove_duplicates(vertices)

    v_label = list(map(str, range(len(vertices))))

    labels = list(map(str, vertices))

    G = Graph()
    G.add_vertices(vertices)
    # Add edges to the graph
    G.add_edges(nodes)

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
                color="#6175c1",  #'#DB4551',
                line=dict(color="rgb(50,50,50)", width=1),
            ),
            text=labels,
            hoverinfo="text",
            opacity=0.8,
        )
    )

    def make_annotations(pos, text, font_size=10, font_color="rgb(250,250,250)"):
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
        showline=False,  # hide axis line, grid, ticklabels and  title
        zeroline=False,
        showgrid=False,
        showticklabels=False,
    )

    fig.update_layout(
        title="Tree with Reingold-Tilford Layout",
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
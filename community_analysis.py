from IPython.display import HTML, clear_output
import sys
sys.path.append("/home/ec2-user/SageMaker/")
import boto3
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, timedelta, date

import networkx as nx
import plotly.graph_objects as go

wh = Waveform_Helper()
athena = Athena_Query()

## Function for Degree and Betweenness Centrality:
def calculate_centrality_measures(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    return degree_centrality, betweenness_centrality

## Function for Plotting Correlation Network using Plotly:
def plot_correlation_network_plotly(G, correlation_matrix, feature_sets, set_colors, threshold, save_path):
    fig = go.Figure()

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(color='black'), hoverinfo='none'))

    for node in G.nodes():
        x, y = pos[node]
        color = set_colors[feature_sets.loc[node]['Set']]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(color=color, size=10), text=node, hoverinfo='text'))

    fig.update_layout(
        title="Correlation Network",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Save the Plotly figure as an HTML file
    fig.write_html(save_path)

## Function for Plotting Correlation Network using NetworkX and Matplotlib:
def plot_correlation_network_matplotlib(G, correlation_matrix, feature_sets, set_colors, threshold):
    pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 10))
    node_colors = [set_colors[feature_sets.loc[node]['Set']] for node in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Correlation Network")
    plt.show()

## Function for Plotting Community Network using PyVis:
from pyvis.network import Network

def plot_community_network_pyvis(G, save_path):
    net = Network(notebook=True, width="1000px", height="700px", bgcolor='#222222', font_color='white')
    node_degree = dict(G.degree)
    
    nx.set_node_attributes(G, node_degree, 'size')
    net.from_nx(G)
    net.show(save_path)

## Function for Plotting Centrality Measures:
def plot_centrality_measures(G):
    degree_dict = nx.degree_centrality(G)
    betweenness_dict = nx.betweenness_centrality(G)
    closeness_dict = nx.closeness_centrality(G)

    degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['centrality'])
    betweenness_df = pd.DataFrame.from_dict(betweenness_dict, orient='index', columns=['centrality'])
    closeness_df = pd.DataFrame.from_dict(closeness_dict, orient='index', columns=['centrality'])

    # Plot top 10 nodes for each centrality measure
    degree_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar", title="Degree Centrality")
    betweenness_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar", title="Betweenness Centrality")
    closeness_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar", title="Closeness Centrality")
    plt.show()

## Function for Plotting Communities using PyVis:
def plot_communities_pyvis(G, save_path):
    communities = community.best_partition(G)
    nx.set_node_attributes(G, communities, 'group')
    com_net = Network(notebook=True, width="1000px", height="1000px

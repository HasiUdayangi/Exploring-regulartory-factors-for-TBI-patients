from IPython.display import HTML, clear_output
import boto3
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, timedelta, date
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.data import Data, DataLoader, Dataset
wh = Waveform_Helper()

athena = Athena_Query()


## Function for Plotting Correlation Heatmap:
def plot_correlation_heatmap(dataframe, metrics, group_boundaries, save_path):
    correlation_matrix = dataframe[metrics].corr()

    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)

    for boundary in group_boundaries:
        plt.axhline(y=boundary, color='r', linewidth=2)
        plt.axvline(x=boundary, color='r', linewidth=2)

    plt.title("Feature Correlation Heatmap")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

## Function for Visualizing Correlation Network:
def visualize_correlation_network(correlation_matrix, feature_sets, threshold, set_colors, save_path):
    G = nx.Graph()

    for feature in correlation_matrix.columns:
        G.add_node(feature)

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j])

    node_colors = [set_colors[feature_sets.loc[node]['Set']] for node in G.nodes]

    plt.figure(figsize=(20, 20))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Correlation Network")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

## Function for Creating and Analyzing Subgraphs:
def create_and_analyze_subgraph(G, feature_sets, set1, set2):
    edges = [(u, v) for u, v in G.edges() if feature_sets.loc[u, 'Set'] == set1 and feature_sets.loc[v, 'Set'] == set2]
    subgraph = G.edge_subgraph(edges)
    print(f"Number of edges in {set1}-{set2} subgraph:", subgraph.number_of_edges())
    return subgraph

## Function for Calculating Density:
def calculate_density(num_edges, num_nodes_group1, num_nodes_group2):
    total_possible_edges = num_nodes_group1 * num_nodes_group2
    density = num_edges / total_possible_edges
    return density

## Function for Plotting Group Connection Graph:
def plot_group_connection_graph(normalized_densities, set_colors, save_path):
    G = nx.Graph()

    groups = ['HRV', 'Vital', 'Lab', 'GCS', 'Age']
    for group in groups:
        G.add_node(group)

    edges = [
        ('HRV', 'Vital', normalized_densities[0]),
        ('HRV', 'Lab', normalized_densities[1]),
        ('HRV', 'GCS', normalized_densities[2]),
        ('HRV', 'Age', normalized_densities[3]),
        ('Vital', 'Lab', normalized_densities[4]),
        ('Vital', 'GCS', normalized_densities[5]),
        ('Vital', 'Age', normalized_densities[6]),
        ('Lab', 'GCS', normalized_densities[7]),
        ('Lab', 'Age', normalized_densities[8]),
        ('GCS', 'Age', normalized_densities[9]),
        # Add other edges similarly
    ]

    G.add_weighted_edges_from(edges)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 10))

    nx.draw_networkx_nodes(G, pos, node_size=700)
    for (u, v, wt) in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=wt['weight'])

    nx.draw_networkx_labels(G, pos, font_size=20)
    plt.title("Connections Between Biomarker Groups")
    plt.axis('off')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


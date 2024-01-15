from IPython.display import HTML, clear_output
import sys
import boto3
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, timedelta, date
import seaborn as sns
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.data import Data, DataLoader, Dataset

from node2vec import Node2Vec
from sklearn.decomposition import PCA


wh = Waveform_Helper()
athena = Athena_Query()

##1. Visualization and Analysis Functions

def visualize_correlation_heatmap(merged_df, metrics):
    correlation_matrix = merged_df[metrics].corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    group_boundaries = [0, 16, 31, 45, 49, 51]
    for boundary in group_boundaries:
        plt.axhline(y=boundary, color='r', linewidth=2)
        plt.axvline(x=boundary, color='r', linewidth=2)
    plt.savefig(f'heatmap.png', dpi=100, bbox_inches='tight')
    plt.title("Feature Correlation Heatmap")
    plt.show()

def visualize_graph(G, metrics, feature_sets, threshold=0.5):
    # Initialize a graph
    G = nx.Graph()

    # Add nodes (features) to the graph and assign colors
    for feature in metrics:
        subset = feature_sets.loc[feature.strip()]['Set']
        G.add_node(feature, color=subset)

    correlation_matrix = merged_df[metrics].corr()

    # Add edges (significant correlations) with a weight of 1
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=1)

    # Create a mapping from node names to integers
    node_to_index = {node: i for i, node in enumerate(G.nodes)}

    # Map the nodes to integers in the edge list
    edge_index = torch.tensor([(node_to_index[u], node_to_index[v]) for u, v in G.edges], dtype=torch.long).t().contiguous()

    # Convert the adjacency matrix to a dense PyTorch tensor
    adjacency_matrix = nx.adjacency_matrix(G)
    adjacency_matrix = torch.tensor(adjacency_matrix.todense(), dtype=torch.float)

    # Print adjacency matrix and edge indices
    print("Edge Index Shape:", edge_index.shape)
    print("Edge Index Content:", edge_index)
    print("Adjacency Matrix:")
    print(adjacency_matrix)
    print("Edge Indices:")
    print(edge_index)

    # Compute the layout positions of the nodes using a force-directed layout algorithm
    pos = nx.spring_layout(G, seed=42)

    # Define node colors based on subsets
    node_colors = [G.nodes[node]['color'] for node in G.nodes]

    plt.figure(figsize=(20, 20))
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, cmap=plt.cm.Blues)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Draw node labels if needed
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', font_weight='bold')
    plt.savefig("gnn_graph.png")
    plt.axis('off')
    plt.show()

def create_custom_dataset(G):
    # Add nodes (features) to the graph and assign colors
    node_features = {feature: G.nodes[feature] for feature in G.nodes}
    # Add edges with weights
    edge_index = torch.tensor([(u, v) for u, v, d in G.edges(data=True)], dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor([d['weight'] for u, v, d in G.edges(data=True)], dtype=torch.float).view(1, -1)

    # Convert the adjacency matrix to a dense PyTorch tensor
    adjacency_matrix = nx.adjacency_matrix(G)
    adjacency_matrix = torch.tensor(adjacency_matrix.todense(), dtype=torch.float)

    # Create a Data object for PyTorch Geometric
    data = Data(x=adjacency_matrix, edge_index=edge_index, edge_attr=edge_weights)

    return data


def train_graph_autoencoder(model, train_loader, num_epochs=1000, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            data = batch.to(device)  # Send the data to the appropriate device (CPU or GPU)
            output = model(data.x, data.edge_index)
            loss = F.mse_loss(output, data.x)  # Mean Squared Error as the reconstruction loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch + 1}, Loss: {average_loss}')



def node2vec_embedding(G, dimensions=64, walk_length=30, num_walks=200, workers=4):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_embeddings = model.wv
    return node_embeddings

def visualize_node_embeddings(node_embeddings_2d, node_names):
    # Reduce dimensionality with PCA
    pca = PCA(n_components=2)
    node_embeddings_2d = pca.fit_transform(node_embeddings_2d.vectors)

    # Create a scatter plot for visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c='b', marker='o')

    # Annotate nodes with their labels or IDs
    for i, txt in enumerate(node_names):
        plt.annotate(txt, (node_embeddings_2d[i, 0], node_embeddings_2d[i, 1]))

    plt.title('Node Embedding Visualization (2D)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

##  2. Graph Neural Network (GNN) Model and Training Functions


class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_gcn_model(num_features, hidden_dim, num_classes):
    model = GCN(num_features, hidden_dim, num_classes)
    return model

def train_gcn_model(model, train_loader, num_epochs=1000, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            data = batch.to(device)  # Send the data to the appropriate device (CPU or GPU)
            output = model(data.x, data.edge_index)
            loss = torch.nn.functional.mse_loss(output, data.x)  # Mean Squared Error as the reconstruction loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch + 1}, Loss: {average_loss}')

def top_nodes_based_on_degree(G, k=20):
    # Step 1: Compute node degrees
    node_degrees = dict(G.degree())

    # Step 2: Sort nodes based on degrees in descending order
    sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)

    # Step 3: Get top k nodes
    top_k_nodes = sorted_nodes[:k]

    # Print or use the top k nodes as needed
    print(f"Top {k} Nodes based on Degree:", top_k_nodes)
    return top_k_nodes

## 3. Hyperparameter Tuning and Training GCN with Optimal Hyperparameters


def hyperparameter_tuning(hidden_dims, learning_rates, num_epochs, num_classes, data, edge_index, device):
    best_loss = float('inf')
    best_params = None

    for hidden_dim in hidden_dims:
        for learning_rate in learning_rates:
            model = GCN(data.x.size(1), hidden_dim, num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()  # Mean Squared Error Loss

            for epoch in range(num_epochs):
                train_loss = train(model, data, optimizer, criterion, device)
                val_loss = evaluate(model, data, criterion, device)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = {'hidden_dim': hidden_dim, 'learning_rate': learning_rate}

    print("Best Hyperparameters:", best_params)
    return best_params

def train_gcn_with_optimal_hyperparameters(num_features, best_hidden_dim, best_learning_rate, data, edge_index, num_epochs=10000, device="cuda"):
    model = GCN(num_features, best_hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data.x, edge_index)
        loss = torch.nn.functional.mse_loss(output, data.x)  # Mean Squared Error as the reconstruction loss
        loss.backward()
        optimizer.step()

## 4. Feature Importance Analysis and Visualization

def feature_importance_analysis(node_embeddings, G, top_k=20):
    # Calculate L2 norms for node embeddings
    node_norms = torch.norm(node_embeddings, dim=1)

    # Select top k features based on L2 norms
    _, indices = torch.topk(node_norms, k=top_k, largest=True)
    top_k_nodes = [i.item() for i in indices]

    # Map the indices back to feature names based on your graph's node labels
    feature_names = [list(G.nodes())[i] for i in top_k_nodes]

    # Get top k scores for the selected features (nodes)
    top_k_scores = [node_norms[i].item() for i in top_k_nodes]

    return feature_names, top_k_scores

def visualize_feature_importance(feature_names, top_scores, title, save_path):
    # Plot the bar chart for the top k features and their scores
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(len(feature_names)), top_scores, align='center')
    plt.yticks(np.arange(len(feature_names)), feature_names)
    plt.xlabel('Importance Score')
    plt.ylabel('Node Attributes')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert the y-axis to show the most important feature at the top
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

## 5. Visualization of Graphs

def visualize_graph_with_scores(G, indices, node_norms, save_path):
    # Create a new graph containing only the top nodes and their edges
    top_nodes = [list(G.nodes())[i] for i in indices]
    G_top = G.subgraph(top_nodes)

    # Define a color map based on importance scores using the 'coolwarm' colormap
    color_map = [plt.cm.coolwarm(node_norm / max(node_norms)) for node_norm in node_norms]

    # Draw the entire graph with all edges
    plt.figure(figsize=(25, 25))
    pos = nx.spring_layout(G, seed=42)

    # Draw all nodes with labels and colors based on real importance scores
    all_nodes = nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=100)
    all_edges = nx.draw_networkx_edges(G, pos, edge_color='lightgrey', alpha=0.3)

    # Draw top nodes with labels and colors based on real importance scores
    top_nodes = nx.draw_networkx_nodes(G_top, pos, node_color='red', node_size=500)
    top_edges = nx.draw_networkx_edges(G_top, pos, edge_color='grey')

    # Annotate top nodes with their labels and scores
    node_labels = {node: f"{node}\n{score:.2f}" for node, score in zip(top_nodes, node_norms)}
    labels = nx.draw_networkx_labels(G_top, pos, labels=node_labels, font_size=10)

    # Remove node labels and axis
    plt.axis('off')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

def visualize_radar_chart(normalized_scores, feature_names, save_path):
    # Create an array of angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()

    # Duplicate the first angle and append it to match the extra normalized score
    angles += angles[:1]

    # Create a figure and axis
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Fill the radar plot
    ax.fill(angles, normalized_scores + [normalized_scores[0]], 'b', alpha=0.1)  # Append the first score to match the length

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)

    # Set the y-labels and limit the y-axis to 0-1
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)
    plt.title('Normalized Important Scores for All Features')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

def visualize_top_20_radar_chart(top_20_normalized_scores, top_20_feature_names, save_path):
    # Create an array of angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, len(top_20_feature_names), endpoint=False).tolist()
    angles += angles[:1]  # Duplicate the first angle to close the radar chart

    # Create a figure and axis
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Fill the radar plot for top 20 features
    ax.fill(angles, top_20_normalized_scores + [top_20_normalized_scores[0]], 'b', alpha=0.1)

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_20_feature_names)

    # Set the y-labels and limit the y-axis to 0-1
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)
    plt.title('Normalized Important Scores for Top 20 Features')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()





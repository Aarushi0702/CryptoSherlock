import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Dict
from data_io import Transaction

class GraphBuilder:
    """Builds directed graphs from cryptocurrency transaction data"""
    def __init__(self):
        # Initialize directed graph and position storage for visualization
        self.graph = nx.DiGraph()
        self.node_positions = {}

    def build_address_graph(self, transactions: List[Transaction]) -> nx.DiGraph:
        """
        Build directed graph where nodes are cryptocurrency addresses and edges represent payments
        
        Args:
            transactions: List of Transaction objects containing payment data
            
        Returns:
            NetworkX directed graph with address nodes and payment edges
        """
        self.graph.clear()
        
        # Process each transaction to add nodes and edges
        for tx in transactions:
            # Add sender address as node if it exists
            if getattr(tx, "from_addr", None):
                self.graph.add_node(
                    tx.from_addr,
                    node_type="address",
                    classification=getattr(tx, "classification", "unknown"),
                )
            # Add receiver address as node if it exists
            if getattr(tx, "to_addr", None):
                self.graph.add_node(
                    tx.to_addr,
                    node_type="address",
                    classification=getattr(tx, "classification", "unknown"),
                )
            # Add directed edge representing the payment
            if getattr(tx, "from_addr", None) and getattr(tx, "to_addr", None):
                self.graph.add_edge(
                    tx.from_addr,
                    tx.to_addr,
                    amount=tx.value,               # Transaction amount
                    timestamp=tx.timestamp,        # When transaction occurred
                    tx_hash=tx.hash,              # Transaction hash for reference
                    edge_type="payment",          # Type of relationship
                )
        
        return self.graph

    def compute_layout(self, graph: nx.DiGraph, seed: int = 42) -> Dict:
        """
        Compute node positions using spring layout algorithm for graph visualization
        
        Args:
            graph: NetworkX directed graph
            seed: Random seed for deterministic layout
            
        Returns:
            Dictionary mapping node IDs to (x, y) coordinate tuples
        """
        if len(graph.nodes()) == 0:
            return {}
        # Use spring layout with specific parameters for good visualization
        pos = nx.spring_layout(graph, seed=seed, k=1, iterations=50)
        return pos

class InteractiveGraphVisualizer:
    """Creates interactive visualizations of cryptocurrency transaction graphs using Plotly"""
    
    def __init__(self):
        # Color mapping for different address classifications
        self.color_map = {
            "licit": "#00FF00",      # Green for legitimate addresses
            "illicit": "#FF0000",    # Red for illicit addresses
            "unknown": "#808080",    # Gray for unknown classification
            "exchange": "#0000FF",   # Blue for exchange addresses
            "mixer": "#800080",      # Purple for mixing services
        }

    def create_plotly_graph(self, graph: nx.DiGraph, positions: Dict, title: str = "Transaction Flow") -> go.Figure:
        """
        Create an interactive Plotly graph visualization with variable edge widths based on transaction amounts
        
        Args:
            graph: NetworkX directed graph of transactions
            positions: Dictionary of node positions from layout computation
            title: Title for the graph visualization
            
        Returns:
            Plotly Figure object ready for display
        """
        if len(graph.nodes()) == 0:
            return self.create_empty_graph()

        # Group edges by width to optimize rendering performance
        width_buckets: Dict[float, list] = {}
        for u, v, ed in graph.edges(data=True):
            amount = ed.get("amount", 0.1)
            # Calculate edge width using logarithmic scale
            width = max(0.5, min(8, np.log10(amount * 100 + 1) + 1))
            width_key = round(width, 1)
            width_buckets.setdefault(width_key, []).append((u, v, ed))

        edge_traces = []
        # Create separate Plotly traces for each edge width group
        for width, edges in width_buckets.items():
            edge_x, edge_y, edge_hover = [], [], []
            for (u, v, ed) in edges:
                # Get coordinates for source and target nodes
                x0, y0 = positions.get(u, (0, 0))
                x1, y1 = positions.get(v, (0, 0))
                edge_x.extend([x0, x1, None])  # None creates line break
                edge_y.extend([y0, y1, None])
                
                # Create hover information for edge
                amount = ed.get("amount", 0)
                timestamp = ed.get("timestamp", "Unknown")
                edge_hover.extend([
                    f"From: {u[:12]}...To: {v[:12]}...Amount: {amount:.6f} BTC Time: {timestamp}",
                    f"From: {u[:12]}...To: {v[:12]}...Amount: {amount:.6f} BTC Time: {timestamp}",
                    "",
                ])

            # Create Plotly scatter trace for this width group
            edge_traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=width, color=f"rgba(125,125,125,{min(0.8, width/8)})"),
                    hoverinfo="text",
                    text=edge_hover,
                    mode="lines",
                    showlegend=False,
                )
            )

        # Create node trace with colors, sizes, and hover information
        node_trace = self.create_node_trace(graph, positions)

        # Compose complete figure with edges and nodes
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Edge thickness represents transaction amount • Node colors: Green=Licit, Red=Illicit, Gray=Unknown",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=10),
                    )
                ],
                # Hide axis labels and grid for cleaner visualization
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
            ),
        )

        return fig

    def create_empty_graph(self):
        """Create a figure displaying 'No graph data available' message when graph is empty"""
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    def create_node_trace(self, graph: nx.DiGraph, positions: Dict) -> go.Scatter:
        """
        Create Plotly scatter trace for graph nodes with colors, sizes, and detailed hover information
        
        Args:
            graph: NetworkX directed graph
            positions: Dictionary mapping node IDs to (x, y) coordinates
            
        Returns:
            Plotly Scatter object representing all nodes
        """
        node_x, node_y, node_info, node_colors, node_sizes = [], [], [], [], []

        # Process each node to extract visual attributes and information
        for node in graph.nodes():
            # Get node position coordinates
            x, y = positions.get(node, (0, 0))
            node_x.append(x)
            node_y.append(y)

            # Determine node color based on classification
            classification = graph.nodes[node].get("classification", "unknown")
            node_colors.append(self.color_map.get(classification, self.color_map["unknown"]))

            # Calculate node size based on degree (number of connections)
            degree = graph.degree(node)
            node_sizes.append(max(10, min(40, degree * 4)))

            # Calculate transaction statistics for hover information
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            total_in = sum(graph[pred][node].get("amount", 0) for pred in graph.predecessors(node))
            total_out = sum(graph[node][succ].get("amount", 0) for succ in graph.successors(node))

            # Create detailed hover tooltip information
            node_info.append(
                f"Address: {node}"
                f"\nClassification: {classification}"
                f"\nIn-degree: {in_degree}"
                f"\nOut-degree: {out_degree}"
                f"\nTotal In: {total_in:.6f} BTC"
                f"\nTotal Out: {total_out:.6f} BTC"
            )

        # Create and return Plotly scatter trace for nodes
        trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_info,
            marker=dict(
                size=node_sizes, 
                color=node_colors, 
                line=dict(width=2, color="black"), 
                opacity=0.9
            ),
        )

        return trace

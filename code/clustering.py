import networkx as nx
from typing import List, Dict, Set
from data_io import Transaction

class EntityClusterer:
    def __init__(self):
        # Dictionary to store clusters of addresses
        self.address_clusters = {}
        
    def build_clusters(self, transactions: List[Transaction]) -> Dict[str, str]:
        """
        Build address clusters from a list of transactions using heuristics such as common input ownership.
        Returns a dictionary mapping each address to a cluster ID.
        """
        cluster_map = {}
        cluster_counter = 0
        
        for tx in transactions:
            # Extract input and output addresses from the transaction
            input_addresses = self._extract_addresses(tx.inputs)
            output_addresses = self._extract_addresses(tx.outputs)
            
            # If multiple input addresses, apply common input ownership heuristic
            # This assumes addresses used together as inputs likely belong to the same entity
            if len(input_addresses) > 1:
                existing_clusters = set()
                for addr in input_addresses:
                    if addr in cluster_map:
                        existing_clusters.add(cluster_map[addr])
                
                if existing_clusters:
                    # Merge addresses into the first found existing cluster
                    target_cluster = list(existing_clusters)[0]
                    for addr in input_addresses:
                        cluster_map[addr] = target_cluster
                else:
                    # Create a new cluster for these input addresses
                    cluster_id = f"cluster_{cluster_counter}"
                    cluster_counter += 1
                    for addr in input_addresses:
                        cluster_map[addr] = cluster_id
            
            # Assign single addresses to unique clusters if not already assigned
            all_addresses = input_addresses + output_addresses
            for addr in all_addresses:
                if addr not in cluster_map:
                    cluster_map[addr] = f"cluster_{cluster_counter}"
                    cluster_counter += 1
                    
        return cluster_map
    
    def _extract_addresses(self, address_data) -> List[str]:
        """
        Extract addresses from input/output fields safely, supporting multiple formats:
        - Live API format: List of strings ['addr1', 'addr2']
        - Synthetic format: List of dicts [{'address': 'addr1'}, {'address': 'addr2'}]
        """
        if not address_data:
            return []
            
        addresses = []
        for item in address_data:
            if isinstance(item, str):
                # Live API format - item is already an address string
                addresses.append(item)
            elif isinstance(item, dict):
                # Synthetic format - extract from dict using common keys
                addr = item.get('address') or item.get('addresses', [None])[0]
                if addr:
                    addresses.append(addr)
        
        # Remove duplicates and filter out empty/unknown addresses
        return [addr for addr in list(set(addresses)) if addr and addr != 'unknown']
    
    def collapse_to_cluster_graph(self, original_graph: nx.DiGraph, cluster_map: Dict[str, str]) -> nx.DiGraph:
        """
        Collapse a detailed transaction graph into a cluster-level graph,
        where each node represents a cluster of addresses.
        Edges represent aggregated transactions between clusters.
        """
        cluster_graph = nx.DiGraph()
        
        # Add nodes for each cluster
        clusters = set(cluster_map.values())
        for cluster_id in clusters:
            cluster_graph.add_node(cluster_id, node_type="cluster")
        
        # Add edges between clusters with aggregated amounts and transaction counts
        for u, v, edge_data in original_graph.edges(data=True):
            if u in cluster_map and v in cluster_map:
                cluster_u = cluster_map[u]
                cluster_v = cluster_map[v]
                
                # Avoid self-loops where source and destination clusters are the same
                if cluster_u != cluster_v:
                    if cluster_graph.has_edge(cluster_u, cluster_v):
                        # Aggregate existing edge data (sum amounts, count transactions)
                        existing_amount = cluster_graph[cluster_u][cluster_v].get('amount', 0)
                        cluster_graph[cluster_u][cluster_v]['amount'] = existing_amount + edge_data.get('amount', 0)
                        cluster_graph[cluster_u][cluster_v]['tx_count'] = cluster_graph[cluster_u][cluster_v].get('tx_count', 0) + 1
                    else:
                        # Create new edge between clusters
                        cluster_graph.add_edge(
                            cluster_u, cluster_v,
                            amount=edge_data.get('amount', 0),
                            tx_count=1
                        )
                        
        return cluster_graph
    
    def get_cluster_members(self, cluster_id: str, cluster_map: Dict[str, str]) -> Set[str]:
        """
        Return all address members belonging to a given cluster ID.
        """
        return {addr for addr, cid in cluster_map.items() if cid == cluster_id}
    
    def get_cluster_stats(self, cluster_map: Dict[str, str]) -> Dict:
        """
        Compute comprehensive statistics about the clustering result:
        - total_clusters: number of unique clusters formed
        - total_addresses: total number of addresses that were clustered
        - avg_cluster_size: average number of addresses per cluster
        - largest_cluster_size: size of the largest cluster (max addresses)
        - cluster_details: dictionary mapping each cluster_id to its size
        """
        clusters = {}
        for addr, cluster_id in cluster_map.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(addr)
            
        return {
            "total_clusters": len(clusters),
            "total_addresses": len(cluster_map),
            "avg_cluster_size": sum(len(members) for members in clusters.values()) / len(clusters) if clusters else 0,
            "largest_cluster_size": max(len(members) for members in clusters.values()) if clusters else 0,
            "cluster_details": {cid: len(members) for cid, members in clusters.items()}
        }

def extract_addresses_safely(address_field) -> List[str]:
    """
    Utility function to safely extract addresses from inputs/outputs fields.
    Supports both live API (list of strings) and synthetic (list of dicts) formats.
    Can be used by other detection modules for consistent address extraction.
    """
    if not address_field:
        return []
    
    addresses = []
    for item in address_field:
        if isinstance(item, str):
            # Live API format - direct string address
            addresses.append(item)
        elif isinstance(item, dict):
            # Synthetic format - extract from dict using common field names
            addr = item.get('address') or item.get('addresses', [None])[0]
            if addr:
                addresses.append(addr)
    
    # Remove duplicates and filter out empty or 'unknown' placeholder addresses
    return [addr for addr in list(set(addresses)) if addr and addr != 'unknown']

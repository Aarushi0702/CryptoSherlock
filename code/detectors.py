import networkx as nx
import numpy as np
import math
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import dateutil.parser

class PatternDetector:
    """Detects suspicious cryptocurrency transaction patterns using graph analysis"""
    
    def __init__(self):
        """Initialize confidence weights for different suspicious patterns"""
        self.confidence_weights = {
            'peel_chain': 0.8,           # Sequential fund transfers with decreasing amounts
            'coinjoin': 0.6,             # Privacy mixing with equal outputs
            'structuring': 0.6,          # Multiple small deposits below thresholds
            'rapid_movement': 0.7,       # Fast sequential transactions across addresses
            'layering': 0.6,             # Complex multi-hop obfuscation
            'tf_crowdfunding': 0.6,      # Terrorist financing through micro-donations
            'chain_hopping': 0.9         # Cross-chain money laundering via exchanges
        }

    def detect_all_patterns(self, graph: nx.DiGraph, target_node: str,
                            transaction_data: Dict = None, chain_hop_events: List = None) -> Dict:
        """
        Run detection for all supported suspicious patterns
        
        Args:
            graph: NetworkX directed graph of transactions
            target_node: Address to analyze for patterns
            transaction_data: Raw transaction data for CoinJoin detection
            chain_hop_events: List of cross-chain events for chain hopping detection
        
        Returns:
            Dictionary with detection results for each pattern
        """
        results = {}
        results['peel_chain'] = self.detect_peel_chain(graph, target_node)
        results['structuring'] = self.detect_structuring(graph, target_node)
        results['rapid_movement'] = self.detect_rapid_movement(graph, target_node)
        results['layering'] = self.detect_layering(graph, target_node)
        results['tf_crowdfunding'] = self.detect_tf_crowdfunding(graph, target_node)
        
        # CoinJoin detection requires raw transaction data
        if transaction_data:
            results['coinjoin'] = self.detect_coinjoin_like(transaction_data)
        else:
            results['coinjoin'] = {
                'pattern': 'coinjoin',
                'confidence': 0.0,
                'explanation': 'No transaction data provided'
            }
        
        # Chain hopping detection from cross-chain events
        results['chain_hopping'] = self.detect_chain_hopping(chain_hop_events or [], target_node)
        
        return results

    def detect_chain_hopping(self, chain_hop_events: List, target_address: str) -> Dict:
        """
        Detect cross-chain chain hopping pattern from detected events
        
        Args:
            chain_hop_events: List of ChainHopEvent objects
            target_address: Address to check for involvement in chain hopping
        
        Returns:
            Detection result with confidence score and evidence
        """
        try:
            if not chain_hop_events:
                return self._create_low_confidence_result('chain_hopping', 'No chain hopping events detected')
            
            # Filter events involving the target address
            relevant_events = []
            for event in chain_hop_events:
                if (hasattr(event, 'btc_out_addr') and target_address in str(event.btc_out_addr)) or \
                   (hasattr(event, 'eth_in_addr') and target_address in str(event.eth_in_addr)):
                    relevant_events.append(event)

            if not relevant_events:
                return self._create_low_confidence_result('chain_hopping', 'No chain hopping events for target address')

            # Calculate confidence based on number of hops and timing patterns
            num_hops = len(relevant_events)
            time_windows = []
            
            for event in relevant_events:
                if hasattr(event, 'btc_time') and hasattr(event, 'eth_time'):
                    try:
                        # Parse timestamps from different formats
                        btc_time = event.btc_time if isinstance(event.btc_time, datetime) else dateutil.parser.parse(str(event.btc_time))
                        eth_time = event.eth_time if isinstance(event.eth_time, datetime) else dateutil.parser.parse(str(event.eth_time))
                        time_diff = abs((eth_time - btc_time).total_seconds() / 3600)  # Convert to hours
                        time_windows.append(time_diff)
                    except:
                        continue

            avg_time_window = sum(time_windows) / len(time_windows) if time_windows else 24
            
            # Confidence scoring: more hops = higher confidence, shorter time windows = higher confidence
            hop_score = min(num_hops / 3, 1.0)  # Cap maximum confidence at 3 or more hops
            time_score = max(0, 1 - (avg_time_window / 48))  # Penalize long time windows (>48h)
            confidence = (hop_score * 0.7 + time_score * 0.3) * 0.9  # Weighted confidence with high base

            labels = [getattr(event, 'label', 'Unknown') for event in relevant_events]

            return {
                'pattern': 'chain_hopping',
                'confidence': confidence,
                'explanation': f"Chain hopping detected: {num_hops} cross-chain transfers via exchanges",
                'evidence': {
                    'hop_count': num_hops,
                    'exchange_labels': labels,
                    'avg_time_window_hours': round(avg_time_window, 2) if time_windows else None,
                    'events': [getattr(event, 'to_dict', lambda: str(event))() for event in relevant_events]
                }
            }

        except Exception as e:
            return self._create_low_confidence_result('chain_hopping', f'Detection failed: {str(e)}')

    def detect_rapid_movement(self, graph: nx.DiGraph, start_node: str) -> Dict:
        """
        Detect rapid sequential movements using real timestamp analysis
        
        Args:
            graph: Transaction graph
            start_node: Starting address for path analysis
        
        Returns:
            Detection result with timing evidence
        """
        try:
            if start_node not in graph:
                return self._create_low_confidence_result('rapid_movement', 'Start node not in graph')

            min_hops = 3
            rapid_threshold_minutes = 60  # Threshold for "rapid" movement

            # Find all paths starting from start_node with minimum hops
            paths = []
            for target in graph.nodes():
                if target != start_node:
                    try:
                        path = nx.shortest_path(graph, start_node, target)
                        if len(path) >= min_hops:
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue

            if not paths:
                return self._create_low_confidence_result('rapid_movement', 'No valid paths found')

            # Analyze real timestamps along the longest path
            longest_path = max(paths, key=len)
            timestamps = []

            for i in range(len(longest_path) - 1):
                current_node = longest_path[i]
                next_node = longest_path[i + 1]
                if graph.has_edge(current_node, next_node):
                    edge_data = graph[current_node][next_node]
                    timestamp_str = edge_data.get('timestamp', '')
                    if timestamp_str:
                        try:
                            timestamp = dateutil.parser.parse(timestamp_str)
                            timestamps.append(timestamp)
                        except Exception:
                            continue

            if len(timestamps) < 2:
                return self._create_low_confidence_result('rapid_movement', 'Insufficient timestamp data')

            # Calculate real time gaps between consecutive transactions
            rapid_gaps = 0
            total_gaps = 0
            gap_minutes = []

            for i in range(len(timestamps) - 1):
                gap = (timestamps[i + 1] - timestamps[i]).total_seconds() / 60
                gap_minutes.append(gap)
                total_gaps += 1
                if gap <= rapid_threshold_minutes:
                    rapid_gaps += 1

            fast_fraction = rapid_gaps / total_gaps if total_gaps > 0 else 0
            avg_gap = sum(gap_minutes) / len(gap_minutes) if gap_minutes else 0

            # Require at least 60% of gaps to be rapid for detection
            if fast_fraction < 0.6:
                return self._create_low_confidence_result('rapid_movement',
                    f'Only {fast_fraction:.1%} of gaps are rapid (< {rapid_threshold_minutes} min)')

            confidence = min(fast_fraction * (len(timestamps) / 8), 1.0)

            return {
                'pattern': 'rapid_movement',
                'confidence': confidence,
                'explanation': f"Rapid movement: {rapid_gaps}/{total_gaps} gaps < {rapid_threshold_minutes} min",
                'evidence': {
                    'rapid_gaps': rapid_gaps,
                    'total_gaps': total_gaps,
                    'fast_fraction': round(fast_fraction, 3),
                    'avg_gap_minutes': round(avg_gap, 2),
                    'threshold_minutes': rapid_threshold_minutes,
                    'real_timestamps_used': len(timestamps)
                }
            }

        except Exception as e:
            return self._create_low_confidence_result('rapid_movement', f'Detection failed: {str(e)}')

    def detect_layering(self, graph: nx.DiGraph, start_node: str) -> Dict:
        """
        Detect layering pattern using Shannon entropy analysis and consolidation detection
        
        Args:
            graph: Transaction graph
            start_node: Starting address for layering analysis
        
        Returns:
            Detection result with entropy trend and consolidation evidence
        """
        try:
            if start_node not in graph:
                return self._create_low_confidence_result('layering', 'Start node not in graph')

            min_hops = 6  # Layering requires long transaction chains

            # Find long paths that could indicate layering
            long_paths = []
            for target in graph.nodes():
                if target != start_node:
                    try:
                        path = nx.shortest_path(graph, start_node, target)
                        if len(path) >= min_hops:
                            long_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue

            if not long_paths:
                return self._create_low_confidence_result('layering', 'No sufficiently long paths')

            # Analyze longest path with Shannon entropy calculation
            longest_path = max(long_paths, key=len)
            path_length = len(longest_path)

            entropies = []
            consolidation_scores = []

            for i, node in enumerate(longest_path[:-1]):
                # Calculate Shannon entropy of outflow distributions
                outflows = []
                for successor in graph.successors(node):
                    amount = graph[node][successor].get('amount', 0)
                    if amount > 0:
                        outflows.append(amount)

                if len(outflows) > 1:
                    total = sum(outflows)
                    if total > 0:
                        probabilities = [amount / total for amount in outflows]
                        # Shannon entropy formula: -Σ(p * log2(p))
                        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                        entropies.append(entropy)
                    else:
                        entropies.append(0)
                else:
                    entropies.append(0)

                # Check for consolidation in later stages of the path
                if i > path_length // 2:
                    in_degree = graph.in_degree(node)
                    consolidation_scores.append(in_degree)

            # Calculate entropy trend (increasing entropy indicates layering)
            if len(entropies) >= 3:
                early_entropy = np.mean(entropies[:len(entropies)//2]) if entropies[:len(entropies)//2] else 0
                late_entropy = np.mean(entropies[len(entropies)//2:]) if entropies[len(entropies)//2:] else 0
                entropy_trend = late_entropy / early_entropy if early_entropy > 0 else 1.0
            else:
                entropy_trend = 1.0

            # Detect late-stage consolidation
            consolidation_point = -1
            max_in_degree = 0
            if consolidation_scores:
                max_in_degree = max(consolidation_scores)
                if max_in_degree >= 3:
                    consolidation_point = len(longest_path) - len(consolidation_scores) + consolidation_scores.index(max_in_degree)

            has_consolidation = consolidation_point != -1 and max_in_degree >= 3

            # Calculate layering confidence based on entropy trend, path length, and consolidation
            entropy_score = min(entropy_trend / 2, 0.4) if entropy_trend > 1.1 else 0
            length_score = min(path_length / 15, 0.4)
            consolidation_score = 0.6 if has_consolidation else 0.1

            confidence = entropy_score + length_score + consolidation_score

            if confidence < 0.3:
                return self._create_low_confidence_result('layering',
                    f'Insufficient layering pattern (entropy: {entropy_trend:.2f}, consolidation: {has_consolidation})')

            return {
                'pattern': 'layering',
                'confidence': min(confidence, 1.0),
                'explanation': f"Layering: {path_length} hops, entropy trend {entropy_trend:.2f}",
                'evidence': {
                    'hop_count': path_length,
                    'shannon_entropies': [round(e, 3) for e in entropies[:5]],
                    'entropy_trend': round(entropy_trend, 3),
                    'consolidation_point': consolidation_point,
                    'max_consolidation_degree': max_in_degree,
                    'has_late_consolidation': has_consolidation
                }
            }

        except Exception as e:
            return self._create_low_confidence_result('layering', f'Detection failed: {str(e)}')

    def detect_structuring(self, graph: nx.DiGraph, target_node: str, window_hours: int = 24) -> Dict:
        """
        Detect structuring pattern based on small repeated deposits within a time window
        
        Args:
            graph: Transaction graph
            target_node: Address receiving potential structured deposits
            window_hours: Time window for clustering deposits
        
        Returns:
            Detection result with donor count and timing evidence
        """
        try:
            if target_node not in graph:
                return self._create_low_confidence_result('structuring', 'Target node not in graph')

            small_deposit_threshold = 0.01  # BTC threshold for small deposits
            min_donors = 10                 # Minimum unique donors required
            min_deposits = 15               # Minimum total deposits required

            # Collect all small incoming transactions
            incoming_txs = []
            unique_donors = set()

            for predecessor in graph.predecessors(target_node):
                if graph.has_edge(predecessor, target_node):
                    edge_data = graph[predecessor][target_node]
                    amount = edge_data.get('amount', 0)
                    timestamp_str = edge_data.get('timestamp', '')

                    if amount <= small_deposit_threshold:
                        timestamp = None
                        if timestamp_str:
                            try:
                                timestamp = dateutil.parser.parse(timestamp_str)
                            except Exception:
                                pass

                        incoming_txs.append({
                            'amount': amount,
                            'from_addr': predecessor,
                            'timestamp': timestamp
                        })
                        unique_donors.add(predecessor)

            effective_donors = len(unique_donors)
            effective_deposits = len(incoming_txs)
            onward_rate = 0

            # Apply time window filtering to find clusters of deposits
            if window_hours > 0:
                timestamped_txs = [tx for tx in incoming_txs if tx['timestamp'] is not None]
                if timestamped_txs:
                    timestamped_txs.sort(key=lambda x: x['timestamp'])

                    # Find the best time window with maximum donors
                    best_window_size = 0
                    best_window_start = None

                    for start_tx in timestamped_txs:
                        window_end = start_tx['timestamp'] + timedelta(hours=window_hours)
                        window_donors = set()
                        window_txs = []

                        for tx in timestamped_txs:
                            if start_tx['timestamp'] <= tx['timestamp'] <= window_end:
                                window_donors.add(tx['from_addr'])
                                window_txs.append(tx)

                        if len(window_donors) > best_window_size:
                            best_window_size = len(window_donors)
                            best_window_start = start_tx['timestamp']

                    effective_donors = best_window_size

                    # Calculate onward transaction rate after deposits
                    if best_window_start:
                        window_end = best_window_start + timedelta(hours=window_hours)
                        onward_txs = 0
                        total_out = 0

                        for succ in graph.successors(target_node):
                            if graph.has_edge(target_node, succ):
                                out_timestamp_str = graph[target_node][succ].get('timestamp', '')
                                total_out += 1
                                if out_timestamp_str:
                                    try:
                                        out_timestamp = dateutil.parser.parse(out_timestamp_str)
                                        if window_end <= out_timestamp <= window_end + timedelta(hours=24):
                                            onward_txs += 1
                                    except Exception:
                                        pass

                        onward_rate = onward_txs / total_out if total_out > 0 else 0

            if effective_donors < min_donors or effective_deposits < min_deposits:
                return self._create_low_confidence_result('structuring',
                    f'Insufficient activity: {effective_donors} donors, {effective_deposits} deposits')

            # Calculate confidence with penalty for high onward rate (legitimate activity)
            base_confidence = min(effective_donors / (min_donors * 2), 0.7)
            onward_penalty = max(0, onward_rate - 0.5) * 0.3
            confidence = min(base_confidence - onward_penalty + 0.3, 1.0)

            return {
                'pattern': 'structuring',
                'confidence': confidence,
                'explanation': f"{effective_deposits} deposits from {effective_donors} donors in {window_hours}h",
                'evidence': {
                    'effective_donors': effective_donors,
                    'effective_deposits': effective_deposits,
                    'window_hours': window_hours,
                    'threshold_btc': small_deposit_threshold,
                    'onward_rate': round(onward_rate, 3)
                }
            }

        except Exception as e:
            return self._create_low_confidence_result('structuring', f'Detection failed: {str(e)}')

    def detect_tf_crowdfunding(self, graph: nx.DiGraph, target_node: str, window_hours: int = 24) -> Dict:
        """
        Detect terrorist financing crowdfunding pattern through micro-donations
        
        Args:
            graph: Transaction graph
            target_node: Address potentially receiving terrorist financing
            window_hours: Time window for clustering donations
        
        Returns:
            Detection result with donor analysis and fund flow evidence
        """
        try:
            if target_node not in graph:
                return self._create_low_confidence_result('tf_crowdfunding', 'Target node not in graph')

            small_donation_threshold = 0.01  # BTC threshold for micro-donations
            min_donors = 15                  # Higher threshold than structuring

            donations = []
            unique_donors = set()
            total_raised = 0

            # Collect all small donations to target
            for donor in graph.predecessors(target_node):
                if graph.has_edge(donor, target_node):
                    edge_data = graph[donor][target_node]
                    amount = edge_data.get('amount', 0)
                    timestamp_str = edge_data.get('timestamp', '')

                    if amount <= small_donation_threshold:
                        timestamp = None
                        if timestamp_str:
                            try:
                                timestamp = dateutil.parser.parse(timestamp_str)
                            except Exception:
                                pass

                        donations.append({
                            'donor': donor,
                            'amount': amount,
                            'timestamp': timestamp
                        })
                        unique_donors.add(donor)
                        total_raised += amount

            effective_donors = len(unique_donors)
            onward_rate = 0

            # Find the best cluster of donations within time window
            if window_hours > 0:
                timestamped_donations = [d for d in donations if d['timestamp'] is not None]
                if timestamped_donations:
                    best_cluster_size = 0
                    best_window_start = None

                    for start_donation in timestamped_donations:
                        window_end = start_donation['timestamp'] + timedelta(hours=window_hours)
                        cluster_donors = set()

                        for donation in timestamped_donations:
                            if start_donation['timestamp'] <= donation['timestamp'] <= window_end:
                                cluster_donors.add(donation['donor'])

                        if len(cluster_donors) > best_cluster_size:
                            best_cluster_size = len(cluster_donors)
                            best_window_start = start_donation['timestamp']

                    effective_donors = best_cluster_size

                    # Calculate onward transaction rate (higher rate suggests terrorist financing)
                    if best_window_start:
                        window_end = best_window_start + timedelta(hours=window_hours)
                        onward_txs = 0
                        total_out = 0

                        for succ in graph.successors(target_node):
                            if graph.has_edge(target_node, succ):
                                total_out += 1
                                out_timestamp_str = graph[target_node][succ].get('timestamp', '')
                                if out_timestamp_str:
                                    try:
                                        out_timestamp = dateutil.parser.parse(out_timestamp_str)
                                        if window_end <= out_timestamp <= window_end + timedelta(hours=48):
                                            onward_txs += 1
                                    except Exception:
                                        pass

                        onward_rate = onward_txs / total_out if total_out > 0 else 0

            if effective_donors < min_donors:
                return self._create_low_confidence_result('tf_crowdfunding',
                    f'Only {effective_donors} donors (need ≥{min_donors})')

            # Calculate confidence with bonus for high onward rate (funds being used)
            base_confidence = min(effective_donors / (min_donors * 1.5), 0.8)
            onward_bonus = min(onward_rate * 0.4, 0.2)
            confidence = min(base_confidence + onward_bonus + 0.2, 1.0)

            return {
                'pattern': 'tf_crowdfunding',
                'confidence': confidence,
                'explanation': f"{len(donations)} donations from {effective_donors} donors, {total_raised:.4f} BTC raised",
                'evidence': {
                    'effective_donors': effective_donors,
                    'total_donations': len(donations),
                    'total_raised_btc': round(total_raised, 6),
                    'onward_rate': round(onward_rate, 3)
                }
            }

        except Exception as e:
            return self._create_low_confidence_result('tf_crowdfunding', f'Detection failed: {str(e)}')

    def detect_peel_chain(self, graph: nx.DiGraph, start_node: str, min_hops: int = 3) -> Dict:
        """
        Detect peel chain pattern: sequential fund transfers with decreasing amounts
        
        Args:
            graph: Transaction graph
            start_node: Starting address for peel chain analysis
            min_hops: Minimum number of hops required for detection
        
        Returns:
            Detection result with amount sequence and dominance evidence
        """
        try:
            if start_node not in graph:
                return self._create_low_confidence_result('peel_chain', 'Start node not in graph')

            # Find all paths from start_node with sufficient length
            paths = []
            for target in graph.nodes():
                if target != start_node:
                    try:
                        path = nx.shortest_path(graph, start_node, target)
                        if len(path) >= min_hops:
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue

            if not paths:
                return self._create_low_confidence_result('peel_chain', 'No valid paths found')

            # Analyze amount patterns along longest path
            longest_path = max(paths, key=len)
            amounts = []
            dominant_ratios = []

            for i in range(len(longest_path) - 1):
                current_node = longest_path[i]
                next_node = longest_path[i + 1]
                if not graph.has_edge(current_node, next_node):
                    continue

                edge_data = graph[current_node][next_node]
                amount = edge_data.get('amount', 0)
                amounts.append(amount)

                # Calculate if this output is dominant among all outputs from current node
                total_out = sum(graph[current_node][successor].get('amount', 0)
                               for successor in graph.successors(current_node))
                if total_out > 0:
                    dominant_ratios.append(amount / total_out)
                else:
                    dominant_ratios.append(0)

            if len(amounts) < 2:
                return self._create_low_confidence_result('peel_chain', 'Insufficient path data')

            # Check for decreasing amount pattern (hallmark of peel chains)
            decreasing_count = sum(1 for i in range(len(amounts) - 1)
                                  if amounts[i] > amounts[i + 1])
            decreasing_fraction = decreasing_count / max(len(amounts) - 1, 1)

            # Check for dominant output pattern (most funds go to one address)
            dominant_count = sum(1 for ratio in dominant_ratios if ratio >= 0.6)
            dominant_fraction = dominant_count / max(len(dominant_ratios), 1)

            # Require both decreasing amounts and dominant outputs
            if decreasing_fraction < 0.5 or dominant_fraction < 0.4:
                return self._create_low_confidence_result('peel_chain',
                    f'Insufficient pattern: {decreasing_fraction:.1%} decreasing, {dominant_fraction:.1%} dominant')

            # Calculate confidence based on pattern strength and path length
            confidence = (decreasing_fraction * 0.6 + dominant_fraction * 0.4) * min(len(amounts) / 5, 1.0)

            return {
                'pattern': 'peel_chain',
                'confidence': min(confidence, 1.0),
                'explanation': f"{decreasing_count}/{len(amounts)-1} hops show decreasing outputs",
                'evidence': {
                    'amount_sequence': [round(a, 6) for a in amounts[:5]],
                    'hop_count': len(amounts),
                    'decreasing_fraction': round(decreasing_fraction, 3),
                    'dominant_fraction': round(dominant_fraction, 3)
                }
            }

        except Exception as e:
            return self._create_low_confidence_result('peel_chain', f'Detection failed: {str(e)}')

    def detect_coinjoin_like(self, transaction_data: Dict) -> Dict:
        """
        Detect CoinJoin-like mixing based on multiple inputs and equal outputs
        
        Args:
            transaction_data: Raw transaction data with inputs and outputs
        
        Returns:
            Detection result with equal output analysis and script diversity
        """
        try:
            if not transaction_data or 'inputs' not in transaction_data or 'outputs' not in transaction_data:
                return self._create_low_confidence_result('coinjoin', 'Insufficient transaction data')

            inputs = transaction_data['inputs']
            outputs = transaction_data['outputs']

            if len(inputs) < 2:
                return self._create_low_confidence_result('coinjoin', 'Too few inputs')

            # Group outputs by amount to find equal outputs (CoinJoin characteristic)
            output_amounts = {}
            for output in outputs:
                amount = round(output.get('amount', 0), 3)
                if amount > 0:
                    output_amounts[amount] = output_amounts.get(amount, 0) + 1

            if not output_amounts:
                return self._create_low_confidence_result('coinjoin', 'No valid outputs')

            # Find the largest group of equal outputs
            max_equal_outputs = max(output_amounts.values())
            common_amount = max(output_amounts.keys(), key=lambda k: output_amounts[k])

            if max_equal_outputs < 3:
                return self._create_low_confidence_result('coinjoin',
                    f'Only {max_equal_outputs} equal outputs (need ≥3)')

            # Check input script diversity (another CoinJoin indicator)
            script_types = set(inp.get('script_type', '') for inp in inputs if inp.get('script_type'))
            script_diversity = len(script_types) / len(inputs) if inputs else 0

            # Calculate confidence based on equal outputs and script diversity
            output_score = min(max_equal_outputs / 8, 1.0)
            diversity_score = min(script_diversity * 2, 1.0)
            confidence = output_score * 0.7 + diversity_score * 0.3

            return {
                'pattern': 'coinjoin',
                'confidence': confidence,
                'explanation': f"{max_equal_outputs} equal outputs of {common_amount} BTC",
                'evidence': {
                    'equal_outputs_count': max_equal_outputs,
                    'common_amount': common_amount,
                    'input_count': len(inputs),
                    'script_diversity': round(script_diversity, 3)
                }
            }

        except Exception as e:
            return self._create_low_confidence_result('coinjoin', f'Detection failed: {str(e)}')

    def _create_low_confidence_result(self, pattern: str, explanation: str) -> Dict:
        """
        Helper method to create uniform low confidence results for failed detections
        
        Args:
            pattern: Name of the pattern that failed to detect
            explanation: Reason for low confidence
        
        Returns:
            Standardized low confidence result dictionary
        """
        return {
            'pattern': pattern,
            'confidence': 0.0,
            'explanation': explanation,
            'evidence': {}
        }

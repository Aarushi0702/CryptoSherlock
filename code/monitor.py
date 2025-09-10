import time
from typing import Dict, List, Set
from datetime import datetime

class WatchlistMonitor:
    def __init__(self):
        self.watchlist: Set[str] = set()
        self.alerts: List[Dict] = []
        self.thresholds = {
            'volume_btc': 10.0,
            'transaction_count': 50, 
            'risk_score': 0.7 
        }

    def add_to_watchlist(self, address: str):
        """Add address to monitoring watchlist"""
        self.watchlist.add(address)
        return f"Added {address[:12]}... to watchlist"

    def remove_from_watchlist(self, address: str):
        """Remove address from watchlist"""
        self.watchlist.discard(address)
        return f"Removed {address[:12]}... from watchlist"

    def check_watchlist_activity(self, graph, detector, scorer) -> List[Dict]:
    
        new_alerts = []
        for address in self.watchlist:
            if address in graph.nodes():
                # Calculate VOLUME_BTC metrics
                total_in_volume = sum(
                    graph[pred][address].get('amount', 0)
                    for pred in graph.predecessors(address)
                )
                total_out_volume = sum(
                    graph[address][succ].get('amount', 0)
                    for succ in graph.successors(address)
                )
                total_volume = total_in_volume + total_out_volume

                # Calculate TRANSACTION_COUNT
                transaction_count = graph.in_degree(address) + graph.out_degree(address)

                # Run pattern detection
                detection_results = detector.detect_all_patterns(graph, address)
                risk_score, risk_level, summary = scorer.compute_risk_score(detection_results)

                # Check ALL thresholds - INCLUDES VOLUME_BTC AND TRANSACTION_COUNT
                alerts_triggered = []
                if total_volume >= self.thresholds['volume_btc']:
                    alerts_triggered.append(f"Volume: {total_volume:.2f} BTC ≥ {self.thresholds['volume_btc']}")
                if transaction_count >= self.thresholds['transaction_count']:
                    alerts_triggered.append(f"Transactions: {transaction_count} ≥ {self.thresholds['transaction_count']}")
                if risk_score >= self.thresholds['risk_score']:
                    alerts_triggered.append(f"Risk Score: {risk_score:.2f} ≥ {self.thresholds['risk_score']}")

                # Generate alert if ANY threshold exceeded
                if alerts_triggered:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'address': address,
                        'risk_score': risk_score,
                        'risk_level': risk_level,
                        'total_volume_btc': total_volume, # INCLUDED
                        'transaction_count': transaction_count, # INCLUDED
                        'patterns': [p['pattern'] for p in summary['active_patterns']],
                        'alert_triggers': alerts_triggered,
                        'alert_type': 'MULTI_THRESHOLD_EXCEEDED'
                    }
                    new_alerts.append(alert)
                    self.alerts.append(alert)
        return new_alerts

    def check_illegal_transactions_alerts(self, graph, detector, scorer) -> List[Dict]:
  
        new_alerts = []
        
        for node in graph.nodes():
            classification = graph.nodes[node].get('classification', 'unknown')
            if classification == 'illicit':
                # Run pattern detection for this node
                detection_results = detector.detect_all_patterns(graph, node)
                risk_score, risk_level, summary = scorer.compute_risk_score(detection_results)
                
                # Create alert for illegal transaction
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'address': node,
                    'classification': classification,
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'message': f"🚨 ILLEGAL TRANSACTION DETECTED at {node[:16]}...",
                    'patterns': [p['pattern'] for p in summary.get('active_patterns', [])],
                    'alert_type': 'ILLEGAL_TRANSACTION'
                }
                new_alerts.append(alert)
                
        # Add to existing alerts list
        self.alerts.extend(new_alerts)
        return new_alerts

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from recent time period"""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        recent_alerts = []
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert['timestamp']).timestamp()
            if alert_time >= cutoff:
                recent_alerts.append(alert)
        return recent_alerts

    def get_watchlist_status(self) -> Dict:

        return {
            'addresses_monitored': len(self.watchlist),
            'total_alerts': len(self.alerts),
            'recent_alerts': len(self.get_recent_alerts()),
            'watchlist_addresses': list(self.watchlist),
            'thresholds': self.thresholds.copy() # Include all thresholds
        }

    def update_thresholds(self, volume_btc: float = None, transaction_count: int = None, risk_score: float = None):
   
        if volume_btc is not None:
            self.thresholds['volume_btc'] = volume_btc
        if transaction_count is not None:
            self.thresholds['transaction_count'] = transaction_count
        if risk_score is not None:
            self.thresholds['risk_score'] = risk_score


import time
from typing import Dict, List, Set
from datetime import datetime

class WatchlistMonitor:
    """Class to monitor watchlisted addresses for suspicious activity based on defined thresholds"""
    def __init__(self):
        # Set of addresses currently being monitored
        self.watchlist: Set[str] = set()
        # List of alerts generated over time
        self.alerts: List[Dict] = []
        # Threshold values for triggering alerts
        self.thresholds = {
            'volume_btc': 10.0,              # Minimum total volume in BTC
            'transaction_count': 50,          # Minimum transaction count
            'risk_score': 0.7                 # Minimum composite risk score
        }
    
    def add_to_watchlist(self, address: str):
        """Add a cryptocurrency address to the watchlist for monitoring"""
        self.watchlist.add(address)
        return f"Added {address[:12]}... to watchlist"
    
    def remove_from_watchlist(self, address: str):
        """Remove an address from the watchlist"""
        self.watchlist.discard(address)
        return f"Removed {address[:12]}... from watchlist"
    
    def check_watchlist_activity(self, graph, detector, scorer) -> List[Dict]:
        """
        Assess addresses in the watchlist for suspicious activity by applying thresholds
        
        Args:
            graph: NetworkX graph containing transaction data
            detector: Pattern detector for identifying suspicious behavior
            scorer: Risk scorer for computing overall risk levels
            
        Returns:
            List of alert dictionaries for addresses exceeding thresholds
        """
        new_alerts = []
        
        # Analyze each address in the watchlist
        for address in self.watchlist:
            if address in graph.nodes():
                # Calculate total incoming transaction volume for the address
                total_in_volume = sum(
                    graph[pred][address].get('amount', 0) 
                    for pred in graph.predecessors(address)
                )
                # Calculate total outgoing transaction volume for the address
                total_out_volume = sum(
                    graph[address][succ].get('amount', 0) 
                    for succ in graph.successors(address)
                )
                total_volume = total_in_volume + total_out_volume
                
                # Calculate total number of transactions (in + out)
                transaction_count = graph.in_degree(address) + graph.out_degree(address)
                
                # Use provided detector object to find suspicious patterns
                detection_results = detector.detect_all_patterns(graph, address)
                # Use scorer object to get risk score and level
                risk_score, risk_level, summary = scorer.compute_risk_score(detection_results)
                
                alerts_triggered = []
                
                # Check if total volume exceeds threshold
                if total_volume >= self.thresholds['volume_btc']:
                    alerts_triggered.append(f"Volume: {total_volume:.2f} BTC ≥ {self.thresholds['volume_btc']}")
                
                # Check if transaction count exceeds threshold
                if transaction_count >= self.thresholds['transaction_count']:
                    alerts_triggered.append(f"Transactions: {transaction_count} ≥ {self.thresholds['transaction_count']}")
                
                # Check if risk score exceeds threshold
                if risk_score >= self.thresholds['risk_score']:
                    alerts_triggered.append(f"Risk Score: {risk_score:.2f} ≥ {self.thresholds['risk_score']}")
                
                # Create alert if any thresholds exceeded
                if alerts_triggered:
                    alert = {
                        'timestamp': datetime.now().isoformat(),    # When alert was generated
                        'address': address,                         # Address that triggered alert
                        'risk_score': risk_score,                   # Computed risk score
                        'risk_level': risk_level,                   # Risk level classification
                        'total_volume_btc': total_volume,           # Total transaction volume
                        'transaction_count': transaction_count,     # Total transaction count
                        'patterns': [p['pattern'] for p in summary['active_patterns']],  # Detected patterns
                        'alert_triggers': alerts_triggered,         # Which thresholds were exceeded
                        'alert_type': 'MULTI_THRESHOLD_EXCEEDED'    # Type of alert
                    }
                    
                    new_alerts.append(alert)
                    self.alerts.append(alert)
        
        return new_alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """
        Fetch alerts from the last given number of hours
        
        Args:
            hours: Number of hours to look back for alerts
            
        Returns:
            List of alert dictionaries from the specified time period
        """
        cutoff = datetime.now().timestamp() - (hours * 3600)
        recent_alerts = []
        
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert['timestamp']).timestamp()
            if alert_time >= cutoff:
                recent_alerts.append(alert)
        
        return recent_alerts
    
    def get_watchlist_status(self) -> Dict:
        """
        Return summary status of the watchlist and alerts
        
        Returns:
            Dictionary containing watchlist statistics and configuration
        """
        return {
            'addresses_monitored': len(self.watchlist),        # Number of addresses being monitored
            'total_alerts': len(self.alerts),                    # Total alerts generated
            'recent_alerts': len(self.get_recent_alerts()),      # Alerts within recent timeframe
            'watchlist_addresses': list(self.watchlist),         # List of watchlisted addresses
            'thresholds': self.thresholds.copy()                  # Current threshold settings
        }
    
    def update_thresholds(self, volume_btc: float = None, transaction_count: int = None, risk_score: float = None):
        """
        Update monitoring thresholds for alerts
        
        Args:
            volume_btc: New volume threshold in BTC (optional)
            transaction_count: New transaction count threshold (optional)
            risk_score: New risk score threshold (optional)
        """
        if volume_btc is not None:
            self.thresholds['volume_btc'] = volume_btc
        if transaction_count is not None:
            self.thresholds['transaction_count'] = transaction_count
        if risk_score is not None:
            self.thresholds['risk_score'] = risk_score

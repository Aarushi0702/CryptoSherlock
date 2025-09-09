import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import json
from dataclasses import dataclass, asdict
from enum import Enum
import streamlit as st

class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    HIGH_RISK_WALLET = "HIGH_RISK_WALLET"
    SUSPICIOUS_CLUSTER = "SUSPICIOUS_CLUSTER"
    PATTERN_DETECTED = "PATTERN_DETECTED"
    CHAIN_HOPPING = "CHAIN_HOPPING"
    LARGE_TRANSACTION = "LARGE_TRANSACTION"
    RAPID_MOVEMENT = "RAPID_MOVEMENT"
    MIXING_ACTIVITY = "MIXING_ACTIVITY"
    THRESHOLD_BREACH = "THRESHOLD_BREACH"

@dataclass
class Alert:
    id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    address: str
    risk_score: float
    evidence: Dict
    auto_generated: bool = True
    acknowledged: bool = False
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['alert_type'] = self.alert_type.value
        data['severity'] = self.severity.value
        return data

class AlertManager:
    """Manages real-time alerts for suspicious cryptocurrency activities"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_thresholds = {
            'risk_score_threshold': 0.7,
            'large_transaction_btc': 10.0,
            'rapid_movement_minutes': 60,
            'cluster_size_threshold': 5,
            'chain_hop_time_window': 24
        }
        
    def add_alert_callback(self, callback: Callable):
        """Register callback function to be called when new alerts are generated"""
        self.alert_callbacks.append(callback)
        
    def generate_alert(self, alert_type: AlertType, severity: AlertSeverity, 
                      title: str, message: str, address: str, 
                      risk_score: float, evidence: Dict) -> Alert:
        """Generate a new alert and notify all registered callbacks"""
        
        alert = Alert(
            id=f"alert_{int(time.time())}_{len(self.alerts)}",
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            address=address,
            risk_score=risk_score,
            evidence=evidence
        )
        
        self.alerts.append(alert)
        
        # Trigger all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
                
        return alert
    
    def check_high_risk_wallet(self, address: str, risk_score: float, 
                              detection_results: Dict) -> Optional[Alert]:
        """Check if a wallet qualifies as high-risk and generate alert"""
        
        if risk_score >= self.alert_thresholds['risk_score_threshold']:
            severity = AlertSeverity.CRITICAL if risk_score >= 0.9 else AlertSeverity.HIGH
            
            active_patterns = [pattern for pattern, result in detection_results.items() 
                             if result.get('confidence', 0) > 0.5]
            
            return self.generate_alert(
                alert_type=AlertType.HIGH_RISK_WALLET,
                severity=severity,
                title=f"🚨 High-Risk Wallet Detected",
                message=f"Wallet {address[:16]}... shows risk score {risk_score:.3f} with patterns: {', '.join(active_patterns)}",
                address=address,
                risk_score=risk_score,
                evidence={
                    'patterns_detected': active_patterns,
                    'detection_results': detection_results,
                    'threshold_exceeded': self.alert_thresholds['risk_score_threshold']
                }
            )
        return None
    
    def check_suspicious_cluster(self, cluster_id: str, member_addresses: List[str], 
                               cluster_risk_score: float, aggregated_patterns: Dict) -> Optional[Alert]:
        """Check if a cluster is suspicious and generate alert"""
        
        if (len(member_addresses) >= self.alert_thresholds['cluster_size_threshold'] and 
            cluster_risk_score >= 0.6):
            
            severity = AlertSeverity.HIGH if cluster_risk_score >= 0.8 else AlertSeverity.MEDIUM
            
            return self.generate_alert(
                alert_type=AlertType.SUSPICIOUS_CLUSTER,
                severity=severity,
                title=f"⚠️ Suspicious Address Cluster",
                message=f"Cluster {cluster_id} with {len(member_addresses)} addresses shows coordinated suspicious activity (Risk: {cluster_risk_score:.3f})",
                address=cluster_id,
                risk_score=cluster_risk_score,
                evidence={
                    'cluster_size': len(member_addresses),
                    'member_addresses': member_addresses[:10],  # Limit for storage
                    'aggregated_patterns': aggregated_patterns
                }
            )
        return None
    
    def check_chain_hopping(self, chain_hop_events: List, target_address: str) -> Optional[Alert]:
        """Check for chain hopping activity and generate alert"""
        
        if chain_hop_events:
            # Filter events involving target address
            relevant_events = []
            for event in chain_hop_events:
                if (hasattr(event, 'btc_out_addr') and target_address in str(event.btc_out_addr)) or \
                   (hasattr(event, 'eth_in_addr') and target_address in str(event.eth_in_addr)):
                    relevant_events.append(event)
            
            if relevant_events:
                return self.generate_alert(
                    alert_type=AlertType.CHAIN_HOPPING,
                    severity=AlertSeverity.HIGH,
                    title=f"🔗 Chain Hopping Detected",
                    message=f"Address {target_address[:16]}... involved in {len(relevant_events)} cross-chain transfers",
                    address=target_address,
                    risk_score=0.9,
                    evidence={
                        'hop_count': len(relevant_events),
                        'events': [getattr(event, 'to_dict', lambda: str(event))() for event in relevant_events[:5]]
                    }
                )
        return None
    
    def check_large_transaction(self, transaction, threshold: float = None) -> Optional[Alert]:
        """Check for unusually large transactions"""
        
        threshold = threshold or self.alert_thresholds['large_transaction_btc']
        
        if transaction.value >= threshold:
            severity = AlertSeverity.HIGH if transaction.value >= threshold * 2 else AlertSeverity.MEDIUM
            
            return self.generate_alert(
                alert_type=AlertType.LARGE_TRANSACTION,
                severity=severity,
                title=f"💰 Large Transaction Alert",
                message=f"Large transaction of {transaction.value:.6f} BTC detected from {transaction.from_address[:16]}... to {transaction.to_address[:16]}...",
                address=transaction.from_address,
                risk_score=min(transaction.value / 100, 1.0),  # Scale risk with amount
                evidence={
                    'transaction_id': transaction.txid,
                    'amount_btc': transaction.value,
                    'from_address': transaction.from_address,
                    'to_address': transaction.to_address,
                    'threshold': threshold
                }
            )
        return None
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff]
    
    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical unacknowledged alerts"""
        return [alert for alert in self.alerts 
                if alert.severity == AlertSeverity.CRITICAL and not alert.acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def clear_old_alerts(self, days: int = 30):
        """Remove alerts older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff]
    
    def update_thresholds(self, new_thresholds: Dict):
        """Update alert thresholds"""
        self.alert_thresholds.update(new_thresholds)
    
    def export_alerts(self, format: str = 'json') -> str:
        """Export alerts in specified format"""
        alerts_data = [alert.to_dict() for alert in self.alerts]
        
        if format == 'json':
            return json.dumps(alerts_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

class RealTimeMonitor:
    """Real-time monitoring service for blockchain activities"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_check_time = datetime.now()
        self.check_interval = 30  # seconds
        
    def start_monitoring(self, graph_builder, detector, scorer, clusterer):
        """Start real-time monitoring thread"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(graph_builder, detector, scorer, clusterer),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self, graph_builder, detector, scorer, clusterer):
        """Main monitoring loop - runs in background thread"""
        
        while self.monitoring_active:
            try:
                # Check session state for new data
                if hasattr(st.session_state, 'transactions') and st.session_state.transactions:
                    self._check_new_activities(graph_builder, detector, scorer, clusterer)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_new_activities(self, graph_builder, detector, scorer, clusterer):
        """Check for new suspicious activities"""
        
        try:
            # Get current data
            transactions = st.session_state.transactions
            graph = st.session_state.graph
            
            if not graph or not transactions:
                return
            
            # Check each address in the graph for suspicious activity
            for address in graph.nodes():
                try:
                    # Run pattern detection
                    detection_results = detector.detect_all_patterns(
                        graph, address, 
                        transaction_data=None,
                        chain_hop_events=getattr(st.session_state, 'chain_hop_events', [])
                    )
                    
                    # Calculate risk score
                    risk_score, risk_level, summary = scorer.compute_risk_score(detection_results)
                    
                    # Check for high-risk wallet
                    self.alert_manager.check_high_risk_wallet(address, risk_score, detection_results)
                    
                    # Check for chain hopping
                    if hasattr(st.session_state, 'chain_hop_events'):
                        self.alert_manager.check_chain_hopping(
                            st.session_state.chain_hop_events, address
                        )
                    
                except Exception as e:
                    continue
            
            # Check for large transactions
            recent_transactions = [tx for tx in transactions 
                                 if self._is_recent_transaction(tx)]
            
            for tx in recent_transactions:
                self.alert_manager.check_large_transaction(tx)
            
            # Update last check time
            self.last_check_time = datetime.now()
            
        except Exception as e:
            print(f"Error checking activities: {e}")
    
    def _is_recent_transaction(self, transaction) -> bool:
        """Check if transaction is recent (within last check interval)"""
        try:
            import pandas as pd
            tx_time = pd.to_datetime(transaction.timestamp)
            return (datetime.now() - tx_time.to_pydatetime()).total_seconds() <= self.check_interval * 2
        except:
            return False

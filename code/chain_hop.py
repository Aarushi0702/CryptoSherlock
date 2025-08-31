import json
from datetime import timedelta, datetime
import dateutil.parser

class ChainHopEvent:
    """Represents a detected chain hopping event between Bitcoin and Ethereum transactions"""
    def __init__(self, btc_txid, btc_out_addr, eth_in_addr, btc_time, eth_time, label):
        self.btc_txid = btc_txid           # Bitcoin transaction ID
        self.btc_out_addr = btc_out_addr   # BTC output address (exchange/bridge)
        self.eth_in_addr = eth_in_addr     # ETH input address (exchange/bridge)
        self.btc_time = btc_time           # Bitcoin transaction timestamp
        self.eth_time = eth_time           # Ethereum transaction timestamp
        self.label = label                 # Human-readable label for the hop

    def to_dict(self):
        """Convert event attributes to dictionary format for serialization"""
        return {
            "btc_txid": self.btc_txid,
            "btc_out_addr": self.btc_out_addr,
            "eth_in_addr": self.eth_in_addr,
            "btc_time": self.btc_time.isoformat() if hasattr(self.btc_time, 'isoformat') else str(self.btc_time),
            "eth_time": self.eth_time.isoformat() if hasattr(self.eth_time, 'isoformat') else str(self.eth_time),
            "label": self.label
        }

def load_known_entities(path="known_crosschain_entities.json"):
    """
    Load known bridge/exchange entity addresses from JSON file.
    Returns default hardcoded entities if file not found.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default known exchange/bridge addresses
        return {
            "btc": {
                "1DEP8i3QJCsomS4BSMY2RpU1upv62aGvhD": "Binance BTC Deposit",
                "1FfmbHfnpaZjKFvyi1okTjJJusN455paPH": "Coinbase BTC Deposit"
            },
            "eth": {
                "0x742d35Cc6634C0532925a3b844Bc454e4438f44e": "Binance ETH Hot Wallet",
                "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": "Coinbase ETH Hot Wallet"
            }
        }

def detect_chain_hop(btc_txs, eth_txs, known_entities, time_window_hours=24):
    """
    Detect potential chain hopping events by correlating BTC and ETH transactions.
    
    Args:
        btc_txs: List of Bitcoin transactions
        eth_txs: List of Ethereum transactions  
        known_entities: Dictionary of known exchange/bridge addresses
        time_window_hours: Maximum time difference for correlation (default 24 hours)
    
    Returns:
        List of ChainHopEvent objects representing detected chain hops
    """
    events = []
    btc_entities = known_entities.get("btc", {})  # Known BTC exchange addresses
    eth_entities = known_entities.get("eth", {})  # Known ETH exchange addresses

    # Iterate through all Bitcoin transactions
    for btc in btc_txs:
        # Skip transactions without outputs attribute
        if not hasattr(btc, 'outputs'):
            continue
            
        btc_time = btc.timestamp
        
        # Parse timestamp from string if needed
        if isinstance(btc_time, str):
            try:
                btc_time = dateutil.parser.parse(btc_time)
            except:
                pass
        elif isinstance(btc_time, datetime):
            pass
        else:
            continue
            
        # Check each output of the BTC transaction
        for out in btc.outputs:
            # Handle both dictionary and string formats for addresses
            if isinstance(out, dict):
                btc_out_addr = out.get('address') or (out.get('addresses', [None])[0] if out.get('addresses') else None)
            else:
                btc_out_addr = out
            
            # If BTC output goes to a known exchange/bridge address
            if btc_out_addr in btc_entities:
                # Check all Ethereum transactions for correlation
                for eth in eth_txs:
                    # Skip ETH transactions without inputs
                    if not hasattr(eth, 'inputs'):
                        continue
                        
                    eth_time = eth.timestamp
                    
                    # Parse ETH timestamp from string if needed
                    if isinstance(eth_time, str):
                        try:
                            eth_time = dateutil.parser.parse(eth_time)
                        except:
                            pass
                    elif isinstance(eth_time, datetime):
                        pass
                    else:
                        continue
                        
                    # Check each input of the ETH transaction
                    for eth_in in eth.inputs:
                        # Handle both dictionary and string formats for addresses
                        if isinstance(eth_in, dict):
                            eth_in_addr = eth_in.get('address') or (eth_in.get('addresses', [None])[0] if eth_in.get('addresses') else None)
                        else:
                            eth_in_addr = eth_in
                            
                        # If ETH input comes from a known exchange/bridge address
                        if eth_in_addr in eth_entities:
                            try:
                                # Calculate time difference between BTC and ETH transactions
                                diff = abs((eth_time - btc_time).total_seconds())
                                
                                # If within time window, record as potential chain hop
                                if diff <= time_window_hours * 3600:
                                    events.append(ChainHopEvent(
                                        btc_txid=btc.txid,
                                        btc_out_addr=btc_out_addr,
                                        eth_in_addr=eth_in_addr,
                                        btc_time=btc_time,
                                        eth_time=eth_time,
                                        label=f"{btc_entities[btc_out_addr]} → {eth_entities[eth_in_addr]}"
                                    ))
                            except:
                                # Skip if any error occurs during processing
                                pass
    return events

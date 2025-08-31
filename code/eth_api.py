import requests
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import time

class EthereumTransaction:
    """
    Represents an Ethereum transaction in a format compatible with Bitcoin transaction structure
    for cross-chain analysis in the CryptoSherlock system
    """
    def __init__(self, txid: str, from_address: str, to_address: str, value: float,
                 timestamp: datetime, classification: str = "unknown",
                 inputs: List[str] = None, outputs: List[str] = None):
        self.txid = txid                            # Transaction hash/ID
        self.hash = txid                           # Alias for transaction ID
        self.from_addr = from_address              # Sender address (alternative name)
        self.from_address = from_address           # Sender address
        self.to_addr = to_address                  # Receiver address (alternative name)
        self.to_address = to_address               # Receiver address
        self.value = value                         # Transaction value in Ether
        # Convert datetime to ISO string format if needed
        self.timestamp = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        self.classification = classification       # Transaction classification (licit/illicit/unknown)
        self.inputs = inputs or []                 # List of input addresses
        self.outputs = outputs or []               # List of output addresses
        self.meta = {"source": "etherscan", "chain": "ethereum"}  # Metadata for source tracking

def fetch_ethereum_transactions(address: str, limit: int = 50, api_key: str = "YourApiKeyToken") -> List[EthereumTransaction]:
    """
    Fetch Ethereum transactions for a given address using the Etherscan API
    
    Args:
        address: Ethereum address to fetch transactions for
        limit: Maximum number of transactions to fetch
        api_key: Etherscan API key for authentication
    
    Returns:
        List of EthereumTransaction objects
    """
    try:
        # Etherscan API endpoint
        url = f"https://api.etherscan.io/api"
        
        # API parameters for fetching account transactions
        params = {
            "module": "account",        # Account module
            "action": "txlist",         # List transactions action
            "address": address,         # Target address
            "startblock": 0,           # Start from genesis block
            "endblock": 99999999,      # End at latest block
            "page": 1,                 # Page number
            "offset": limit,           # Number of transactions per page
            "sort": "desc",            # Sort by newest first
            "apikey": api_key          # API key
        }
        
        # Make API request with timeout
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Etherscan API error: {response.status_code}")

        # Parse JSON response
        data = response.json()
        if data.get("status") != "1":
            print(f"Etherscan API warning: {data.get('message')}")
            return []

        transactions = []
        # Process each transaction from API response
        for tx in data.get("result", [])[:limit]:
            # Convert wei to ether (1 ether = 10^18 wei)
            value_eth = float(tx.get("value", 0)) / 1e18
            
            # Convert Unix timestamp to datetime object
            timestamp = datetime.fromtimestamp(int(tx.get("timeStamp", 0)))
            
            # Create EthereumTransaction object
            eth_tx = EthereumTransaction(
                txid=tx.get("hash"),
                from_address=tx.get("from"),
                to_address=tx.get("to"),
                value=value_eth,
                timestamp=timestamp,
                classification="unknown",
                inputs=[tx.get("from")],      # Ethereum typically has single input
                outputs=[tx.get("to")]        # Ethereum typically has single output
            )
            transactions.append(eth_tx)
        
        print(f"Loaded {len(transactions)} Ethereum transactions")
        return transactions

    except Exception as e:
        print(f"Error fetching Ethereum transactions: {e}")
        return []

def generate_synthetic_ethereum_transactions() -> List[EthereumTransaction]:
    """
    Generate synthetic Ethereum transactions for demo/testing purposes
    Creates transactions involving known exchange addresses for chain hopping detection
    
    Returns:
        List of synthetic EthereumTransaction objects
    """
    transactions = []
    base_time = datetime.now()
    
    # Known Ethereum exchange addresses for chain hopping analysis
    known_exchanges = [
        "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Binance hot wallet
        "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0",  # Coinbase hot wallet
    ]
    
    # Generate 10 synthetic transactions
    for i in range(10):
        tx = EthereumTransaction(
            txid=f"0x{i:064x}",                    # Generate hex transaction ID
            from_address=known_exchanges[i % 2],    # Alternate between exchanges
            to_address=f"0x{'a' * 40}",            # Dummy recipient address
            value=0.5 + i * 0.1,                   # Incrementing transaction values
            timestamp=base_time,                   # Current timestamp
            classification="exchange",             # Mark as exchange transaction
            inputs=[known_exchanges[i % 2]],       # Input from exchange
            outputs=[f"0x{'a' * 40}"]             # Output to dummy address
        )
        transactions.append(tx)
    
    return transactions

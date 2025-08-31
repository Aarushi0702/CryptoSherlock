import requests
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import time

class LiveTransaction:
    """
    Represents a live Bitcoin transaction fetched from Blockstream API
    Compatible with existing transaction data structures in the system
    """
    def __init__(self, txid: str, from_address: str, to_address: str, value: float,
                 timestamp: datetime, classification: str = "unknown",
                 inputs: List[str] = None, outputs: List[str] = None):
        self.txid = txid                          # Transaction ID/hash
        self.hash = txid                         # Alias for transaction ID
        self.from_addr = from_address            # Sender address (alternative name)
        self.from_address = from_address         # Sender address
        self.to_addr = to_address                # Receiver address (alternative name)
        self.to_address = to_address             # Receiver address
        self.value = value                       # Transaction value in BTC
        # Convert datetime to ISO string format for consistency
        self.timestamp = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        self.classification = classification     # Transaction classification
        self.inputs = inputs or []               # List of input addresses
        self.outputs = outputs or []             # List of output addresses
        self.meta = {"source": "blockstream", "live_data": True}  # Metadata for tracking

    def to_dict(self):
        """Convert transaction object to dictionary format for serialization"""
        return {
            "txid": self.txid,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "value": self.value,
            "timestamp": self.timestamp,
            "classification": self.classification,
            "inputs": self.inputs,
            "outputs": self.outputs
        }

def fetch_address_transactions(address: str, limit: int = 50) -> List[LiveTransaction]:
    """
    Fetch recent transactions for a Bitcoin address using free Blockstream API
    
    Args:
        address: Bitcoin address to fetch transactions for
        limit: Maximum number of transactions to fetch
    
    Returns:
        List of LiveTransaction objects
    """
    try:
        print(f"Fetching live data for address: {address}")
        # Blockstream API endpoint for address transactions
        url = f"https://blockstream.info/api/address/{address}/txs"
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            raise RuntimeError(f"Blockstream API error: {response.status_code} - {response.text}")
        
        data = response.json()
        transactions = []
        
        # Process each transaction from the API response
        for tx_data in data[:limit]:
            tx_hash = tx_data.get('txid')
            if not tx_hash:
                continue
            
            # Fetch detailed transaction information
            tx_detail = fetch_transaction_detail(tx_hash)
            if tx_detail:
                # Convert to LiveTransaction objects
                live_txs = convert_tx_detail_to_transactions(tx_detail, address)
                transactions.extend(live_txs)
            
            # Rate limiting to be respectful to free API
            time.sleep(0.1)
        
        print(f"✅ Successfully loaded {len(transactions)} transactions from Blockstream API")
        return transactions[:limit]
        
    except Exception as e:
        print(f"Error fetching transactions for {address}: {e}")
        return []

def fetch_transaction_detail(tx_hash: str) -> Optional[Dict]:
    """
    Fetch detailed transaction information from Blockstream API
    
    Args:
        tx_hash: Transaction hash to fetch details for
    
    Returns:
        Dictionary containing transaction details or None if failed
    """
    try:
        url = f"https://blockstream.info/api/tx/{tx_hash}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch tx {tx_hash}: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching transaction detail for {tx_hash}: {e}")
        return None

def convert_tx_detail_to_transactions(tx_detail: Dict, focus_address: str) -> List[LiveTransaction]:
    """
    Convert Blockstream API transaction details into LiveTransaction objects
    
    Args:
        tx_detail: Raw transaction data from Blockstream API
        focus_address: The address we're analyzing (used for filtering relevant flows)
    
    Returns:
        List of LiveTransaction objects representing the transaction flows
    """
    transactions = []
    try:
        tx_hash = tx_detail.get("txid")
        
        # Extract timestamp from block confirmation or use current time
        block_time = tx_detail.get('status', {}).get('block_time')
        if block_time:
            timestamp = datetime.fromtimestamp(block_time)
        else:
            timestamp = datetime.now()  # Unconfirmed transaction

        # Extract input addresses from transaction inputs
        input_addresses = []
        for vin in tx_detail.get('vin', []):
            if 'prevout' in vin and 'scriptpubkey_address' in vin['prevout']:
                input_addresses.append(vin['prevout']['scriptpubkey_address'])

        # Extract output addresses and their values
        output_addresses = []
        output_values = []
        for vout in tx_detail.get('vout', []):
            if 'scriptpubkey_address' in vout:
                output_addresses.append(vout['scriptpubkey_address'])
                output_values.append(vout.get('value', 0))  # Value in satoshis

        created_transactions = False
        
        # Create LiveTransaction objects for each relevant input-output combination
        for i, input_addr in enumerate(input_addresses):
            for j, output_addr in enumerate(output_addresses):
                # Only create transactions involving the focus address
                if input_addr != output_addr and (input_addr == focus_address or output_addr == focus_address):
                    # Convert satoshis to BTC
                    value_btc = output_values[j] / 1e8 if j < len(output_values) else 0.0
                    
                    # Classify transaction based on value and addresses
                    classification = classify_transaction(input_addr, output_addr, value_btc)
                    
                    tx = LiveTransaction(
                        txid=tx_hash,
                        from_address=input_addr,
                        to_address=output_addr,
                        value=value_btc,
                        timestamp=timestamp,
                        classification=classification,
                        inputs=input_addresses,
                        outputs=output_addresses
                    )
                    transactions.append(tx)
                    created_transactions = True

        # If no specific combinations found, create a summary transaction
        if not created_transactions and (input_addresses or output_addresses):
            total_value = sum(output_values) / 1e8  # Convert to BTC
            from_addr = input_addresses[0] if input_addresses else "unknown"
            to_addr = output_addresses[0] if output_addresses else "unknown"
            
            tx = LiveTransaction(
                txid=tx_hash,
                from_address=from_addr,
                to_address=to_addr,
                value=total_value,
                timestamp=timestamp,
                classification="unknown",
                inputs=input_addresses,
                outputs=output_addresses
            )
            transactions.append(tx)

    except Exception as e:
        print(f"Error converting transaction detail: {e}")

    return transactions

def classify_transaction(from_addr: str, to_addr: str, value: float) -> str:
    """
    Basic classification of transactions based on value patterns
    
    Args:
        from_addr: Source address
        to_addr: Destination address  
        value: Transaction value in BTC
    
    Returns:
        String classification of the transaction
    """
    if value > 10:
        return "high_value"      # Large transactions (>10 BTC)
    elif value < 0.001:
        return "micro_payment"   # Very small transactions (dust/micro-payments)
    elif value > 1:
        return "significant"     # Medium-large transactions (1-10 BTC)
    else:
        return "unknown"         # Default classification

def fetch_multiple_addresses(addresses: List[str], limit_per_address: int = 20) -> List[LiveTransaction]:
    """
    Fetch transactions for multiple Bitcoin addresses in sequence
    
    Args:
        addresses: List of Bitcoin addresses to fetch data for
        limit_per_address: Maximum transactions to fetch per address
    
    Returns:
        Combined list of LiveTransaction objects from all addresses
    """
    all_transactions = []
    
    for address in addresses:
        try:
            print(f"Fetching data for address: {address}")
            txs = fetch_address_transactions(address, limit_per_address)
            all_transactions.extend(txs)
            
            # Rate limiting between address requests
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error fetching data for {address}: {e}")
            continue
    
    print(f"Total transactions loaded from {len(addresses)} addresses: {len(all_transactions)}")
    return all_transactions

# Test functionality with known Bitcoin addresses
if __name__ == "__main__":
    # Test with a known Binance deposit address
    test_address = "1DEP8i3QJCsomS4BSMY2RpU1upv62aGvhD"
    print(f"Testing with address: {test_address}")
    
    transactions = fetch_address_transactions(test_address, limit=5)
    print(f"Found {len(transactions)} transactions")
    
    # Display transaction details
    for tx in transactions:
        print(f"TX: {tx.txid[:16]}... | {tx.from_address[:16]}... → {tx.to_address[:16]}... | {tx.value:.6f} BTC | {tx.classification}")
    
    print("\n" + "="*50)
    print("Testing multiple addresses...")
    
    # Test with multiple addresses including the Genesis block address
    test_addresses = [
        "1DEP8i3QJCsomS4BSMY2RpU1upv62aGvhD",  # Binance deposit address
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"   # Genesis block address (Satoshi's)
    ]
    
    multi_transactions = fetch_multiple_addresses(test_addresses, limit_per_address=3)
    print(f"Total transactions from multiple addresses: {len(multi_transactions)}")

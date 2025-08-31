import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os

# Try to import live API functionality, gracefully handle if not available
try:
    from live_api import fetch_address_transactions, fetch_multiple_addresses, LiveTransaction
    LIVE_API_AVAILABLE = True
except ImportError:
    LIVE_API_AVAILABLE = False

class Transaction:
    """Represents a cryptocurrency transaction with common attributes"""
    def __init__(self, txid: str, from_address: str, to_address: str, value: float,
                 timestamp: str, classification: str = "unknown", inputs: List = None, outputs: List = None):
        self.txid = txid                           # Transaction ID
        self.hash = txid                           # Alternative reference to transaction ID
        self.from_addr = from_address              # Source address (alternative name)
        self.from_address = from_address           # Source address
        self.to_addr = to_address                  # Destination address (alternative name)
        self.to_address = to_address               # Destination address
        self.value = value                         # Transaction amount
        self.timestamp = timestamp                 # When transaction occurred
        self.classification = classification       # Known classification (licit/illicit/unknown)
        self.inputs = inputs or []                 # List of input addresses or data
        self.outputs = outputs or []               # List of output addresses or data
        self.meta = {}                            # Metadata dictionary for additional pattern data

class DataLoader:
    """Handles loading datasets from multiple sources with caching capabilities"""
    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode                 # Whether to operate in demo mode
        self.cache = {}                           # Cache to store previously loaded datasets

    def load_dataset(self, source: str = "synthetic", filepath: str = None,
                     address: str = None, limit: int = 50) -> List[Transaction]:
        """
        Main interface to load different dataset types
        
        Args:
            source: Type of data source ("synthetic", "live_api", "live_multi", "file")
            filepath: Path to CSV file (for file source)
            address: Cryptocurrency address(es) to fetch (for live API sources)
            limit: Maximum number of transactions to fetch
        
        Returns:
            List of Transaction objects
        """
        if source == "live_api" and address and LIVE_API_AVAILABLE:
            return self.load_from_live_api(address, limit)
        elif source == "live_multi" and address and LIVE_API_AVAILABLE:
            # Parse comma-separated addresses
            addresses = [addr.strip() for addr in address.split(',')]
            return self.load_from_multiple_addresses(addresses, limit)
        elif source == "file" and filepath:
            return self.load_from_file(filepath)
        else:
            return self.load_synthetic_data()

    def load_from_live_api(self, address: str, limit: int = 50) -> List[Transaction]:
        """Load transactions from a live API for a single address"""
        try:
            # Check cache first to avoid redundant API calls
            cache_key = f"live_{address}_{limit}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            print(f"Fetching live data for address: {address}")
            live_transactions = fetch_address_transactions(address, limit)

            # Convert LiveTransaction objects to our Transaction format
            transactions = []
            for live_tx in live_transactions:
                tx = Transaction(
                    txid=live_tx.txid,
                    from_address=live_tx.from_address,
                    to_address=live_tx.to_address,
                    value=live_tx.value,
                    timestamp=live_tx.timestamp,
                    classification=live_tx.classification,
                    inputs=live_tx.inputs,
                    outputs=live_tx.outputs
                )
                tx.meta = live_tx.meta
                transactions.append(tx)

            # Cache results for future requests
            self.cache[cache_key] = transactions
            print(f"Loaded {len(transactions)} live transactions")
            return transactions
        except Exception as e:
            print(f"Error loading from live API: {e}")
            print("Falling back to synthetic data...")
            return self.load_synthetic_data()

    def load_from_multiple_addresses(self, addresses: List[str], limit_per_address: int = 20) -> List[Transaction]:
        """Load transactions for multiple addresses from a live API"""
        try:
            print(f"Fetching data for {len(addresses)} addresses")
            live_transactions = fetch_multiple_addresses(addresses, limit_per_address)

            # Convert all LiveTransaction objects to Transaction format
            transactions = []
            for live_tx in live_transactions:
                tx = Transaction(
                    txid=live_tx.txid,
                    from_address=live_tx.from_address,
                    to_address=live_tx.to_address,
                    value=live_tx.value,
                    timestamp=live_tx.timestamp,
                    classification=live_tx.classification,
                    inputs=live_tx.inputs,
                    outputs=live_tx.outputs
                )
                tx.meta = live_tx.meta
                transactions.append(tx)

            print(f"Loaded {len(transactions)} transactions from multiple addresses")
            return transactions
        except Exception as e:
            print(f"Error loading from multiple addresses: {e}")
            return self.load_synthetic_data()

    def load_from_file(self, filepath: str) -> List[Transaction]:
        """Load transactions from a CSV file; expects JSON strings in inputs, outputs, and meta columns"""
        try:
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}, using synthetic data")
                return self.load_synthetic_data()

            df = pd.read_csv(filepath)
            transactions = []

            for _, row in df.iterrows():
                inputs = []
                outputs = []
                
                # Parse JSON strings from CSV columns
                try:
                    if pd.notna(row.get('inputs')):
                        inputs = json.loads(row['inputs'])
                except:
                    inputs = []
                    
                try:
                    if pd.notna(row.get('outputs')):
                        outputs = json.loads(row['outputs'])
                except:
                    outputs = []

                tx = Transaction(
                    txid=row.get('txid', f"synthetic_{len(transactions)}"),
                    from_address=row.get('from_address', 'unknown'),
                    to_address=row.get('to_address', 'unknown'),
                    value=float(row.get('value', 0)),
                    timestamp=row.get('timestamp', datetime.now().isoformat()),
                    classification=row.get('classification', 'unknown'),
                    inputs=inputs,
                    outputs=outputs
                )
                
                # Parse meta field from JSON string
                try:
                    if pd.notna(row.get('meta')):
                        tx.meta = json.loads(row['meta'])
                except:
                    tx.meta = {}
                    
                transactions.append(tx)

            print(f"Loaded {len(transactions)} transactions from file")
            return transactions
        except Exception as e:
            print(f"Error loading from file: {e}")
            return self.load_synthetic_data()

    def generate_crosschain_synthetic_data(self) -> List[Transaction]:
        """Generate synthetic cross-chain transactions interacting with known exchange addresses"""
        transactions = []
        base_time = datetime.now() - timedelta(days=30)

        # Known Bitcoin exchange addresses for chain hopping detection
        exchange_btc_addrs = [
            "1DEP8i3QJCsomS4BSMY2RpU1upv62aGvhD",  # Binance BTC
            "1FfmbHfnpaZjKFvyi1okTjJJusN455paPH",  # Coinbase BTC
        ]

        # Generate transactions sending BTC to exchanges (potential chain hops)
        for i in range(5):
            tx = Transaction(
                txid=f"crosschain_btc_{i}",
                from_address=f"suspect_addr_{i}",
                to_address=exchange_btc_addrs[i % 2],
                value=5.0 + i,
                timestamp=(base_time + timedelta(hours=i*2)).isoformat(),
                classification="suspicious",
                inputs=[{"address": f"suspect_addr_{i}", "amount": 5.0 + i, "script_type": "p2wpkh"}],
                outputs=[{"address": exchange_btc_addrs[i % 2], "amount": 4.8 + i}]
            )
            tx.meta = {
                "source": "synthetic",
                "pattern": "chain_hopping",
                "exchange_route": exchange_btc_addrs[i % 2][:10] + "...",
                "confidence": 0.7
            }
            transactions.append(tx)

        return transactions

    def load_synthetic_data(self) -> List[Transaction]:
        """Generate comprehensive synthetic transactions for various laundering patterns and legitimate activities"""
        transactions = []
        base_time = datetime.now() - timedelta(days=30)

        # === PEEL CHAIN PATTERN ===
        # Sequential transactions with decreasing amounts
        peel_addresses = [f"peel_addr_{i}" for i in range(9)]
        original_amount = 100.0
        current_amount = original_amount
        
        for i in range(len(peel_addresses) - 1):
            fee_amount = current_amount * 0.05      # 5% fee at each hop
            remaining_amount = current_amount - fee_amount
            
            classification = "illicit" if i % 3 == 0 else "unknown"  # Mix classifications
            
            tx = Transaction(
                txid=f"peel_tx_{i:03d}",
                from_address=peel_addresses[i],
                to_address=peel_addresses[i + 1],
                value=current_amount,
                timestamp=(base_time + timedelta(hours=i)).isoformat(),
                classification=classification,
                inputs=[{"address": peel_addresses[i], "amount": current_amount, "script_type": "p2wpkh"}],
                outputs=[{"address": peel_addresses[i + 1], "amount": remaining_amount}]
            )
            
            tx.meta = {
                "source": "synthetic",
                "pattern": "peel_chain",
                "hop_number": i + 1,
                "original_amount": original_amount
            }
            
            transactions.append(tx)
            current_amount = remaining_amount

        # === STRUCTURING PATTERN ===
        # Multiple small deposits below reporting thresholds
        struct_target = "struct_target_001"
        total_donors = 25
        
        for i in range(total_donors):
            amount = np.random.uniform(0.005, 0.009)  # Small amounts under threshold
            output_amount = amount * 0.95             # 5% fee
            
            classification = "illicit" if i % 10 == 0 else "unknown"
            
            tx = Transaction(
                txid=f"struct_tx_{i:03d}",
                from_address=f"donor_{i:03d}",
                to_address=struct_target,
                value=amount,
                timestamp=(base_time + timedelta(hours=2, minutes=i)).isoformat(),
                classification=classification,
                inputs=[{"address": f"donor_{i:03d}", "amount": amount, "script_type": "p2wpkh"}],
                outputs=[{"address": struct_target, "amount": output_amount}]
            )
            
            tx.meta = {
                "source": "synthetic",
                "pattern": "structuring",
                "deposit_number": i + 1,
                "total_donors": total_donors
            }
            
            transactions.append(tx)

        # === COINJOIN PATTERN ===
        # Privacy mixing with equal outputs (hallmark of CoinJoin)
        coinjoin_scenarios = [
            {"participants": 7, "equal_amount": 0.1},
            {"participants": 6, "equal_amount": 0.1},
            {"participants": 5, "equal_amount": 0.1}
        ]
        
        for idx, scenario in enumerate(coinjoin_scenarios):
            participants = scenario["participants"]
            equal_amount = scenario["equal_amount"]
            
            # Create varied input amounts but equal outputs (CoinJoin characteristic)
            inputs = []
            for i in range(participants):
                input_amount = np.random.uniform(0.15, 0.4)
                inputs.append({
                    "address": f"coinjoin_in_{idx}_{i}",
                    "amount": input_amount,
                    "script_type": np.random.choice(["p2wpkh", "p2tr", "p2sh"])
                })
            
            # All outputs are exactly equal (hallmark of CoinJoin)
            outputs = []
            for i in range(participants):
                outputs.append({
                    "address": f"coinjoin_out_{idx}_{i}",
                    "amount": equal_amount
                })
            
            total_input = sum(inp["amount"] for inp in inputs)
            
            tx = Transaction(
                txid=f"coinjoin_tx_{idx:03d}",
                from_address=f"coinjoin_mixer_{idx}",
                to_address=f"coinjoin_out_{idx}_0",
                value=total_input,
                timestamp=(base_time + timedelta(hours=14 + idx * 2)).isoformat(),
                classification="unknown",
                inputs=inputs,
                outputs=outputs
            )
            
            tx.meta = {
                "source": "synthetic",
                "pattern": "coinjoin",
                "equal_outputs": participants,
                "equal_amount": equal_amount
            }
            
            transactions.append(tx)

        # === RAPID MOVEMENT PATTERN ===
        # Fast sequential transactions across addresses
        rapid_addresses = [f"rapid_addr_{i}" for i in range(8)]
        rapid_amount = 50.0
        
        for i in range(len(rapid_addresses) - 1):
            gap_minutes = np.random.randint(20, 45)   # Small random delays (rapid movement)
            fee = rapid_amount * 0.02
            output_amount = rapid_amount - fee
            
            classification = "illicit" if i % 4 == 0 else "unknown"
            
            tx = Transaction(
                txid=f"rapid_tx_{i:03d}",
                from_address=rapid_addresses[i],
                to_address=rapid_addresses[i + 1],
                value=rapid_amount,
                timestamp=(base_time + timedelta(hours=16 + i, minutes=gap_minutes)).isoformat(),
                classification=classification,
                inputs=[{"address": rapid_addresses[i], "amount": rapid_amount, "script_type": "p2wpkh"}],
                outputs=[{"address": rapid_addresses[i + 1], "amount": output_amount}]
            )
            
            tx.meta = {
                "source": "synthetic",
                "pattern": "rapid_movement",
                "gap_minutes": gap_minutes,
                "hop_number": i + 1
            }
            
            transactions.append(tx)
            rapid_amount = output_amount

        # === LAYERING PATTERN ===
        # Complex multi-hop obfuscation with entropy analysis
        layer_source = "layer_source"
        layer_mids = [f"layer_mid_{i}" for i in range(3)]
        initial_split_amount = 200.0 / 3  # Split 200 BTC into 3 parts
        
        # Phase 1: Dispersion
        for i, mid_addr in enumerate(layer_mids):
            classification = "illicit" if i == 0 else "unknown"
            
            tx = Transaction(
                txid=f"layer_disperse_{i:03d}",
                from_address=layer_source,
                to_address=mid_addr,
                value=initial_split_amount,
                timestamp=(base_time + timedelta(hours=18 + i)).isoformat(),
                classification=classification,
                inputs=[{"address": layer_source, "amount": initial_split_amount, "script_type": "p2wpkh"}],
                outputs=[{"address": mid_addr, "amount": initial_split_amount * 0.95}]
            )
            
            tx.meta = {
                "source": "synthetic",
                "pattern": "layering",
                "phase": "dispersion",
                "entropy_level": i + 1
            }
            
            transactions.append(tx)

        # Phase 2: Spreading (each mid address spreads to 3 addresses)
        for mid_idx, mid_addr in enumerate(layer_mids):
            for spread_idx in range(3):
                spread_addr = f"layer_spread_{mid_idx}_{spread_idx}"
                spread_amount = (initial_split_amount * 0.95) / 3
                
                tx = Transaction(
                    txid=f"layer_spread_{mid_idx}_{spread_idx:03d}",
                    from_address=mid_addr,
                    to_address=spread_addr,
                    value=spread_amount,
                    timestamp=(base_time + timedelta(hours=20 + mid_idx, minutes=spread_idx * 6)).isoformat(),
                    classification="unknown",
                    inputs=[{"address": mid_addr, "amount": spread_amount, "script_type": "p2wpkh"}],
                    outputs=[{"address": spread_addr, "amount": spread_amount * 0.95}]
                )
                
                tx.meta = {
                    "source": "synthetic",
                    "pattern": "layering",
                    "phase": "spreading",
                    "entropy_level": 9  # High entropy after multiple hops
                }
                
                transactions.append(tx)

        # === TERRORIST FINANCING CROWDFUNDING ===
        # Micro-donations that could fund terrorist activities
        tf_target = "tf_target_001"
        total_tf_donors = 30
        
        for i in range(total_tf_donors):
            donation_amount = np.random.uniform(0.003, 0.009)
            output_amount = donation_amount * 0.98  # Small fee
            
            classification = "illicit" if i % 15 == 0 else "unknown"
            
            tx = Transaction(
                txid=f"tf_donation_{i:03d}",
                from_address=f"tf_donor_{i:03d}",
                to_address=tf_target,
                value=donation_amount,
                timestamp=(base_time + timedelta(hours=20, minutes=i * 2)).isoformat(),
                classification=classification,
                inputs=[{"address": f"tf_donor_{i:03d}", "amount": donation_amount, "script_type": "p2wpkh"}],
                outputs=[{"address": tf_target, "amount": output_amount}]
            )
            
            tx.meta = {
                "source": "synthetic",
                "pattern": "tf_crowdfunding",
                "donation_number": i + 1,
                "total_donors": total_tf_donors
            }
            
            transactions.append(tx)

        # === LEGITIMATE TRANSACTIONS ===
        
        # E-commerce transactions
        for i in range(10):
            purchase_amount = np.random.uniform(0.4, 2.5)
            merchant_receives = purchase_amount * 0.97  # 3% processing fee
            
            tx = Transaction(
                txid=f"legit_ecom_{i:03d}",
                from_address=f"customer_{i:03d}",
                to_address=f"merchant_{i % 3:03d}",     # 3 merchants
                value=purchase_amount,
                timestamp=(base_time + timedelta(hours=9 + i * 2)).isoformat(),
                classification="licit",
                inputs=[{"address": f"customer_{i:03d}", "amount": purchase_amount, "script_type": "p2wpkh"}],
                outputs=[{"address": f"merchant_{i % 3:03d}", "amount": merchant_receives}]
            )
            
            tx.meta = {
                "source": "synthetic",
                "pattern": "legitimate",
                "transaction_type": "e-commerce"
            }
            
            transactions.append(tx)

        # Exchange deposits
        for i in range(10):
            deposit_amount = np.random.uniform(7, 48)
            exchange_receives = deposit_amount * 0.999  # Minimal fee
            
            tx = Transaction(
                txid=f"legit_exchange_{i:03d}",
                from_address=f"user_wallet_{i:03d}",
                to_address="exchange_hot_wallet",
                value=deposit_amount,
                timestamp=(base_time + timedelta(hours=8 + i * 1.5)).isoformat(),
                classification="licit",
                inputs=[{"address": f"user_wallet_{i:03d}", "amount": deposit_amount, "script_type": "p2wpkh"}],
                outputs=[{"address": "exchange_hot_wallet", "amount": exchange_receives}]
            )
            
            tx.meta = {
                "source": "synthetic",
                "pattern": "legitimate",
                "transaction_type": "exchange_deposit"
            }
            
            transactions.append(tx)

        # Add cross-chain synthetic data
        crosschain_transactions = self.generate_crosschain_synthetic_data()
        transactions.extend(crosschain_transactions)

        print(f"Generated {len(transactions)} high-quality synthetic transactions with rich metadata")
        return transactions

    def clear_cache(self):
        """Clear the transaction cache"""
        self.cache.clear()
        print("Cache cleared")

    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        return {
            "cached_queries": len(self.cache),
            "cache_keys": list(self.cache.keys())
        }

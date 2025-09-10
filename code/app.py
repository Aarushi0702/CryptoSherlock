import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from data_io import DataLoader, Transaction
from graph_viz import GraphBuilder, InteractiveGraphVisualizer
from detectors import PatternDetector
from scoring import RiskScorer
from reporting import SARGenerator, PatternAssetExporter
from monitor import WatchlistMonitor
from clustering import EntityClusterer
import tempfile
import os

# Import chain hopping modules
try:
    from chain_hop import detect_chain_hop, load_known_entities, ChainHopEvent
    from eth_api import fetch_ethereum_transactions, generate_synthetic_ethereum_transactions
    CHAIN_HOP_AVAILABLE = True
except ImportError:
    CHAIN_HOP_AVAILABLE = False

def filter_graph_by_time(graph, start_dt, end_dt):
    import networkx as nx
    import pandas as pd
    Gf = nx.DiGraph()
    for n, d in graph.nodes(data=True):
        Gf.add_node(n, **d)
    s = pd.to_datetime(start_dt, utc=True) if start_dt else None
    e = pd.to_datetime(end_dt, utc=True) if end_dt else None
    for u, v, ed in graph.edges(data=True):
        ts = ed.get('timestamp')
        if not ts:
            continue
        t = pd.to_datetime(ts, errors='coerce', utc=True)
        if pd.isna(t):
            continue
        if (s is None or t >= s) and (e is None or t <= e):
            Gf.add_edge(u, v, **ed)
    return Gf

def compute_cluster_decision(graph, member_nodes):
    patterns = ['peel_chain', 'structuring', 'rapid_movement', 'layering', 'tf_crowdfunding']
    if CHAIN_HOP_AVAILABLE:
        patterns.append('chain_hopping')
    aggregated = {p: {'pattern': p, 'confidence': 0.0, 'explanation': '', 'evidence': {}} for p in patterns}
    for node in member_nodes:
        try:
            res = st.session_state.detector.detect_peel_chain(graph, node)
            if res.get('confidence', 0) > aggregated['peel_chain']['confidence']:
                aggregated['peel_chain'] = res
        except Exception:
            pass
        try:
            res = st.session_state.detector.detect_structuring(graph, node)
            if res.get('confidence', 0) > aggregated['structuring']['confidence']:
                aggregated['structuring'] = res
        except Exception:
            pass
        try:
            res = st.session_state.detector.detect_rapid_movement(graph, node)
            if res.get('confidence', 0) > aggregated['rapid_movement']['confidence']:
                aggregated['rapid_movement'] = res
        except Exception:
            pass
        try:
            res = st.session_state.detector.detect_layering(graph, node)
            if res.get('confidence', 0) > aggregated['layering']['confidence']:
                aggregated['layering'] = res
        except Exception:
            pass
        try:
            res = st.session_state.detector.detect_tf_crowdfunding(graph, node)
            if res.get('confidence', 0) > aggregated['tf_crowdfunding']['confidence']:
                aggregated['tf_crowdfunding'] = res
        except Exception:
            pass
        if CHAIN_HOP_AVAILABLE:
            try:
                res = st.session_state.detector.detect_chain_hopping(
                    st.session_state.get('chain_hop_events', []), node
                )
                if res.get('confidence', 0) > aggregated['chain_hopping']['confidence']:
                    aggregated['chain_hopping'] = res
            except Exception:
                pass
    score, level, summary = st.session_state.scorer.compute_risk_score(aggregated)
    return score, level, summary, aggregated

def export_pattern_assets(subgraph, fig, pattern_name, target_address, detection_result):
    risk_score, risk_level, _ = st.session_state.scorer.compute_risk_score({pattern_name: detection_result})
    graph_json = st.session_state.asset_exporter.build_graph_json(subgraph)
    pattern_report = st.session_state.asset_exporter.generate_pattern_report(
        pattern_name, target_address, risk_score, risk_level, detection_result
    )
    st.download_button(
        label="📈 Export Pattern Graph (JSON)",
        data=graph_json,
        file_name=f"{pattern_name}_graph_{target_address}.json",
        mime="application/json"
    )
    bundle_bytes = st.session_state.asset_exporter.build_zip_bundle(graph_json, pattern_report, fig)
    st.download_button(
        label="📦 Download Pattern Bundle (Graph + Explanation)",
        data=bundle_bytes,
        file_name=f"{pattern_name}_bundle_{target_address}.zip",
        mime="application/zip"
    )

def display_alerts():
    """Display alerts for illegal transactions on the dashboard"""
    if not st.session_state.graph:
        return
    
    # Check for illegal transaction alerts
    illegal_alerts = st.session_state.monitor.check_illegal_transactions_alerts(
        st.session_state.graph, 
        st.session_state.detector,
        st.session_state.scorer
    )
    
    if illegal_alerts:
        st.error("🚨 **SECURITY ALERTS**")
        
        # Create alert container
        alert_container = st.container()
        with alert_container:
            cols = st.columns([3, 1])
            
            with cols[0]:
                st.subheader(f"⚠️ {len(illegal_alerts)} Illegal Transaction(s) Detected")
                
                for alert in illegal_alerts:
                    with st.expander(f"🔴 {alert['message']}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Risk Level", alert['risk_level'])
                        with col2:
                            st.metric("Risk Score", f"{alert['risk_score']:.3f}")
                        with col3:
                            st.metric("Address", alert['address'][:12] + "...")
                        
                        if alert['patterns']:
                            st.write("**Detected Patterns:**")
                            for pattern in alert['patterns']:
                                st.write(f"• {pattern.replace('_', ' ').title()}")
                        
                        st.write(f"**Detected at:** {alert['timestamp']}")
            
            with cols[1]:
                st.metric("Total Alerts", len(illegal_alerts))
                if st.button("🔕 Acknowledge Alerts"):
                    st.success("Alerts acknowledged!")
    
    # Also check for recent watchlist alerts
    recent_alerts = st.session_state.monitor.get_recent_alerts(hours=24)
    high_risk_alerts = [alert for alert in recent_alerts if alert.get('risk_level') == 'HIGH']
    
    if high_risk_alerts:
        st.warning(f"⚠️ {len(high_risk_alerts)} High-Risk Watchlist Alert(s) in Last 24h")
        with st.expander("View High-Risk Alerts"):
            for alert in high_risk_alerts[-3:]:  # Show last 3
                st.write(f"• **{alert['address'][:16]}...** - Risk: {alert['risk_level']} ({alert['risk_score']:.3f})")

def display_graph_statistics():
    """Display graph statistics in the sidebar"""
    if not st.session_state.graph:
        return
    
    st.subheader("📊 Graph Statistics")
    num_nodes = st.session_state.graph.number_of_nodes()
    num_edges = st.session_state.graph.number_of_edges()
    
    st.metric("Addresses", num_nodes)
    st.metric("Transactions", num_edges)
    
    # Classification breakdown
    classifications = {}
    for node, data in st.session_state.graph.nodes(data=True):
        classification = data.get('classification', 'unknown')
        classifications[classification] = classifications.get(classification, 0) + 1
    
    st.write("**Classification Breakdown:**")
    for classification, count in classifications.items():
        st.write(f"• {classification.title()}: {count}")

def main():
    st.set_page_config(
        page_title="CryptoSherlock X - Live Edition",
        page_icon="🔍",
        layout="wide"
    )

    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader(demo_mode=True)
        st.session_state.graph_builder = GraphBuilder()
        st.session_state.visualizer = InteractiveGraphVisualizer()
        st.session_state.detector = PatternDetector()
        st.session_state.scorer = RiskScorer()
        st.session_state.sar_generator = SARGenerator()
        st.session_state.asset_exporter = PatternAssetExporter()
        st.session_state.monitor = WatchlistMonitor()
        st.session_state.clusterer = EntityClusterer()
        st.session_state.transactions = []
        st.session_state.eth_transactions = []
        st.session_state.chain_hop_events = []
        st.session_state.graph = None
        st.session_state.cluster_map = {}
        st.session_state.cluster_graph = None
        st.session_state.view_enable_cluster = False
        st.session_state.view_time_range = None
        st.session_state.data_source = "synthetic"

    st.title("🔍 CryptoSherlock X - Live Edition")
    st.markdown("*Explainable Cryptocurrency Forensics with Live Blockchain Data & Cross-Chain Analysis*")

    # Enhanced sidebar with chain hopping
    with st.sidebar:
        st.markdown("### 📊 Data Source")
        data_source = st.selectbox(
            "Choose data source:",
            ["synthetic", "live_api", "live_multi", "file"],
            format_func=lambda x: {
                "synthetic": "🎭 Synthetic Demo Data",
                "live_api": "🌐 Live Bitcoin API (Single Address)",
                "live_multi": "🌐 Live Bitcoin API (Multiple Addresses)",
                "file": "📁 Upload CSV File"
            }[x]
        )

        # Chain hopping data section
        if CHAIN_HOP_AVAILABLE:
            st.markdown("### ⛓️ Chain Hopping Detection")
            enable_chain_hop = st.checkbox("🔗 Enable Cross-Chain Analysis", value=True)
            if enable_chain_hop:
                eth_source = st.selectbox(
                    "Ethereum data source:",
                    ["synthetic", "live_etherscan"],
                    format_func=lambda x: {
                        "synthetic": "🎭 Synthetic ETH Data",
                        "live_etherscan": "🌐 Live Etherscan API"
                    }[x]
                )

                if eth_source == "live_etherscan":
                    eth_address = st.text_input(
                        "Ethereum Address:",
                        placeholder="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                        help="Enter Ethereum address for cross-chain analysis"
                    )
                    etherscan_api_key = st.text_input(
                        "Etherscan API Key:",
                        placeholder="YourApiKeyToken",
                        type="password",
                        help="Get free API key from etherscan.io"
                    )
        else:
            st.warning("⚠️ Chain hopping modules not available. Install dependencies.")
            enable_chain_hop = False

        # Data loading section
        if data_source in ["live_api", "live_multi"]:
            if data_source == "live_api":
                address_input = st.text_input(
                    "Bitcoin Address:",
                    placeholder="1DEP8i3QJCsomS4BSMY2RpU1upv62aGvhD",
                    help="Enter a Bitcoin address to analyze"
                )
            else: # live_multi
                address_input = st.text_area(
                    "Bitcoin Addresses (one per line or comma-separated):",
                    placeholder="1DEP8i3QJCsomS4BSMY2RpU1upv62aGvhD\n1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                    help="Enter multiple Bitcoin addresses to analyze"
                )

            transaction_limit = st.slider("Max transactions per address:", 10, 100, 50)
            if st.button("🔄 Fetch Live Data"):
                if address_input:
                    with st.spinner("Fetching live blockchain data..."):
                        try:
                            if data_source == "live_multi":
                                addresses = [addr.strip() for addr in address_input.replace('\n', ',').split(',') if addr.strip()]
                                address_input = ','.join(addresses)
                            st.session_state.transactions = st.session_state.data_loader.load_dataset(
                                source=data_source,
                                address=address_input,
                                limit=transaction_limit
                            )
                            if st.session_state.transactions:
                                st.success(f"✅ Loaded {len(st.session_state.transactions)} live transactions!")
                                st.session_state.data_source = data_source
                            else:
                                st.warning("No transactions found for this address. Using synthetic data.")
                        except Exception as e:
                            st.error(f"Error fetching live data: {e}")
                            st.info("Falling back to synthetic data...")

        elif data_source == "file":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                if st.button("📁 Load File"):
                    with st.spinner("Loading file..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        st.session_state.transactions = st.session_state.data_loader.load_dataset(
                            source="file",
                            filepath=tmp_file_path
                        )
                        st.success(f"✅ Loaded {len(st.session_state.transactions)} transactions from file!")
                        st.session_state.data_source = data_source
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass

        # Load synthetic data by default
        if not st.session_state.transactions or data_source == "synthetic":
            if st.button("🎭 Load Synthetic Demo Data") or not st.session_state.transactions:
                with st.spinner("Generating synthetic dataset..."):
                    st.session_state.transactions = st.session_state.data_loader.load_dataset(source="synthetic")
                    st.session_state.data_source = "synthetic"

        # Load Ethereum transactions for chain hopping
        if CHAIN_HOP_AVAILABLE and 'enable_chain_hop' in locals() and enable_chain_hop:
            if eth_source == "synthetic":
                if st.button("🎭 Load Synthetic ETH Data"):
                    st.session_state.eth_transactions = generate_synthetic_ethereum_transactions()
                    st.success(f"✅ Loaded {len(st.session_state.eth_transactions)} synthetic ETH transactions")
            elif eth_source == "live_etherscan" and 'eth_address' in locals() and eth_address and 'etherscan_api_key' in locals() and etherscan_api_key:
                if st.button("🔄 Fetch Live ETH Data"):
                    with st.spinner("Fetching Ethereum transactions..."):
                        try:
                            st.session_state.eth_transactions = fetch_ethereum_transactions(
                                eth_address, limit=50, api_key=etherscan_api_key
                            )
                            st.success(f"✅ Loaded {len(st.session_state.eth_transactions)} live ETH transactions")
                        except Exception as e:
                            st.error(f"Error fetching ETH data: {e}")

            # Detect chain hopping events
            if st.session_state.transactions and st.session_state.eth_transactions:
                if st.button("🔍 Detect Chain Hopping"):
                    try:
                        known_entities = load_known_entities()
                        st.session_state.chain_hop_events = detect_chain_hop(
                            st.session_state.transactions,
                            st.session_state.eth_transactions,
                            known_entities,
                            time_window_hours=24
                        )
                        if st.session_state.chain_hop_events:
                            st.success(f"⚠️ Detected {len(st.session_state.chain_hop_events)} chain hopping events!")
                        else:
                            st.info("✅ No chain hopping detected")
                    except Exception as e:
                        st.error(f"Chain hopping detection failed: {e}")

        # Cache management
        if st.session_state.data_source in ["live_api", "live_multi"]:
            st.markdown("### 🗂️ Cache Management")
            cache_info = st.session_state.data_loader.get_cache_info()
            st.caption(f"Cached queries: {cache_info['cached_queries']}")
            if st.button("🗑️ Clear Cache"):
                st.session_state.data_loader.clear_cache()
                st.success("Cache cleared!")

        # Data source indicator
        st.markdown("### 📈 Current Data")
        data_source_icons = {
            "synthetic": "🎭",
            "live_api": "🌐",
            "live_multi": "🌐",
            "file": "📁"
        }
        st.info(f"{data_source_icons.get(st.session_state.data_source, '❓')} **{st.session_state.data_source.replace('_', ' ').title()}**")
        if st.session_state.transactions:
            st.metric("Total Transactions", len(st.session_state.transactions))
            if st.session_state.data_source in ["live_api", "live_multi"]:
                live_count = sum(1 for tx in st.session_state.transactions if hasattr(tx, 'meta') and tx.meta.get('live_data'))
                st.metric("Live Transactions", live_count)
        if CHAIN_HOP_AVAILABLE and st.session_state.eth_transactions:
            st.metric("ETH Transactions", len(st.session_state.eth_transactions))
        if CHAIN_HOP_AVAILABLE and st.session_state.chain_hop_events:
            st.metric("Chain Hop Events", len(st.session_state.chain_hop_events))

    # Build graph from loaded transactions
    if st.session_state.transactions:
        st.session_state.graph = st.session_state.graph_builder.build_address_graph(
            st.session_state.transactions
        )

    # Rest of sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ View Options")
        st.session_state.view_enable_cluster = st.checkbox("👥 Entity clustering", value=False)

        # Time range filter
        if st.session_state.transactions:
            all_ts = [tx.timestamp for tx in st.session_state.transactions if tx.timestamp]
            if all_ts:
                try:
                    timestamps = pd.to_datetime(all_ts)
                    min_ts = timestamps.min().to_pydatetime()
                    max_ts = timestamps.max().to_pydatetime()
                    st.session_state.view_time_range = st.slider(
                        "Time window",
                        min_value=min_ts,
                        max_value=max_ts,
                        value=(min_ts, max_ts),
                        format="YYYY-MM-DD HH:mm"
                    )
                except:
                    now = datetime.utcnow()
                    st.session_state.view_time_range = (now - timedelta(days=1), now)

    # Clustering
    if st.session_state.view_enable_cluster and st.session_state.transactions and st.session_state.graph:
        st.session_state.cluster_map = st.session_state.clusterer.build_clusters(st.session_state.transactions)
        st.session_state.cluster_graph = st.session_state.clusterer.collapse_to_cluster_graph(
            st.session_state.graph, st.session_state.cluster_map
        )

    # Enhanced main tabs with Chain Hopping
    tab_names = [
        "🏠 Dashboard",
        "⛓️ Peel Chain",
        "💸 Structuring",
        "🌀 CoinJoin",
        "⚡ Rapid Movement",
        "🕸️ Layering",
        "💰 TF Crowdfunding"
    ]
    if CHAIN_HOP_AVAILABLE:
        tab_names.append("🔗 Chain Hopping")

    tabs = st.tabs(tab_names)

    with tabs[0]: # Dashboard
        display_dashboard()
    with tabs[1]: # Peel Chain
        display_peel_chain_analysis()
    with tabs[2]: # Structuring
        display_structuring_analysis()
    with tabs[3]: # CoinJoin
        display_coinjoin_analysis()
    with tabs[4]: # Rapid Movement
        display_rapid_movement_analysis()
    with tabs[5]: # Layering
        display_layering_analysis()
    with tabs[6]: # TF Crowdfunding
        display_tf_crowdfunding_analysis()
    if CHAIN_HOP_AVAILABLE and len(tabs) > 7:
        with tabs[7]: # Chain Hopping
            display_chain_hopping_analysis()

def display_dashboard():
    if not st.session_state.graph:
        st.warning("No data loaded")
        return

    start_dt, end_dt = st.session_state.view_time_range if st.session_state.view_time_range else (None, None)
    filtered_graph = filter_graph_by_time(st.session_state.graph, start_dt, end_dt)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Complete Transaction Network")
        positions = st.session_state.graph_builder.compute_layout(filtered_graph)
        fig = st.session_state.visualizer.create_plotly_graph(
            filtered_graph, positions, "Complete Transaction Flow Network"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ADD ALERT SYSTEM HERE
        display_alerts()

        if st.session_state.view_enable_cluster and st.session_state.cluster_graph is not None:
            st.subheader("Entity Cluster Network")
            c_filtered = filter_graph_by_time(st.session_state.cluster_graph, start_dt, end_dt)
            c_pos = st.session_state.graph_builder.compute_layout(c_filtered)
            cfig = st.session_state.visualizer.create_plotly_graph(c_filtered, c_pos, "Collapsed Entity Graph")
            st.plotly_chart(cfig, use_container_width=True)

    with col2:
        display_graph_statistics()

        # Chain hopping summary
        if CHAIN_HOP_AVAILABLE and st.session_state.chain_hop_events:
            st.subheader("🔗 Chain Hopping Summary")
            st.metric("Cross-Chain Events", len(st.session_state.chain_hop_events))

        if st.session_state.view_enable_cluster and st.session_state.cluster_graph is not None:
            st.subheader("👥 Cluster Risk Decision")
            seed = st.text_input("Seed address (belongs to a cluster)", value="")
            if seed and seed in st.session_state.cluster_map:
                cid = st.session_state.cluster_map[seed]
                members = {n for n, cid2 in st.session_state.cluster_map.items() if cid2 == cid}
                score, level, summary, aggregated = compute_cluster_decision(st.session_state.graph, members)
                st.markdown(f"### Cluster ID: `{cid}` • Members: {len(members)}")
                st.metric("Cluster Risk Score", f"{score:.3f}")
                st.metric("Cluster Risk Level", level)
                with st.expander("Aggregated Pattern Evidence"):
                    st.json(aggregated)
                sar_report = st.session_state.sar_generator.generate_sar_report(
                    address=f"cluster:{cid}", risk_summary=summary, detection_results=aggregated
                )
                st.download_button(
                    "📄 Download Cluster SAR",
                    data=json.dumps(sar_report, indent=2),
                    file_name=f"sar_cluster_{cid}.json",
                    mime="application/json"
                )

        st.subheader("📊 Export Options")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📈 Export Graph"):
                export_graph_data()
        with col_b:
            if 'analysis_results' in st.session_state:
                if st.button("📄 Export SAR"):
                    generate_and_download_sar()

def display_peel_chain_analysis():
    st.header("⛓️ Peel Chain Money Laundering")
    st.subheader("Pattern Explanation")
    st.write("""
    Peel Chain is a money laundering technique where criminals:
    1) Transfer a large amount to the first address,
    2) Send most funds to the next address (keeping a small "peel"),
    3) Repeat this across multiple hops,
    4) Each hop reduces the amount, making tracking difficult.
    """)

    start_dt, end_dt = st.session_state.view_time_range if st.session_state.view_time_range else (None, None)
    filtered_graph = filter_graph_by_time(st.session_state.graph, start_dt, end_dt)

    # Show transactions involved
    win_start = pd.to_datetime(start_dt, utc=True) if start_dt else None
    win_end = pd.to_datetime(end_dt, utc=True) if end_dt else None
    peel_transactions = []
    for tx in st.session_state.transactions:
        if ('peel_addr' in tx.from_address or 'peel_addr' in tx.to_address):
            tt = pd.to_datetime(tx.timestamp, errors='coerce', utc=True)
            if (not pd.isna(tt)) and (win_start is None or tt >= win_start) and (win_end is None or tt <= win_end):
                peel_transactions.append(tx)

    st.subheader(f"📋 Transactions Involved ({len(peel_transactions)})")
    if peel_transactions:
        st.dataframe(pd.DataFrame([{
            'Transaction ID': tx.txid[:16] + '...',
            'From Address': tx.from_address,
            'To Address': tx.to_address,
            'Amount (BTC)': f"{tx.value:.6f}",
            'Timestamp': tx.timestamp,
            'Classification': tx.classification.upper()
        } for tx in peel_transactions]), use_container_width=True)
    else:
        st.info("No peel chain transactions found in the selected time window")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Pattern Visualization")
        peel_addresses = [addr for addr in filtered_graph.nodes() if addr.startswith('peel_addr_')]
        peel_subgraph = None
        fig = None
        if peel_addresses:
            peel_subgraph = filtered_graph.subgraph(peel_addresses).copy()
            positions = st.session_state.graph_builder.compute_layout(peel_subgraph)
            fig = st.session_state.visualizer.create_plotly_graph(
                peel_subgraph, positions, "Peel Chain Pattern - Static Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Pattern Analysis")
        target_address = "peel_addr_0"
        if target_address in filtered_graph.nodes():
            detection_results = st.session_state.detector.detect_peel_chain(filtered_graph, target_address)
            risk_score, risk_level, _ = st.session_state.scorer.compute_risk_score({'peel_chain': detection_results})
            icon, level_text = {'LOW': ('🟢', 'Low'), 'MEDIUM': ('🟡', 'Medium'), 'HIGH': ('🔴', 'High')}.get(risk_level, ('⚪', 'Unknown'))
            st.markdown(f"### Risk Level: {icon} **{level_text}**")
            st.metric("Risk Score", f"{risk_score:.3f}")
            st.metric("Confidence", f"{detection_results['confidence']:.3f}")
            st.subheader("🔍 Evidence")
            if detection_results.get('evidence'):
                st.json(detection_results['evidence'])
            st.subheader("📄 Explanation")
            st.info(detection_results['explanation'])
            if peel_subgraph is not None and fig is not None:
                export_pattern_assets(peel_subgraph, fig, "peel_chain", target_address, detection_results)
        else:
            st.warning("⚠️ No peel chain addresses found for analysis.")
            st.info("Peel chain patterns require sequential address transfers. Load data containing peel chain transactions to see analysis.")
def display_structuring_analysis():
    st.header("💸 Structuring (Smurfing) Detection")
    
    st.subheader("Pattern Explanation")
    st.write("""
    Structuring (Smurfing) involves:
    1) Breaking large amounts into small transactions,
    2) Using multiple donor accounts to make deposits,
    3) Staying below regulatory reporting thresholds,
    4) Deposits target the same destination address.
    """)

    start_dt, end_dt = st.session_state.view_time_range if st.session_state.view_time_range else (None, None)
    filtered_graph = filter_graph_by_time(st.session_state.graph, start_dt, end_dt)

    # Show transactions involved
    win_start = pd.to_datetime(start_dt, utc=True) if start_dt else None
    win_end = pd.to_datetime(end_dt, utc=True) if end_dt else None
    
    struct_transactions = []
    for tx in st.session_state.transactions:
        if ('donor_' in tx.from_address) or ('struct_target' in tx.to_address):
            tt = pd.to_datetime(tx.timestamp, errors='coerce', utc=True)
            if (not pd.isna(tt)) and (win_start is None or tt >= win_start) and (win_end is None or tt <= win_end):
                struct_transactions.append(tx)

    st.subheader(f"📋 Transactions Involved ({len(struct_transactions)})")
    if struct_transactions:
        st.dataframe(pd.DataFrame([{
            'Transaction ID': tx.txid[:16] + '...',
            'From Address': tx.from_address,
            'To Address': tx.to_address,
            'Amount (BTC)': f"{tx.value:.6f}",
            'Timestamp': tx.timestamp,
            'Classification': tx.classification.upper()
        } for tx in struct_transactions]), use_container_width=True)
    else:
        st.info("No structuring transactions found in the selected time window")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Pattern Visualization")
        struct_addresses = [addr for addr in filtered_graph.nodes() if ('donor_' in addr or 'struct_target' in addr)]
        struct_subgraph = None
        fig = None

        if struct_addresses:
            struct_subgraph = filtered_graph.subgraph(struct_addresses).copy()
            positions = st.session_state.graph_builder.compute_layout(struct_subgraph)
            
            fig = st.session_state.visualizer.create_plotly_graph(
                struct_subgraph, positions, "Structuring Pattern - Multiple Small Deposits"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Pattern Analysis")
        target_address = "struct_target_001"
        
        if target_address in filtered_graph.nodes():
            detection_results = st.session_state.detector.detect_structuring(filtered_graph, target_address)
            risk_score, risk_level, _ = st.session_state.scorer.compute_risk_score({'structuring': detection_results})
            
            icon, level_text = {'LOW': ('🟢', 'Low'), 'MEDIUM': ('🟡', 'Medium'), 'HIGH': ('🔴', 'High')}.get(risk_level, ('⚪', 'Unknown'))
            
            st.markdown(f"### Risk Level: {icon} **{level_text}**")
            st.metric("Risk Score", f"{risk_score:.3f}")
            st.metric("Confidence", f"{detection_results['confidence']:.3f}")

            st.subheader("🔍 Evidence")
            if detection_results.get('evidence'):
                st.json(detection_results['evidence'])

            st.subheader("📄 Explanation")
            st.info(detection_results['explanation'])

            if struct_subgraph is not None and fig is not None:
                export_pattern_assets(struct_subgraph, fig, "structuring", target_address, detection_results)
        else:
            st.warning("⚠️ No structuring target addresses found for analysis.")
            st.info("Structuring patterns require multiple donors sending to same target. Load data containing structuring transactions to see analysis.")

def display_coinjoin_analysis():
    st.header("🌀 CoinJoin Mixing Detection")
    
    st.subheader("Pattern Explanation")
    st.write("""
    CoinJoin is a privacy technique that:
    1) Combines multiple users' inputs into one transaction,
    2) Creates multiple equal-value outputs,
    3) Makes it hard to link inputs to specific outputs,
    4) Obscures the transaction trail for participants.
    """)

    start_dt, end_dt = st.session_state.view_time_range if st.session_state.view_time_range else (None, None)
    filtered_graph = filter_graph_by_time(st.session_state.graph, start_dt, end_dt)

    # Show transactions involved
    win_start = pd.to_datetime(start_dt, utc=True) if start_dt else None
    win_end = pd.to_datetime(end_dt, utc=True) if end_dt else None
    
    coinjoin_transactions = []
    for tx in st.session_state.transactions:
        if (('coinjoin' in tx.from_address) or ('coinjoin' in tx.to_address) or 
            (hasattr(tx, 'meta') and tx.meta and tx.meta.get('pattern') == 'coinjoin')):
            tt = pd.to_datetime(tx.timestamp, errors='coerce', utc=True)
            if (not pd.isna(tt)) and (win_start is None or tt >= win_start) and (win_end is None or tt <= win_end):
                coinjoin_transactions.append(tx)

    st.subheader(f"📋 Transactions Involved ({len(coinjoin_transactions)})")
    if coinjoin_transactions:
        st.dataframe(pd.DataFrame([{
            'Transaction ID': tx.txid[:16] + '...',
            'From Address': tx.from_address,
            'To Address': tx.to_address,
            'Amount (BTC)': f"{tx.value:.6f}",
            'Inputs': len(tx.inputs) if hasattr(tx, 'inputs') and tx.inputs else 0,
            'Outputs': len(tx.outputs) if hasattr(tx, 'outputs') and tx.outputs else 0,
            'Classification': tx.classification.upper()
        } for tx in coinjoin_transactions]), use_container_width=True)
    else:
        st.info("No CoinJoin transactions found in the selected time window")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Pattern Visualization")
        coinjoin_addresses = [addr for addr in filtered_graph.nodes() if 'coinjoin' in addr]
        coinjoin_subgraph = None
        fig = None

        if coinjoin_addresses:
            coinjoin_subgraph = filtered_graph.subgraph(coinjoin_addresses).copy()
            positions = st.session_state.graph_builder.compute_layout(coinjoin_subgraph)
            
            fig = st.session_state.visualizer.create_plotly_graph(
                coinjoin_subgraph, positions, "CoinJoin Mixing Pattern"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Pattern Analysis")
        
        # Try multiple approaches to find a coinjoin transaction
        coinjoin_tx = None
        
        # First, try to find by meta pattern
        for tx in coinjoin_transactions:
            if hasattr(tx, 'meta') and tx.meta and tx.meta.get('pattern') == 'coinjoin':
                coinjoin_tx = tx
                break
        
        # If not found, use any coinjoin transaction from the filtered list
        if not coinjoin_tx and coinjoin_transactions:
            coinjoin_tx = coinjoin_transactions[0]  # Use the first available
        
        # If still not found, create a synthetic analysis target
        if not coinjoin_tx and coinjoin_addresses:
            # Create a mock transaction data for analysis
            target_address = coinjoin_addresses[0]
            st.info(f"Analyzing CoinJoin pattern for address: {target_address}")
            
            # Mock transaction data for demonstration
            transaction_data = {
                'inputs': [{'address': f'input_{i}', 'value': 0.1} for i in range(3)],
                'outputs': [{'address': f'output_{i}', 'value': 0.1} for i in range(3)]
            }
            
            detection_results = st.session_state.detector.detect_coinjoin_like(transaction_data)
            risk_score, risk_level, _ = st.session_state.scorer.compute_risk_score({'coinjoin': detection_results})
            
            icon, level_text = {'LOW': ('🟢', 'Low'), 'MEDIUM': ('🟡', 'Medium'), 'HIGH': ('🔴', 'High')}.get(risk_level, ('⚪', 'Unknown'))
            
            st.markdown(f"### Risk Level: {icon} **{level_text}**")
            st.metric("Risk Score", f"{risk_score:.3f}")
            st.metric("Confidence", f"{detection_results['confidence']:.3f}")

            st.subheader("🔍 Evidence")
            if detection_results.get('evidence'):
                st.json(detection_results['evidence'])

            st.subheader("📄 Explanation")
            st.info(detection_results['explanation'])

            if coinjoin_subgraph is not None and fig is not None:
                export_pattern_assets(coinjoin_subgraph, fig, "coinjoin", target_address, detection_results)
        
        elif coinjoin_tx:
           
            transaction_data = {
                'inputs': coinjoin_tx.inputs if hasattr(coinjoin_tx, 'inputs') else [],
                'outputs': coinjoin_tx.outputs if hasattr(coinjoin_tx, 'outputs') else []
            }
            detection_results = st.session_state.detector.detect_coinjoin_like(transaction_data)
            risk_score, risk_level, _ = st.session_state.scorer.compute_risk_score({'coinjoin': detection_results})
            
            icon, level_text = {'LOW': ('🟢', 'Low'), 'MEDIUM': ('🟡', 'Medium'), 'HIGH': ('🔴', 'High')}.get(risk_level, ('⚪', 'Unknown'))
            
            st.markdown(f"### Risk Level: {icon} **{level_text}**")
            st.metric("Risk Score", f"{risk_score:.3f}")
            st.metric("Confidence", f"{detection_results['confidence']:.3f}")

            st.subheader("🔍 Evidence")
            if detection_results.get('evidence'):
                st.json(detection_results['evidence'])

            st.subheader("📄 Explanation")
            st.info(detection_results['explanation'])

            if coinjoin_subgraph is not None and fig is not None:
                export_pattern_assets(coinjoin_subgraph, fig, "coinjoin", coinjoin_tx.txid, detection_results)
        
        else:
            # No coinjoin data available
            st.warning("⚠️ No CoinJoin transactions found for analysis.")
            st.info("CoinJoin patterns require transactions with multiple inputs and outputs. Load data containing CoinJoin transactions to see analysis.")

def display_rapid_movement_analysis():
    st.header("⚡ Rapid Movement Detection")
    
    st.subheader("Pattern Explanation")
    st.write("""
    Rapid Movement involves:
    1) Moving funds quickly across multiple addresses,
    2) Very short time gaps between transactions,
    3) Automated or scripted transfers to evade detection,
    4) Often used to stay ahead of investigation timelines.
    """)

    start_dt, end_dt = st.session_state.view_time_range if st.session_state.view_time_range else (None, None)
    filtered_graph = filter_graph_by_time(st.session_state.graph, start_dt, end_dt)

    # Show transactions involved
    win_start = pd.to_datetime(start_dt, utc=True) if start_dt else None
    win_end = pd.to_datetime(end_dt, utc=True) if end_dt else None
    
    rapid_transactions = []
    for tx in st.session_state.transactions:
        if ('rapid_addr' in tx.from_address) or ('rapid_addr' in tx.to_address):
            tt = pd.to_datetime(tx.timestamp, errors='coerce', utc=True)
            if (not pd.isna(tt)) and (win_start is None or tt >= win_start) and (win_end is None or tt <= win_end):
                rapid_transactions.append(tx)

    st.subheader(f"📋 Transactions Involved ({len(rapid_transactions)})")
    if rapid_transactions:
        st.dataframe(pd.DataFrame([{
            'Transaction ID': tx.txid[:16] + '...',
            'From Address': tx.from_address,
            'To Address': tx.to_address,
            'Amount (BTC)': f"{tx.value:.6f}",
            'Timestamp': tx.timestamp,
            'Classification': tx.classification.upper()
        } for tx in rapid_transactions]), use_container_width=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Pattern Visualization")
        rapid_addresses = [addr for addr in filtered_graph.nodes() if 'rapid_addr' in addr]
        rapid_subgraph = None
        fig = None

        if rapid_addresses:
            rapid_subgraph = filtered_graph.subgraph(rapid_addresses).copy()
            positions = st.session_state.graph_builder.compute_layout(rapid_subgraph)
            
            fig = st.session_state.visualizer.create_plotly_graph(
                rapid_subgraph, positions, "Rapid Movement Pattern"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Pattern Analysis")
        target_address = "rapid_addr_0"
        
        if target_address in filtered_graph.nodes():
            detection_results = st.session_state.detector.detect_rapid_movement(filtered_graph, target_address)
            risk_score, risk_level, _ = st.session_state.scorer.compute_risk_score({'rapid_movement': detection_results})
            
            icon, level_text = {'LOW': ('🟢', 'Low'), 'MEDIUM': ('🟡', 'Medium'), 'HIGH': ('🔴', 'High')}.get(risk_level, ('⚪', 'Unknown'))
            
            st.markdown(f"### Risk Level: {icon} **{level_text}**")
            st.metric("Risk Score", f"{risk_score:.3f}")
            st.metric("Confidence", f"{detection_results['confidence']:.3f}")

            st.subheader("🔍 Evidence")
            if detection_results.get('evidence'):
                st.json(detection_results['evidence'])

            st.subheader("📄 Explanation")
            st.info(detection_results['explanation'])

            if rapid_subgraph is not None and fig is not None:
                export_pattern_assets(rapid_subgraph, fig, "rapid_movement", target_address, detection_results)
        else:
            st.warning("⚠️ No rapid movement addresses found for analysis.")
            st.info("Rapid movement patterns require sequential fast transfers. Load data containing rapid movement transactions to see analysis.")

def display_layering_analysis():
    st.header("🕸️ Layering (Complex Obfuscation)")
    
    st.subheader("Pattern Explanation")
    st.write("""
    Layering is a sophisticated technique involving:
    1) Complex multi-stage fund movement,
    2) Multiple intermediary addresses,
    3) Varying amounts and timing to confuse analysis,
    4) Creates maximum entropy in the audit trail.
    """)

    start_dt, end_dt = st.session_state.view_time_range if st.session_state.view_time_range else (None, None)
    filtered_graph = filter_graph_by_time(st.session_state.graph, start_dt, end_dt)

    # Show transactions involved
    win_start = pd.to_datetime(start_dt, utc=True) if start_dt else None
    win_end = pd.to_datetime(end_dt, utc=True) if end_dt else None
    
    layer_transactions = []
    for tx in st.session_state.transactions:
        if ('layer_' in tx.from_address) or ('layer_' in tx.to_address):
            tt = pd.to_datetime(tx.timestamp, errors='coerce', utc=True)
            if (not pd.isna(tt)) and (win_start is None or tt >= win_start) and (win_end is None or tt <= win_end):
                layer_transactions.append(tx)

    st.subheader(f"📋 Transactions Involved ({len(layer_transactions)})")
    if layer_transactions:
        st.dataframe(pd.DataFrame([{
            'Transaction ID': tx.txid[:16] + '...',
            'From Address': tx.from_address,
            'To Address': tx.to_address,
            'Amount (BTC)': f"{tx.value:.6f}",
            'Timestamp': tx.timestamp,
            'Classification': tx.classification.upper()
        } for tx in layer_transactions]), use_container_width=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Pattern Visualization")
        layer_addresses = [addr for addr in filtered_graph.nodes() if 'layer_' in addr]
        layer_subgraph = None
        fig = None

        if layer_addresses:
            layer_subgraph = filtered_graph.subgraph(layer_addresses).copy()
            positions = st.session_state.graph_builder.compute_layout(layer_subgraph)
            
            fig = st.session_state.visualizer.create_plotly_graph(
                layer_subgraph, positions, "Layering Pattern - Complex Multi-hop"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Pattern Analysis")
        target_address = "layer_source"
        
        if target_address in filtered_graph.nodes():
            detection_results = st.session_state.detector.detect_layering(filtered_graph, target_address)
            risk_score, risk_level, _ = st.session_state.scorer.compute_risk_score({'layering': detection_results})
            
            icon, level_text = {'LOW': ('🟢', 'Low'), 'MEDIUM': ('🟡', 'Medium'), 'HIGH': ('🔴', 'High')}.get(risk_level, ('⚪', 'Unknown'))
            
            st.markdown(f"### Risk Level: {icon} **{level_text}**")
            st.metric("Risk Score", f"{risk_score:.3f}")
            st.metric("Confidence", f"{detection_results['confidence']:.3f}")

            st.subheader("🔍 Evidence")
            if detection_results.get('evidence'):
                st.json(detection_results['evidence'])

            st.subheader("📄 Explanation")
            st.info(detection_results['explanation'])

            if layer_subgraph is not None and fig is not None:
                export_pattern_assets(layer_subgraph, fig, "layering", target_address, detection_results)
        else:
            st.warning("⚠️ No layering source addresses found for analysis.")
            st.info("Layering patterns require complex multi-stage movements. Load data containing layering transactions to see analysis.")

def display_tf_crowdfunding_analysis():
    st.header("💰 Terrorist Financing Crowdfunding")
    
    st.subheader("Pattern Explanation")
    st.write("""
    TF Crowdfunding involves:
    1) Many small donations from different sources,
    2) Funds channeled to finance terrorist activities,
    3) Mimics legitimate crowdfunding to avoid detection,
    4) Exploits cryptocurrency's pseudonymous nature.
    """)

    start_dt, end_dt = st.session_state.view_time_range if st.session_state.view_time_range else (None, None)
    filtered_graph = filter_graph_by_time(st.session_state.graph, start_dt, end_dt)

    # Show transactions involved
    win_start = pd.to_datetime(start_dt, utc=True) if start_dt else None
    win_end = pd.to_datetime(end_dt, utc=True) if end_dt else None
    
    tf_transactions = []
    for tx in st.session_state.transactions:
        if ('tf_donor' in tx.from_address) or ('tf_target' in tx.to_address):
            tt = pd.to_datetime(tx.timestamp, errors='coerce', utc=True)
            if (not pd.isna(tt)) and (win_start is None or tt >= win_start) and (win_end is None or tt <= win_end):
                tf_transactions.append(tx)

    st.subheader(f"📋 Transactions Involved ({len(tf_transactions)})")
    if tf_transactions:
        st.dataframe(pd.DataFrame([{
            'Transaction ID': tx.txid[:16] + '...',
            'From Address': tx.from_address,
            'To Address': tx.to_address,
            'Amount (BTC)': f"{tx.value:.6f}",
            'Timestamp': tx.timestamp,
            'Classification': tx.classification.upper()
        } for tf in tf_transactions]), use_container_width=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Pattern Visualization")
        tf_addresses = [addr for addr in filtered_graph.nodes() if ('tf_donor' in addr or 'tf_target' in addr)]
        tf_subgraph = None
        fig = None

        if tf_addresses:
            tf_subgraph = filtered_graph.subgraph(tf_addresses).copy()
            positions = st.session_state.graph_builder.compute_layout(tf_subgraph)
            
            fig = st.session_state.visualizer.create_plotly_graph(
                tf_subgraph, positions, "TF Crowdfunding - Multiple Donors to Target"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Pattern Analysis")
        target_address = "tf_target_001"
        
        if target_address in filtered_graph.nodes():
            detection_results = st.session_state.detector.detect_tf_crowdfunding(filtered_graph, target_address)
            risk_score, risk_level, _ = st.session_state.scorer.compute_risk_score({'tf_crowdfunding': detection_results})
            
            icon, level_text = {'LOW': ('🟢', 'Low'), 'MEDIUM': ('🟡', 'Medium'), 'HIGH': ('🔴', 'High')}.get(risk_level, ('⚪', 'Unknown'))
            
            st.markdown(f"### Risk Level: {icon} **{level_text}**")
            st.metric("Risk Score", f"{risk_score:.3f}")
            st.metric("Confidence", f"{detection_results['confidence']:.3f}")

            st.subheader("🔍 Evidence")
            if detection_results.get('evidence'):
                st.json(detection_results['evidence'])

            st.subheader("📄 Explanation")
            st.info(detection_results['explanation'])

            if tf_subgraph is not None and fig is not None:
                export_pattern_assets(tf_subgraph, fig, "tf_crowdfunding", target_address, detection_results)
        else:
            st.warning("⚠️ No TF crowdfunding target addresses found for analysis.")
            st.info("TF crowdfunding patterns require multiple donors to same target. Load data containing TF transactions to see analysis.")

def display_chain_hopping_analysis():
    st.header("🔗 Cross-Chain Money Laundering")
    
    st.subheader("Pattern Explanation")
    st.write("""
    Chain hopping involves:
    1) Moving funds from Bitcoin to an exchange/bridge,
    2) Converting BTC to ETH (or other cryptocurrencies),
    3) Moving the converted funds to different addresses,
    4) Using multiple blockchains to obfuscate the money trail.
    """)

    if not st.session_state.chain_hop_events:
        st.warning("⚠️ No chain hopping events detected. Enable cross-chain analysis in the sidebar and load both Bitcoin and Ethereum data.")
        return

    # Display detected events
    st.subheader(f"🔍 Detected Events ({len(st.session_state.chain_hop_events)})")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create DataFrame for events
        events_data = []
        for event in st.session_state.chain_hop_events:
            try:
                btc_time = event.btc_time if hasattr(event.btc_time, 'isoformat') else pd.to_datetime(event.btc_time)
                eth_time = event.eth_time if hasattr(event.eth_time, 'isoformat') else pd.to_datetime(event.eth_time)
                time_gap = abs((eth_time - btc_time).total_seconds() / 3600)
                
                events_data.append({
                    'BTC Transaction': event.btc_txid[:16] + '...' if event.btc_txid else 'N/A',
                    'BTC → Exchange': event.btc_out_addr[:16] + '...' if event.btc_out_addr else 'N/A',
                    'Exchange → ETH': event.eth_in_addr[:16] + '...' if event.eth_in_addr else 'N/A',
                    'Time Gap (hours)': round(time_gap, 2),
                    'Route': event.label if hasattr(event, 'label') else 'Unknown'
                })
            except Exception as e:
                st.error(f"Error processing event: {e}")
                continue

        if events_data:
            df = pd.DataFrame(events_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No valid event data to display")

        # Visualization placeholder
        st.subheader("Cross-Chain Flow Visualization")
        st.info("🚧 Advanced cross-chain visualization coming soon. Current detection shows exchange routing patterns.")

    with col2:
        st.subheader("🎯 Pattern Analysis")
        
        # Run chain hopping detection
        target_address = st.text_input("Target Address:", value="1DEP8i3QJCsomS4BSMY2RpU1upv62aGvhD")
        
        if target_address:
            detection_results = st.session_state.detector.detect_chain_hopping(
                st.session_state.chain_hop_events,
                target_address
            )

            risk_score, risk_level, _ = st.session_state.scorer.compute_risk_score({'chain_hopping': detection_results})
            icon, level_text = {'LOW': ('🟢', 'Low'), 'MEDIUM': ('🟡', 'Medium'), 'HIGH': ('🔴', 'High')}.get(risk_level, ('⚪', 'Unknown'))
            
            st.markdown(f"### Risk Level: {icon} **{level_text}**")
            st.metric("Risk Score", f"{risk_score:.3f}")
            st.metric("Confidence", f"{detection_results['confidence']:.3f}")

            st.subheader("🔍 Evidence")
            if detection_results.get('evidence'):
                st.json(detection_results['evidence'])

            st.subheader("📄 Explanation")
            st.info(detection_results['explanation'])

def display_graph_statistics():
    if not st.session_state.graph:
        return

    st.subheader("Graph Statistics")
    
    if len(st.session_state.graph.nodes()) > 0:
        degree_dict = dict(st.session_state.graph.degree())
        total_degree = sum(degree_dict.values())
        avg_degree = total_degree / len(st.session_state.graph.nodes())
        
        stats = {
            "Nodes": len(st.session_state.graph.nodes()),
            "Edges": len(st.session_state.graph.edges()),
            "Avg Degree": f"{avg_degree:.2f}"
        }
        
        for stat_name, stat_value in stats.items():
            st.metric(stat_name, stat_value)

    st.subheader("Node Classifications")
    classifications = {}
    for node in st.session_state.graph.nodes():
        cls = st.session_state.graph.nodes[node].get('classification', 'unknown')
        classifications[cls] = classifications.get(cls, 0) + 1
    
    for cls, count in classifications.items():
        st.metric(cls.title(), count)

def store_analysis_results(pattern_name, target_address, detection_results):
    risk_score, risk_level, risk_summary = st.session_state.scorer.compute_risk_score({pattern_name: detection_results})
    
    st.session_state.analysis_results = {
        'target_address': target_address,
        'detection_results': {pattern_name: detection_results},
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_summary': risk_summary
    }

def generate_and_download_sar():
    if 'analysis_results' not in st.session_state:
        st.warning("No analysis results available")
        return

    results = st.session_state.analysis_results
    sar_report = st.session_state.sar_generator.generate_sar_report(
        results['target_address'],
        results['risk_summary'],
        results['detection_results']
    )

    report_json = json.dumps(sar_report, indent=2)
    st.download_button(
        label="📄 Download SAR Report",
        data=report_json,
        file_name=f"sar_report_{results['target_address'][:8]}_{sar_report['report_id']}.json",
        mime="application/json"
    )

    with st.expander("SAR Report Preview"):
        st.json(sar_report)

def export_graph_data():
    if not st.session_state.graph:
        st.warning("No graph data available")
        return

    nodes_data = []
    for node in st.session_state.graph.nodes(data=True):
        node_dict = {'id': node[0]}
        node_dict.update(node[1])
        nodes_data.append(node_dict)

    edges_data = []
    for edge in st.session_state.graph.edges(data=True):
        edge_dict = {'source': edge[0], 'target': edge[1]}
        edge_dict.update(edge[2])
        edges_data.append(edge_dict)

    export_data = {
        'nodes': nodes_data,
        'edges': edges_data,
        'metadata': {
            'exported_at': pd.Timestamp.now().isoformat(),
            'node_count': len(nodes_data),
            'edge_count': len(edges_data)
        }
    }

    export_json = json.dumps(export_data, indent=2)
    st.download_button(
        label="📈 Download Graph Data",
        data=export_json,
        file_name=f"graph_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()


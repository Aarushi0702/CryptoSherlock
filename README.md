# 🔍 CryptoSherlock X - Live Edition

*Explainable Cryptocurrency Forensics with Live Blockchain Data & Cross-Chain Analysis*
Advanced machine learning platform for cryptocurrency forensics and compliance investigations.
Real-time blockchain analysis with 7 detection algorithms for identifying suspicious transaction patterns.

## 🚀 Features

### 🎯 **7 ML Pattern Detection Algorithms**
- **⛓️ Peel Chain**: Sequential fund transfers with decreasing amounts
- **💸 Structuring/Smurfing**: Multiple small deposits below reporting thresholds  
- **🌀 CoinJoin**: Privacy mixing with equal outputs detection
- **⚡ Rapid Movement**: Fast sequential transactions across addresses
- **🕸️ Layering**: Complex multi-hop obfuscation with entropy analysis
- **💰 TF Crowdfunding**: Terrorist financing through micro-donations
- **🔗 Chain Hopping**: Cross-chain money laundering via exchanges

### 🌐 **Live Blockchain Data**
- Real-time Bitcoin transaction analysis via **Blockstream API** (free, no API key required)
- Ethereum transaction support via **Etherscan API**
- Multi-address batch processing
- Intelligent caching system

### 🔬 **Advanced Analytics**
- Interactive network visualizations with **Plotly**
- Entity clustering using address heuristics
- Risk scoring with configurable weights
- Time-window filtering and analysis
- Cross-chain laundering detection

### 🚨 **Alerts System**
CryptoSherlock X incorporates an advanced Alerts System designed for continuous monitoring and timely notifications of suspicious activities across blockchain transactions.

Key Features:
-Real-Time Alerts
-Multi-Level Risk Classification
-Pattern-Based Detection
-Continuous Surveillance
-Integrated Alert Management
-Automated Re-Analysis

**Operational Benefits:**
      -Enables swift response to potential illicit transactions
      -Enhances investigative situational awareness through structured and explainable alerts
      -Supports compliance by notifying of transactions that may require filing Suspicious Activity Reports
      -Provides real-time monitoring capabilities with configurable thresholds and detailed evidence tracking

### 📊 **Professional Reporting**
- **SAR (Suspicious Activity Report)** generation
- Pattern-specific evidence exports
- Compliance-ready documentation
- Watchlist monitoring with real-time alerts

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cryptosherlock-x.git
cd cryptosherlock-x
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

###  Demo Mode
Launch with synthetic data for immediate exploration:
- Click "Load Synthetic Demo Data" in the sidebar
- Explore all 7 detection patterns with pre-generated examples
- No API keys required for demo mode

##  Usage Examples

### Analyze a Bitcoin Address
```
In the sidebar, select "Live Bitcoin API (Single Address)"
Enter address: 1DEP8i3QJCsomS4BSMY2RpU1upv62aGvhD
Click "Fetch Live Data"
```

### Cross-Chain Analysis
```
Enable "Cross-Chain Analysis" in sidebar
Load Bitcoin and Ethereum data
Click "Detect Chain Hopping" to find exchange-mediated laundering
```

### Multiple Address Investigation
```
Select "Live Bitcoin API (Multiple Addresses)"
Enter comma-separated or line-separated addresses
System will analyze transaction relationships across all addresses
```

##  System Architecture

```
Data Sources → Core Engine → Pattern Detection → Risk Scoring → Reporting
      ↓             ↓               ↓               ↓           ↓
   Live APIs   Graph Builder     7 ML Algos      Weighted   SAR Reports
   Synthetic   Multi-source      Real-time      Confidence   Exports
   File Upload  Integration      Evidence       Thresholds   Compliance
```

### Core Components

| Module | Purpose | Key Features |
|--------|---------|-------------|
| `app.py` | Streamlit web interface | Interactive dashboard, multi-tab analysis |
| `detectors.py` | ML pattern detection | 7 algorithms with confidence scoring |
| `live_api.py` | Blockchain data fetching | Blockstream API integration |
| `scoring.py` | Risk assessment engine | Weighted scoring, thresholds |
| `reporting.py` | Compliance reporting | SAR generation, export formats |
| `graph_viz.py` | Network visualization | Interactive Plotly graphs |

##  Configuration

### API Setup
- **Bitcoin**: Uses free Blockstream API (no registration required)
- **Ethereum**: Requires free Etherscan API key from [etherscan.io](https://etherscan.io/apis)

### Risk Scoring Weights
```python
# Modify in scoring.py
weights = {
    'chain_hopping': 0.95,    # Highest risk
    'structuring': 0.9,
    'layering': 0.85,
    'peel_chain': 0.8,
    'rapid_movement': 0.7,
    'coinjoin': 0.6,
    'tf_crowdfunding': 0.6
}
```
##  Testing

Run the test suite:
```bash
python -m pytest tests/
```

Test individual components:
```bash
python -m pytest tests/test_detectors.py -v
```

## ⚠️ Disclaimer

**Important**: CryptoSherlock X is designed for:
- ✅ Legal compliance and investigation purposes
- ✅ Academic research and education  
- ✅ Financial institution risk assessment
- ✅ Law enforcement forensic analysis

**Not intended for**:
- ❌ Facilitating illegal activities
- ❌ Privacy invasion without legal authority
- ❌ Market manipulation or trading decisions

Users must comply with applicable laws and regulations in their jurisdiction.


##  Acknowledgments

- **Blockchain APIs**: Blockstream, Etherscan for providing free data access
- **Visualization**: Plotly team for excellent graphing capabilities
- **Framework**: Streamlit for rapid prototyping and deployment
- **Community**: Open source contributors and security researchers




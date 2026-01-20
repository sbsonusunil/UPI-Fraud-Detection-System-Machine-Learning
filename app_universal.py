import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import json
import io

# PAGE CONFIG
st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded")

# SESSION STATE INITIALIZATION
if "history" not in st.session_state:
    st.session_state.history = []
if "total_transactions" not in st.session_state:
    st.session_state.total_transactions = 0
if "blocked_transactions" not in st.session_state:
    st.session_state.blocked_transactions = 0
if "flagged_transactions" not in st.session_state:
    st.session_state.flagged_transactions = 0
if "mode" not in st.session_state:
    st.session_state.mode = "single"  # single, batch, api
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "use_ml_model" not in st.session_state:
    st.session_state.use_ml_model = False
if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = {
        "high_amount_threshold": 200000,
        "velocity_window_seconds": 300,
        "max_transactions_in_window": 3,
        "unusual_hour_start": 23,
        "unusual_hour_end": 6
    }

# MODERN CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
    font-family: 'DM Sans', sans-serif;
}

.header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    color: white;
    padding: 32px 40px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 20px 60px rgba(30, 58, 138, 0.3);
    position: relative;
    overflow: hidden;
}

.header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

.card {
    background: white;
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    border: 1px solid rgba(59, 130, 246, 0.1);
}

.badge {
    padding: 18px;
    border-radius: 16px;
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    animation: slideIn 0.5s ease-out;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.mode-selector {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
}

.mode-btn {
    flex: 1;
    padding: 16px;
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
}

.mode-btn.active {
    background: #3b82f6;
    color: white;
    border-color: #3b82f6;
}

.metric-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e3a8a;
    font-family: 'Space Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# MODEL LOADING (OPTIONAL)
@st.cache_resource
def load_ml_models():
    """Load ML models if available"""
    try:
        model = joblib.load("models/xgb_model.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
        return model, preprocessor, True
    except:
        return None, None, False


# RULE-BASED FRAUD DETECTION (WORKS WITHOUT ML)
def rule_based_fraud_detection(
    amount: float,
    bank: str,
    txn_type: str,
    network: str,
    hour: int,
    device: str,
    location: str,
    receiver_bank: str = None,
    merchant_category: str = None,
    custom_rules: Dict = None
) -> Tuple[float, List[str]]:
    """
    Universal rule-based fraud detection that works without ML models.
    Returns risk score (0-1) and list of risk indicators.
    """
    
    rules = custom_rules or st.session_state.custom_rules
    risk_score = 0.0
    risk_indicators = []
    
    # Amount-based rules
    if amount > rules["high_amount_threshold"]:
        risk_score += 0.25
        risk_indicators.append(f"🔴 High-value transaction (>₹{rules['high_amount_threshold']:,})")
    
    if amount > 500000:
        risk_score += 0.20
        risk_indicators.append("🔴 Very high amount (>₹5L)")
    
    if amount < 100:
        risk_score += 0.10
        risk_indicators.append("🟡 Unusually low amount")
    
    # Round amount suspicion
    if amount % 10000 == 0 and amount > 50000:
        risk_score += 0.05
        risk_indicators.append("🟡 Suspiciously round amount")
    
    # Network security
    network_risk = {
        "VPN / Proxy": (0.25, "🔴 VPN/Proxy detected"),
        "Public WiFi": (0.15, "🟡 Public WiFi connection"),
        "Mobile Data": (0.0, None),
        "Home WiFi": (0.0, None)
    }
    
    if network in network_risk:
        score, message = network_risk[network]
        if message:
            risk_score += score
            risk_indicators.append(message)
    
    # Time-based rules
    if hour < rules["unusual_hour_end"] or hour >= rules["unusual_hour_start"]:
        risk_score += 0.12
        risk_indicators.append(f"🟡 Unusual hour ({hour}:00)")
    
    # Weekend transactions (higher risk)
    if datetime.datetime.today().weekday() >= 5 and amount > 100000:
        risk_score += 0.08
        risk_indicators.append("🟡 High-value weekend transaction")
    
    # Device-based risk
    if "Unknown" in device or "Emulator" in device:
        risk_score += 0.15
        risk_indicators.append("🔴 Suspicious device detected")
    
    # Geographic risk
    high_risk_states = ["Unknown", "International"]
    if location in high_risk_states:
        risk_score += 0.18
        risk_indicators.append("🔴 High-risk location")
    
    # Transaction type risk
    if txn_type == "Bank Account Transfer" and amount > 200000:
        risk_score += 0.10
        risk_indicators.append("🟡 Large bank transfer")
    
    # Velocity check (if history available)
    if len(st.session_state.history) >= rules["max_transactions_in_window"]:
        recent = st.session_state.history[-(rules["max_transactions_in_window"]):]
        now = datetime.datetime.now()
        
        # Check if all recent transactions happened within the window
        try:
            times = [datetime.datetime.strptime(t["Time"], "%H:%M:%S") for t in recent]
            time_diffs = [(now.replace(second=0, microsecond=0) - t.replace(second=0, microsecond=0)).seconds for t in times]
            
            if all(diff < rules["velocity_window_seconds"] for diff in time_diffs):
                risk_score += 0.22
                risk_indicators.append(f"🔴 High velocity ({len(recent)} txns in {rules['velocity_window_seconds']}s)")
        except:
            pass
    
    # Cross-bank transaction risk
    if receiver_bank and bank != receiver_bank and amount > 100000:
        risk_score += 0.07
        risk_indicators.append("🟡 Cross-bank high-value transfer")
    
    # Merchant category risk (if applicable)
    high_risk_categories = ["Gambling", "Cryptocurrency", "Adult Content"]
    if merchant_category in high_risk_categories:
        risk_score += 0.20
        risk_indicators.append(f"🔴 High-risk category: {merchant_category}")
    
    # Cap the score at 1.0
    risk_score = min(risk_score, 1.0)
    
    return risk_score, risk_indicators


# ML-BASED DETECTION (IF MODEL AVAILABLE)
def ml_based_fraud_detection(
    amount: float,
    bank: str,
    txn_type: str,
    network: str,
    hour: int,
    device: str,
    location: str,
    model,
    preprocessor
) -> float:
    """ML-based fraud detection using trained model"""
    
    df = pd.DataFrame([{
        "transaction type": "P2P" if "P2P" in txn_type else "P2M",
        "merchant_category": "General",
        "transaction_status": "SUCCESS",
        "sender_age_group": "26-35",
        "receiver_age_group": "26-35",
        "sender_state": location,
        "sender_bank": bank,
        "receiver_bank": "SBI",
        "device_type": device.split('-')[0].strip() if '-' in device else device,
        "network_type": network,
        "is_weekend": datetime.datetime.today().weekday() >= 5,
        "year": 2025,
        "month": datetime.datetime.now().month,
        "day": datetime.datetime.now().day,
        "minute": datetime.datetime.now().minute,
        "hour_sin": np.sin(2*np.pi*hour/24),
        "hour_cos": np.cos(2*np.pi*hour/24),
        "day_of_week_sin": 0,
        "day_of_week_cos": 1,
        "amount_log": np.log1p(amount)
    }])
    
    X = preprocessor.transform(df)
    ml_score = model.predict_proba(X)[0][1]
    return ml_score


# COMBINED FRAUD DETECTION
def detect_fraud(
    amount: float,
    bank: str,
    txn_type: str,
    network: str,
    hour: int,
    device: str,
    location: str,
    use_ml: bool = False,
    model = None,
    preprocessor = None,
    **kwargs
) -> Tuple[float, List[str], str]:
    """
    Universal fraud detection combining ML and rules.
    Works with or without ML models.
    """
    
    # Get rule-based score
    rule_score, indicators = rule_based_fraud_detection(
        amount, bank, txn_type, network, hour, device, location,
        kwargs.get('receiver_bank'),
        kwargs.get('merchant_category'),
        kwargs.get('custom_rules')
    )
    
    # Get ML score if available
    ml_score = 0.0
    if use_ml and model is not None and preprocessor is not None:
        try:
            ml_score = ml_based_fraud_detection(
                amount, bank, txn_type, network, hour, device, location,
                model, preprocessor
            )
        except Exception as e:
            st.warning(f"ML model failed: {e}. Using rule-based only.")
    
    # Combine scores (weighted average if ML available)
    if use_ml and ml_score > 0:
        final_score = (0.6 * ml_score) + (0.4 * rule_score)
        detection_method = "ML + Rules"
    else:
        final_score = rule_score
        detection_method = "Rule-based"
    
    return final_score, indicators, detection_method


def get_decision(risk_score: float, manual_review: bool = False) -> Tuple[str, str]:
    """Determine transaction decision"""
    if manual_review:
        return "ROUTE TO FRAUD OPS", "#f59e0b"
    elif risk_score > 0.75:
        return "DECLINE TRANSACTION", "#dc2626"
    elif risk_score > 0.45:
        return "MANUAL REVIEW REQUIRED", "#f59e0b"
    else:
        return "APPROVE TRANSACTION", "#22c55e"


# BATCH PROCESSING
def process_batch_transactions(df: pd.DataFrame, use_ml: bool, model=None, preprocessor=None) -> pd.DataFrame:
    """Process multiple transactions from CSV/Excel"""
    
    results = []
    
    for idx, row in df.iterrows():
        # Extract fields (flexible column names)
        amount = row.get('amount', row.get('Amount', row.get('transaction_amount', 0)))
        bank = row.get('bank', row.get('sender_bank', row.get('Bank', 'Unknown')))
        txn_type = row.get('type', row.get('transaction_type', 'P2P'))
        network = row.get('network', row.get('network_type', 'Mobile Data'))
        hour = row.get('hour', datetime.datetime.now().hour)
        device = row.get('device', row.get('device_type', 'Unknown'))
        location = row.get('location', row.get('sender_state', 'Unknown'))
        
        # Detect fraud
        risk_score, indicators, method = detect_fraud(
            amount, bank, txn_type, network, hour, device, location,
            use_ml, model, preprocessor
        )
        
        decision, _ = get_decision(risk_score)
        
        results.append({
            'Transaction_ID': row.get('id', f'TXN_{idx+1}'),
            'Amount': amount,
            'Bank': bank,
            'Risk_Score': round(risk_score * 100, 2),
            'Decision': decision,
            'Indicators': len(indicators),
            'Method': method
        })
    
    return pd.DataFrame(results)


# HEADER
st.markdown(f"""
<div class="header">
    <div style="display: flex; justify-content: space-between; align-items: center; position: relative; z-index: 1;">
        <div>
            <h1>🛡️ Universal UPI Fraud Detection System</h1>
            <p>Works with ANY Transaction Data • ML + Rule-Based Detection</p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.85rem; opacity: 0.8;">
                {datetime.datetime.now().strftime("%d %b %Y, %H:%M:%S IST")}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# SIDEBAR - CONFIGURATION
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # Detection mode
    st.markdown("#### 🎯 Detection Mode")
    model, preprocessor, models_available = load_ml_models()
    
    if models_available:
        use_ml = st.checkbox("Use ML Model (XGBoost)", value=False, 
                            help="Enable ML-based detection if models are loaded")
        st.session_state.use_ml_model = use_ml
        if use_ml:
            st.success("✅ ML models loaded successfully")
    else:
        st.info("ℹ️ ML models not found. Using rule-based detection.")
        use_ml = False
    
    st.divider()
    
    # Custom rule configuration
    st.markdown("#### 🔧 Custom Rules Configuration")
    with st.expander("Configure Detection Rules", expanded=False):
        st.session_state.custom_rules["high_amount_threshold"] = st.number_input(
            "High Amount Threshold (₹)",
            min_value=10000,
            max_value=1000000,
            value=200000,
            step=10000
        )
        
        st.session_state.custom_rules["velocity_window_seconds"] = st.slider(
            "Velocity Window (seconds)",
            60, 600, 300
        )
        
        st.session_state.custom_rules["max_transactions_in_window"] = st.slider(
            "Max Transactions in Window",
            2, 10, 3
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.custom_rules["unusual_hour_end"] = st.number_input(
                "Unusual Hour End", 0, 23, 6
            )
        with col2:
            st.session_state.custom_rules["unusual_hour_start"] = st.number_input(
                "Unusual Hour Start", 0, 23, 23
            )
    
    st.divider()
    
    # Clear data
    if st.button("🗑️ Clear All Session Data"):
        st.session_state.history = []
        st.session_state.total_transactions = 0
        st.session_state.blocked_transactions = 0
        st.session_state.flagged_transactions = 0
        st.session_state.uploaded_data = None
        st.rerun()


# MODE SELECTION
st.markdown("### 📋 Select Detection Mode")
mode_col1, mode_col2, mode_col3 = st.columns(3)

with mode_col1:
    if st.button("🔍 Single Transaction", use_container_width=True, 
                 type="primary" if st.session_state.mode == "single" else "secondary"):
        st.session_state.mode = "single"

with mode_col2:
    if st.button("📊 Batch Upload (CSV/Excel)", use_container_width=True,
                 type="primary" if st.session_state.mode == "batch" else "secondary"):
        st.session_state.mode = "batch"

with mode_col3:
    if st.button("🔌 API Integration", use_container_width=True,
                 type="primary" if st.session_state.mode == "api" else "secondary"):
        st.session_state.mode = "api"

st.divider()


# SINGLE TRANSACTION MODE
if st.session_state.mode == "single":
    col1, col2 = st.columns([1, 1.3], gap="large")
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Transaction Details")
        
        amount = st.number_input("Amount (₹)", min_value=1, max_value=10000000, value=25000, step=500)
        st.caption(f"💰 **₹{amount:,.2f}**")
        
        bank = st.selectbox("Sender's Bank", 
                           ["State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", 
                            "PNB", "Kotak Mahindra", "Yes Bank", "Other"])
        
        txn_type = st.radio("Transaction Type",
                           ["P2P (Person to Person)", "P2M (Person to Merchant)", 
                            "Bank Account Transfer"], horizontal=True)
        
        with st.expander("🔧 Advanced Parameters", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                device = st.selectbox("Device", 
                                     ["Android - Samsung", "iOS - iPhone", "Android - OnePlus",
                                      "Unknown Device", "Android Emulator"])
                network = st.selectbox("Network", 
                                      ["Mobile Data", "Home WiFi", "Public WiFi", "VPN / Proxy"])
            with col_b:
                location = st.selectbox("Location",
                                       ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu",
                                        "Gujarat", "International", "Unknown"])
                hour = st.slider("Hour (IST)", 0, 23, datetime.datetime.now().hour)
            
            receiver_bank = st.selectbox("Receiver's Bank (optional)",
                                        ["Same as Sender", "State Bank of India", "HDFC Bank", 
                                         "ICICI Bank", "Other"])
            
            merchant_cat = st.selectbox("Merchant Category (for P2M)",
                                       ["General", "Food & Dining", "Shopping", "Travel",
                                        "Gambling", "Cryptocurrency"])
        
        manual_review = st.toggle("👨‍💼 Force Manual Review")
        run = st.button("🔍 **RUN FRAUD ASSESSMENT**", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Assessment Results")
        
        if run:
            with st.spinner("🔄 Analyzing transaction..."):
                time.sleep(0.8)
            
            # Detect fraud
            risk_score, indicators, method = detect_fraud(
                amount, bank, txn_type, network, hour, device, location,
                st.session_state.use_ml_model, model, preprocessor,
                receiver_bank=receiver_bank if receiver_bank != "Same as Sender" else bank,
                merchant_category=merchant_cat
            )
            
            decision, color = get_decision(risk_score, manual_review)
            
            # Update metrics
            st.session_state.total_transactions += 1
            if "DECLINE" in decision:
                st.session_state.blocked_transactions += 1
            elif "REVIEW" in decision or manual_review:
                st.session_state.flagged_transactions += 1
            
            # Display decision
            icon = "✅" if "APPROVE" in decision else "⚠️" if "REVIEW" in decision else "🚫"
            st.markdown(f"<div class='badge' style='background:{color};color:white'>{icon} {decision}</div>",
                       unsafe_allow_html=True)
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score * 100,
                delta={'reference': 45},
                title={'text': f"Risk Score ({method})", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color, 'thickness': 0.75},
                    'steps': [
                        {'range': [0, 45], 'color': '#d1fae5'},
                        {'range': [45, 75], 'color': '#fed7aa'},
                        {'range': [75, 100], 'color': '#fecaca'}
                    ],
                    'threshold': {'line': {'color': color, 'width': 4}, 'value': risk_score * 100}
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk indicators
            if indicators:
                st.warning(f"**⚠️ {len(indicators)} Risk Signal(s) Detected:**")
                for ind in indicators:
                    st.markdown(f"• {ind}")
            else:
                st.success("**✅ No Risk Indicators Detected**")
            
            # Add to history
            st.session_state.history.append({
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Amount": f"₹{amount:,.0f}",
                "Risk %": round(risk_score*100, 2),
                "Decision": decision,
                "Method": method
            })
        
        else:
            st.info("👆 Enter transaction details and click **RUN ASSESSMENT**")
        
        st.markdown("</div>", unsafe_allow_html=True)


# BATCH MODE
elif st.session_state.mode == "batch":
    st.markdown("### 📊 Batch Transaction Analysis")
    
    st.info("""
    **Upload a CSV or Excel file** with your transaction data. The file should contain columns like:
    - `amount` or `transaction_amount`
    - `bank` or `sender_bank`
    - `type` or `transaction_type`
    - `network` or `network_type`
    - Other optional columns: `device`, `location`, `hour`, etc.
    """)
    
    uploaded_file = st.file_uploader("Upload Transaction File", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} transactions from file")
            
            with st.expander("📋 Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🔍 **ANALYZE ALL TRANSACTIONS**", type="primary"):
                with st.spinner(f"Analyzing {len(df)} transactions..."):
                    results_df = process_batch_transactions(
                        df, st.session_state.use_ml_model, model, preprocessor
                    )
                
                st.success(f"✅ Analysis complete!")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Analyzed", len(results_df))
                col2.metric("Approved", len(results_df[results_df['Decision'].str.contains('APPROVE')]))
                col3.metric("Flagged", len(results_df[results_df['Decision'].str.contains('REVIEW')]))
                col4.metric("Blocked", len(results_df[results_df['Decision'].str.contains('DECLINE')]))
                
                # Results table
                st.markdown("### 📊 Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    f"fraud_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
                
                # Visualizations
                col_v1, col_v2 = st.columns(2)
                
                with col_v1:
                    fig_pie = px.pie(results_df, names='Decision', title='Decision Distribution')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_v2:
                    fig_hist = px.histogram(results_df, x='Risk_Score', nbins=20, title='Risk Score Distribution')
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


# API MODE
elif st.session_state.mode == "api":
    st.markdown("### 🔌 API Integration")
    
    tab1, tab2 = st.tabs(["📖 API Documentation", "🧪 Test API"])
    
    with tab1:
        st.markdown("""
        ### REST API Endpoint
        
        **Endpoint:** `POST /api/v1/fraud-detection`
        
        **Request Body:**
        ```json
        {
          "amount": 25000,
          "bank": "HDFC Bank",
          "transaction_type": "P2P",
          "network": "Mobile Data",
          "hour": 14,
          "device": "Android - Samsung",
          "location": "Maharashtra",
          "use_ml": false
        }
        ```
        
        **Response:**
        ```json
        {
          "risk_score": 0.324,
          "decision": "APPROVE TRANSACTION",
          "indicators": [],
          "method": "Rule-based",
          "timestamp": "2026-01-20T14:32:15Z"
        }
        ```
        
        ### Python Client Example
        ```python
        import requests
        
        response = requests.post(
            "https://your-api-endpoint.com/api/v1/fraud-detection",
            json={
                "amount": 25000,
                "bank": "HDFC Bank",
                "transaction_type": "P2P",
                "network": "Mobile Data",
                "hour": 14,
                "device": "Android - Samsung",
                "location": "Maharashtra"
            }
        )
        
        result = response.json()
        print(f"Decision: {result['decision']}")
        print(f"Risk Score: {result['risk_score']}")
        ```
        """)
    
    with tab2:
        st.markdown("#### 🧪 Test API Request")
        
        api_input = st.text_area("JSON Request Body", 
                                 value=json.dumps({
                                     "amount": 25000,
                                     "bank": "HDFC Bank",
                                     "transaction_type": "P2P",
                                     "network": "Mobile Data",
                                     "hour": 14,
                                     "device": "Android - Samsung",
                                     "location": "Maharashtra"
                                 }, indent=2),
                                 height=200)
        
        if st.button("🚀 Send API Request", type="primary"):
            try:
                data = json.loads(api_input)
                
                risk_score, indicators, method = detect_fraud(
                    data['amount'],
                    data['bank'],
                    data['transaction_type'],
                    data['network'],
                    data['hour'],
                    data['device'],
                    data['location'],
                    st.session_state.use_ml_model,
                    model, preprocessor
                )
                
                decision, _ = get_decision(risk_score)
                
                response = {
                    "risk_score": round(risk_score, 3),
                    "decision": decision,
                    "indicators": indicators,
                    "method": method,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                st.success("✅ API Response:")
                st.json(response)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")


# SESSION STATISTICS
if st.session_state.history:
    st.divider()
    st.markdown("### 📊 Session Statistics")
    
    stat1, stat2, stat3, stat4 = st.columns(4)
    
    with stat1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_transactions}</div>
            <div style="color: #64748b;">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat2:
        approved = st.session_state.total_transactions - st.session_state.blocked_transactions - st.session_state.flagged_transactions
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #22c55e;">{approved}</div>
            <div style="color: #64748b;">Approved</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #f59e0b;">{st.session_state.flagged_transactions}</div>
            <div style="color: #64748b;">Flagged</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #dc2626;">{st.session_state.blocked_transactions}</div>
            <div style="color: #64748b;">Blocked</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Transaction history
    st.divider()
    st.markdown("### 📈 Transaction History")
    
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)
    
    # Trend chart
    if len(hist_df) > 1:
        fig_trend = px.line(hist_df, x="Time", y="Risk %", markers=True, title="Risk Trend")
        fig_trend.add_hline(y=45, line_dash="dash", line_color="orange")
        fig_trend.add_hline(y=75, line_dash="dash", line_color="red")
        st.plotly_chart(fig_trend, use_container_width=True)


# FOOTER
st.markdown("""
<div style="text-align: center; padding: 24px; color: #64748b; margin-top: 32px; border-top: 2px solid #e2e8f0;">
    <p style="margin: 0; font-weight: 600; color: #1e3a8a;">🛡️ Universal UPI Fraud Detection System</p>
    <p style="margin: 4px 0 0 0; font-size: 0.85rem;">Works with ANY transaction data • ML + Rule-based • Real-time or Batch Processing</p>
</div>
""", unsafe_allow_html=True)
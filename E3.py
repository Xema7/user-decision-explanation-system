import os
import json
import hashlib
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

# Paths
ACTIONS_PATH = "data/user_actions.csv"
LEDGER_PATH = "ledger/decision_influence_log.jsonl"

st.set_page_config(page_title="T-Trace Explainable Dashboard", layout="wide")

# Dark neon style
st.markdown("""
    <style>
        body, .stApp {
            background-color: #050505;
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 28px;
            color: #00e6e6;
            text-shadow: 0 0 12px #00ffff;
            font-weight: bold;
        }
        .section {
            color: #76D7C4;
            font-size: 20px;
            margin-top: 24px;
        }
        .explain-box {
            background-color: #141414;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #00e6e6;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 T-Trace: Behavioral Influence Visualizer</div>", unsafe_allow_html=True)
st.caption("Multi-category explainable AI for purchase decisions (Model: T-Trace-Multi LR)")

# Data load
if not os.path.exists(ACTIONS_PATH) or not os.path.exists(LEDGER_PATH):
    st.error("Run data generation, model, and logger steps first.")
    st.stop()

actions = pd.read_csv(ACTIONS_PATH)
ledger_records = [json.loads(line) for line in open(LEDGER_PATH, "r", encoding="utf-8")]

# Color & icon dictionary for categories
CAT_META = {
    "fitness": {"color": "#00ff88", "icon": "🏋️"},
    "smartphone": {"color": "#ff8800", "icon": "📱"},
    "gaming": {"color": "#ff00ff", "icon": "🎮"},
    "home_entertainment": {"color": "#00aaff", "icon": "📺"},
    "computer": {"color": "#ffaa00", "icon": "💻"},
}

# Sidebar - User selection
target_users = sorted(set(r["user_id"] for r in ledger_records))
user_id = st.sidebar.selectbox("Select User ID", target_users)

# Filter user records
user_log = [r for r in ledger_records if r["user_id"] == user_id]
actions_u = actions[actions["user_id"] == user_id].copy()
actions_u["timestamp"] = pd.to_datetime(actions_u["timestamp"])
actions_u = actions_u.sort_values("timestamp")

# Ledger hash integrity
def verify_chain(records):
    prev_hash = "0"*64
    for rec in records:
        rec_copy = rec.copy()
        stored_hash = rec_copy.pop("hash")
        payload = json.dumps(rec_copy, sort_keys=True, separators=(",", ":"))
        new_hash = hashlib.sha256((prev_hash + payload).encode()).hexdigest()
        if new_hash != stored_hash:
            return False
        prev_hash = stored_hash
    return True

# Section: Integrity
st.markdown("<div class='section'>🔐 Ledger Integrity Status</div>", unsafe_allow_html=True)
if verify_chain(ledger_records):
    st.success("Ledger Valid — Integrity intact ✔ Blockchain-style proof of transparency")
else:
    st.error("⚠ Ledger appears tampered or corrupted!")

# Section: Timeline view
st.markdown("<div class='section'>🕒 Recent User Activity</div>", unsafe_allow_html=True)
st.dataframe(actions_u.tail(15)[["timestamp", "event_type", "category", "query_text", "product_id"]], height=240)

# Decision selection
st.markdown("<div class='section'>🛍 Purchase Decision Analysis</div>", unsafe_allow_html=True)
decision_ids = [r["decision_id"] for r in user_log]
dec_id = st.selectbox("Select a Purchase Decision", decision_ids)
selected = next(r for r in user_log if r["decision_id"] == dec_id)

product = selected["product_id"]
category = selected["product_category"]
prob = selected["predicted_probability"]
shap_feats = selected["top_shap_features"]
influential_ids = selected["influential_event_ids"]

# Natural explanation text
st.markdown("### 🧾 Explanation Summary (Human-Readable)")
st.markdown(f"""
<div class='explain-box'>
You purchased a **{CAT_META.get(category,{}).get('icon','🛍')} {product}** (category: **{category}**) 
because your previous behavior showed strong interest in:  
<ul>
{''.join([f"<li><b>{feat}</b>: influenced decision by {val:+.2f}</li>" for feat,val in shap_feats.items()])}
</ul>
Model estimated your purchase likelihood as **{prob:.2f}**  
</div>
""", unsafe_allow_html=True)

# Show influential events
infl = actions_u[actions_u["event_id"].isin(influential_ids)]
st.markdown("### 🎯 Specific Actions That Influenced This Decision")
if infl.empty:
    st.info("No strong influencer events could be mapped for this decision.")
else:
    st.dataframe(infl[["timestamp", "event_type", "category", "query_text"]])

# Influence bar chart
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=list(shap_feats.keys()),
    y=list(shap_feats.values()),
    marker=dict(color="#00e6e6")
))
fig_bar.update_layout(
    title="Feature Contribution (SHAP)",
    plot_bgcolor="#050505",
    paper_bgcolor="#050505",
    font=dict(color="white")
)
st.plotly_chart(fig_bar, use_container_width=True)

# Influence Graph
st.markdown("### 🕸 Influence Graph (Action → Purchase)")
G = nx.DiGraph()
G.add_node(dec_id, color="#00ffff", size=32, label=f"PURCHASE\n{category}")

for _, row in infl.iterrows():
    evt = row["event_id"]
    cat = row["category"]
    G.add_node(evt, color=CAT_META.get(cat, {"color": "#aaaaaa"})["color"], size=18,
               label=f"{row['event_type']}\n{cat}")
    G.add_edge(evt, dec_id)

if len(G.nodes) > 1:
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()
    for node,data in G.nodes(data=True):
        x,y = pos[node]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=data["size"], color=data["color"]),
            text=[data["label"]],
            textposition="bottom center"
        ))
    for a,b in G.edges():
        x0,y0 = pos[a]
        x1,y1 = pos[b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(color="#4444ff", width=2)))
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#050505",
        paper_bgcolor="#050505",
        margin=dict(l=20,r=20,t=20,b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough events to show a graph here.")

import os
import json
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import hashlib

ACTIONS_PATH = "data/user_actions.csv"
LEDGER_PATH = "ledger/decision_influence_log.jsonl"

st.set_page_config(page_title="T-Trace Multi-Category Dashboard", layout="wide")

# Dark neon CSS
st.markdown("""
    <style>
        body, .stApp {
            background-color: #050505;
            color: #e0e0e0;
        }
        .title-neon {
            color: #00ffff;
            text-shadow: 0 0 12px #00ffff;
            font-size: 26px;
            font-weight: bold;
        }
        .section {
            color: #76D7C4;
            font-size: 20px;
            margin-top: 24px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-neon'>T-Trace: Multi-Category Behavioral Influence</div>", unsafe_allow_html=True)
st.caption("Explaining purchases (mobile, TV, laptop, console, smartwatch, etc.) from past behavior")

if not (os.path.exists(ACTIONS_PATH) and os.path.exists(LEDGER_PATH)):
    st.error("Run data generation, model, and logger scripts first.")
    st.stop()

actions = pd.read_csv(ACTIONS_PATH)
ledger_records = [json.loads(line) for line in open(LEDGER_PATH, "r", encoding="utf-8")]

if len(ledger_records) == 0:
    st.warning("Ledger is empty: no purchases logged.")
    st.stop()

# Integrity check
def verify_chain(records):
    prev_hash = "0"*64
    for rec in records:
        rec_copy = rec.copy()
        stored_hash = rec_copy.pop("hash")
        payload = json.dumps(rec_copy, sort_keys=True, separators=(",", ":"))
        calc_hash = hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()
        if calc_hash != stored_hash:
            return False
        if rec_copy["prev_hash"] != prev_hash:
            return False
        prev_hash = stored_hash
    return True

chain_ok = verify_chain(ledger_records)

st.markdown("<div class='section'>🔐 Ledger Integrity</div>", unsafe_allow_html=True)
if chain_ok:
    st.success("Hash-linked ledger verified: tamper-evident ✅")
else:
    st.error("Ledger hash chain invalid ❌")

st.write(f"Total logged decisions: {len(ledger_records)}")

# Sidebar selection
user_ids = sorted(set(rec["user_id"] for rec in ledger_records))
selected_user = st.sidebar.selectbox("Select User ID", user_ids)

user_ledger = [rec for rec in ledger_records if rec["user_id"] == selected_user]

st.markdown("<div class='section'>🕒 User Timeline</div>", unsafe_allow_html=True)
user_events = actions[actions["user_id"] == selected_user].copy()
user_events["timestamp"] = pd.to_datetime(user_events["timestamp"])
user_events = user_events.sort_values("timestamp")

st.dataframe(
    user_events.tail(20)[["timestamp", "event_id", "event_type", "category", "query_text", "product_id"]],
    height=260
)

st.markdown("<div class='section'>🧠 Purchase Decision Explanation</div>", unsafe_allow_html=True)
decision_ids = [rec["decision_id"] for rec in user_ledger]
selected_decision_id = st.selectbox("Select Decision ID", decision_ids)

rec = next(r for r in user_ledger if r["decision_id"] == selected_decision_id)

st.write("**Decision ID:**", rec["decision_id"])
st.write("**Product Purchased:**", rec["product_id"])
st.write("**Product Category:**", rec["product_category"])
st.write("**Predicted Purchase Probability:**", f"{rec['predicted_probability']:.3f}")
st.write("**Top Behavioral Influences (SHAP):**")
st.json(rec["top_shap_features"], expanded=False)

# Influential event details
infl_ids = rec["influential_event_ids"]
infl_events = user_events[user_events["event_id"].isin(infl_ids)]

st.write("**Influential Past Events:**")
if infl_events.empty:
    st.info("No influential events found (history too short or low mapping).")
else:
    st.dataframe(
        infl_events[["timestamp", "event_id", "event_type", "category", "query_text", "product_id"]],
        height=260
    )

st.markdown("<div class='section'>🕸 Influence Graph</div>", unsafe_allow_html=True)

G = nx.DiGraph()
G.add_node(rec["decision_id"], color="#00ffff", size=30, label=f"purchase\n{rec['product_category']}")

for _, ev in infl_events.iterrows():
    eid = ev["event_id"]
    col = {
        "fitness": "#00ff88",
        "smartphone": "#ff8800",
        "gaming": "#ff00ff",
        "home_entertainment": "#00aaff",
        "computer": "#ffaa00",
    }.get(ev["category"], "#bbbbbb")

    G.add_node(eid, color=col, size=16, label=f"{ev['event_type']}\n{ev['category']}")
    G.add_edge(eid, rec["decision_id"])

if len(G.nodes) > 1:
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=data["size"], color=data["color"]),
            text=[data["label"]],
            textposition="bottom center"
        ))

    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color="#4444ff", width=2)
        ))

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#050505",
        paper_bgcolor="#050505",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough events to draw an influence graph.")

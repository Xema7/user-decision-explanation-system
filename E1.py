import os
import json
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import hashlib

ACTIONS_PATH = "data/user_actions.csv"
LEDGER_PATH = "ledger/decision_influence_log.jsonl"

st.set_page_config(page_title="T-Trace Explainable Dashboard", layout="wide")

# Dark neon UI style
st.markdown("""
    <style>
        body, .stApp { background-color: #050505; color: #E0E0E0; }
        .title-neon { color: #00FFFF; text-shadow: 0 0 18px #00FFFF; font-size: 26px; font-weight: bold; }
        .section { margin-top: 20px; font-size: 22px; color: #76D7C4; }
        .explain-box { background-color: #111111; padding: 12px; border-radius: 8px; color: #F2F2F2; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-neon'>🧠 T-Trace: Why This Recommendation Appeared</div>", unsafe_allow_html=True)
st.caption("Friendly and transparent AI recommendation explanations")

# Load data
if not (os.path.exists(ACTIONS_PATH) and os.path.exists(LEDGER_PATH)):
    st.error("Run previous scripts first to generate data & ledger.")
    st.stop()

actions = pd.read_csv(ACTIONS_PATH)
ledger_records = [json.loads(line) for line in open(LEDGER_PATH, "r", encoding="utf-8")]

if len(ledger_records) == 0:
    st.warning("No purchases found in ledger.")
    st.stop()

# Category icons
CATEGORY_ICONS = {
    "gaming": "🎮",
    "smartphone": "📱",
    "fitness": "🏋️",
    "computer": "💻",
    "home_entertainment": "📺",
}

# Ledger integrity check
def verify_chain(records):
    prev_hash = "0" * 64
    for rec in records:
        rc = rec.copy()
        stored = rc.pop("hash")
        payload = json.dumps(rc, sort_keys=True, separators=(",", ":"))
        calc = hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()
        if calc != stored: return False
        if rc["prev_hash"] != prev_hash: return False
        prev_hash = stored
    return True

valid_chain = verify_chain(ledger_records)

st.markdown("<div class='section'>🔐 System Trust Check</div>", unsafe_allow_html=True)
st.success("Ledger verified — no tampering detected ✔") if valid_chain else st.error("Tampering detected ❌")

# Sidebar: user select
users = sorted(set(r["user_id"] for r in ledger_records))
selected_user = st.sidebar.selectbox("Select User", users)
user_ledgers = [r for r in ledger_records if r["user_id"] == selected_user]

# Timeline section
st.markdown("<div class='section'>📜 What You Have Been Doing</div>", unsafe_allow_html=True)
ue = actions[actions["user_id"] == selected_user].copy()
ue["timestamp"] = pd.to_datetime(ue["timestamp"])
ue = ue.sort_values("timestamp")

st.dataframe(ue.tail(12)[["timestamp", "event_type", "category", "query_text", "product_id"]])

# Decision explanation section
st.markdown("<div class='section'>🎯 Why You Bought This</div>", unsafe_allow_html=True)

decision_ids = [r["decision_id"] for r in user_ledgers]
sel_dec = st.selectbox("Pick a recent purchase", decision_ids)
rec = next(r for r in user_ledgers if r["decision_id"] == sel_dec)

# Key explanation details
prod = rec["product_id"]
cat = rec["product_category"]
icon = CATEGORY_ICONS.get(cat, "🛍️")
proba = rec["predicted_probability"]
shap_dict = rec["top_shap_features"]

# 💬 Natural language explanation
reason_text = f"""
You purchased **{icon} {prod}** because:

"""

for feat, val in shap_dict.items():
    strength = "🔥 Strong" if abs(val) > 0.25 else "👍 Medium" if abs(val) > 0.12 else "🙂 Small"
    behavior = feat.replace("_events", "").replace("_", " ").title()
    reason_text += f"- {strength} interest in **{behavior}** activities\n"

reason_text += f"\n🤖 The system was **{proba:.0%}** confident you would buy something in this category."

st.markdown(f"<div class='explain-box'>{reason_text}</div>", unsafe_allow_html=True)

# Confidence + Behavior impact chart
impact_df = pd.DataFrame({
    "behavior": list(shap_dict.keys()),
    "impact": list(shap_dict.values())
})
st.bar_chart(impact_df.set_index("behavior"))

# Influence Graph
st.markdown("<div class='section'>🕸 See Which Actions Affected This</div>", unsafe_allow_html=True)

inf_ids = rec["influential_event_ids"]
inf_ev = ue[ue["event_id"].isin(inf_ids)]

G = nx.DiGraph()
G.add_node(sel_dec, label=f"PURCHASE\n{icon}", size=30, color="#00FFFF")

color_map = {
    "fitness": "#00FF88",
    "gaming": "#FF00FF",
    "smartphone": "#FFAA00",
    "computer": "#0099FF",
    "home_entertainment": "#00AAFF"
}

for _, row in inf_ev.iterrows():
    label = row["event_type"].replace("_", " ").title() + f"\n{row['category']}"
    color = color_map.get(row["category"], "#BBBBBB")
    G.add_node(row["event_id"], label=label, size=16, color=color)
    G.add_edge(row["event_id"], sel_dec)

if len(G.nodes) > 1:
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()

    # Draw nodes
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=data["size"], color=data["color"], line=dict(width=2)),
            text=[data["label"]], textposition="bottom center",
            hovertext=f"Action: {data['label']}",
            hoverinfo="text"
        ))

    # Draw influence edges
    for src, dst in G.edges():
        x0, y0 = pos[src]; x1, y1 = pos[dst]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines", line=dict(color="#3050FF", width=2)
        ))

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#050505",
        paper_bgcolor="#050505",
        xaxis=dict(visible=False),  # Hide numbers
        yaxis=dict(visible=False),
        margin=dict(l=20,r=20,t=20,b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough events to display influence graph.")

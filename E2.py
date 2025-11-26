import os
import json
import hashlib

import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

ACTIONS_PATH = "data/user_actions.csv"
LEDGER_PATH = "ledger/decision_influence_log.jsonl"

st.set_page_config(page_title="T-Trace: Multi-Category Behavioral Influence", layout="wide")

# ---------------------- Styling ----------------------
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
        .explanation-box {
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #444444;
            background-color: #101010;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-neon'>T-Trace: Multi-Category Behavioral Influence</div>", unsafe_allow_html=True)
st.caption("Analytical interpretation of how past behavior influences purchase decisions")

# ---------------------- Load data ----------------------
if not (os.path.exists(ACTIONS_PATH) and os.path.exists(LEDGER_PATH)):
    st.error("Run data generation, model, and logger scripts first.")
    st.stop()

actions = pd.read_csv(ACTIONS_PATH)
with open(LEDGER_PATH, "r", encoding="utf-8") as f:
    ledger_records = [json.loads(line) for line in f]

if len(ledger_records) == 0:
    st.warning("Ledger is empty: no logged purchase decisions.")
    st.stop()

# ---------------------- Verify hash chain ----------------------
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
    st.success("Hash-linked ledger verified: records are tamper-evident.")
    st.write(
        "Each decision record is linked to the previous one via a cryptographic hash. "
        "Any modification of historical explanations would invalidate the chain."
    )
else:
    st.error("Ledger hash chain invalid: integrity has been compromised.")

st.write(f"Total logged purchase decisions: **{len(ledger_records)}**")

# ---------------------- Sidebar: user selection ----------------------
user_ids = sorted(set(rec["user_id"] for rec in ledger_records))
selected_user = st.sidebar.selectbox("Select User ID", user_ids)

user_ledger = [rec for rec in ledger_records if rec["user_id"] == selected_user]

# ---------------------- Helper: nice labels for features ----------------------
def feature_to_label(name: str) -> str:
    mapping = {
        "total_events": "overall activity volume",
        "searches": "number of searches",
        "watch_videos": "number of video interactions",
        "read_articles": "article reading activity",
        "compares": "product comparison activity",
        "product_views": "product view activity",
    }
    if name in mapping:
        return mapping[name]
    elif name.endswith("_events"):
        base = name.replace("_events", "")
        return f"interactions in {base} category"
    else:
        return name

# ---------------------- Section: Timeline ----------------------
st.markdown("<div class='section'>🕒 User Timeline (Recent Behavior)</div>", unsafe_allow_html=True)

user_events = actions[actions["user_id"] == selected_user].copy()
user_events["timestamp"] = pd.to_datetime(user_events["timestamp"])
user_events = user_events.sort_values("timestamp")

st.dataframe(
    user_events.tail(20)[["timestamp", "event_id", "event_type", "category", "query_text", "product_id"]],
    height=260
)

# ---------------------- Section: Decision Explanation ----------------------
st.markdown("<div class='section'>🧠 Purchase Decision Explanation</div>", unsafe_allow_html=True)

decision_ids = [rec["decision_id"] for rec in user_ledger]
selected_decision_id = st.selectbox("Select Decision ID", decision_ids)

rec = next(r for r in user_ledger if r["decision_id"] == selected_decision_id)

product = rec["product_id"]
prod_cat = rec.get("product_category", "unknown")
prob = rec["predicted_probability"]
shap_dict = rec["top_shap_features"]

# Build SHAP table sorted by absolute impact (S1)
if shap_dict:
    shap_df = (
        pd.DataFrame(
            [
                {
                    "Feature": k,
                    "Description": feature_to_label(k),
                    "SHAP Value": v,
                    "Abs Impact": abs(v),
                }
                for k, v in shap_dict.items()
            ]
        )
        .sort_values("Abs Impact", ascending=False)
        .reset_index(drop=True)
    )
else:
    shap_df = pd.DataFrame(columns=["Feature", "Description", "SHAP Value", "Abs Impact"])

# --- Analytical narrative explanation ---
st.markdown("**Decision Summary**")
with st.container():
    st.markdown(
        f"<div class='explanation-box'>"
        f"<b>Decision:</b> purchase of <b>{product}</b> (category: <b>{prod_cat}</b>)<br>"
        f"<b>Model-estimated purchase probability:</b> {prob:.3f}<br><br>"
        f"<b>Interpretation:</b> The model inferred elevated purchase intent for this product based on "
        f"user interaction patterns. The most influential behavioral factors are listed below, ordered "
        f"by their absolute contribution to the decision (|SHAP value|). Positive values indicate that "
        f"a feature increased the likelihood of purchase, while negative values indicate a dampening effect."
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("**Top Behavioral Contributors (SHAP-based)**")
st.dataframe(
    shap_df[["Description", "SHAP Value", "Abs Impact"]],
    height=230,
    use_container_width=True
)

# ---------------------- Influential past events ----------------------
st.markdown("<div class='section'>📚 Influential Past Events</div>", unsafe_allow_html=True)

infl_ids = rec.get("influential_event_ids", [])
infl_events = user_events[user_events["event_id"].isin(infl_ids)]

if infl_events.empty:
    st.info("No specific past events were mapped for this decision (history too sparse or not matched).")
else:
    st.write(
        "The following concrete user actions (searches, views, comparisons, etc.) "
        "are associated with the most influential behavioral features above:"
    )
    st.dataframe(
        infl_events[["timestamp", "event_id", "event_type", "category", "query_text", "product_id"]],
        height=260,
        use_container_width=True
    )

# ---------------------- Influence Graph ----------------------
st.markdown("<div class='section'>🕸 Influence Graph (Conceptual)</div>", unsafe_allow_html=True)

# Build graph: influential actions -> decision node
G = nx.DiGraph()
center_label = f"purchase\n{prod_cat}"
G.add_node(rec["decision_id"], color="#00ffff", size=32, label=center_label)

color_map = {
    "fitness": "#00ff88",
    "smartphone": "#ff8800",
    "gaming": "#ff00ff",
    "home_entertainment": "#00aaff",
    "computer": "#ffaa00",
}

for _, ev in infl_events.iterrows():
    eid = ev["event_id"]
    cat = ev["category"]
    node_color = color_map.get(cat, "#bbbbbb")
    label = f"{ev['event_type']}\n{cat}"
    G.add_node(eid, color=node_color, size=18, label=label)
    G.add_edge(eid, rec["decision_id"])

# Draw if there is more than just the center node
if len(G.nodes) > 1:
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()

    # Nodes
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=data["size"], color=data["color"]),
                text=[data["label"]],
                textposition="bottom center",
                hovertext=[f"Node: {node}"],
                hoverinfo="text"
            )
        )

    # Edges
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="#4444ff", width=2),
                hoverinfo="none"
            )
        )

    # Hide numeric axes (they have no semantic meaning)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#050505",
        paper_bgcolor="#050505",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough mapped events to construct an influence graph.")

# ---------------------- Category Legend ----------------------
st.markdown("<div class='section'>🎛 Category Legend</div>", unsafe_allow_html=True)
st.markdown(
    """
- 🟢 **Fitness**  
- 🟠 **Smartphone**  
- 💜 **Gaming**  
- 🔵 **Home Entertainment**  
- 🟡 **Computer**  
- ⚪ Other / Unclassified
"""
)

from langgraph.graph import StateGraph, END
from app.state import AppState

from app.agents.router_agent import router_agent
from app.agents.gap_agent import gap_agent
from app.agents.risk_agent import risk_assessment_agent, risk_materiality_agent


def router_edge(state: AppState):
    # DROP_GAP ends (log only)
    # KEEP_GAP -> gap_agent -> risk
    # NO_GAP_HIGH_RISK -> skip gap_agent -> risk
    if state.gap_route == "DROP_GAP":
        return "DROP"
    if state.gap_route == "KEEP_GAP":
        return "KEEP"
    return "NO_GAP_HIGH_RISK"


def risk_edge(state: AppState):
    return "KEEP_RISK" if state.risk_route == "KEEP_RISK" else "DROP_RISK"


def log_and_end(state: AppState):
    # ---- ROUTER → FINAL STATUS MAPPING ----
    if state.gap_route == "KEEP_GAP":
        final_status = "GAP"
    elif state.gap_route == "NO_GAP_HIGH_RISK":
        final_status = "RISK"
    elif state.gap_route == "DROP_GAP":
        final_status = "DROPPED"
    else:
        final_status = "UNKNOWN"
    # -------------------------------------

    state.audit_log.append({
        "requirement": state.requirement,
        "gap_route": state.gap_route,
        "gap_confidence": state.gap_confidence,
        "gap_reason": state.gap_reason,
        "gap_summary": state.gap_summary,
        "gap_severity": state.gap_severity,
        "risk_route": state.risk_route,
        "risk_confidence": state.risk_confidence,
        "risk_reason": state.risk_reason,
        "risk_statement": state.risk_statement,
        "rating": state.rating,
        "status": final_status,
    })
    return state


def drop_gap_node(state: AppState):
    return log_and_end(state)


def keep_risk_node(state: AppState):
    return log_and_end(state)


def drop_risk_node(state: AppState):
    return log_and_end(state)


graph = StateGraph(AppState)

graph.add_node("router", router_agent)
graph.add_node("gap", gap_agent)
graph.add_node("risk_assess", risk_assessment_agent)
graph.add_node("risk_materiality", risk_materiality_agent)

graph.add_node("drop_gap", drop_gap_node)
graph.add_node("keep_risk", keep_risk_node)
graph.add_node("drop_risk", drop_risk_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    router_edge,
    {
        "DROP": "drop_gap",
        "KEEP": "gap",
        "NO_GAP_HIGH_RISK": "risk_assess",
    }
)

graph.add_edge("gap", "risk_assess")
graph.add_edge("risk_assess", "risk_materiality")

graph.add_conditional_edges(
    "risk_materiality",
    risk_edge,
    {
        "KEEP_RISK": "keep_risk",
        "DROP_RISK": "drop_risk",
    }
)

graph.add_edge("drop_gap", END)
graph.add_edge("keep_risk", END)
graph.add_edge("drop_risk", END)

app = graph.compile()

# 1. Framework Imports
from langgraph.graph import StateGraph, END

# 2. Project Imports (Ensuring they match your folder structure)
from app.state import AppState
from app.agents.router_agent import router_agent
from app.agents.gap_agent import gap_agent
from app.agents.risk_agent import risk_assessment_agent, risk_materiality_agent

# --- STEP 1: THE RECORDER (Finalizing the State) ---
def finalize_and_log(state: AppState):
    """
    This function acts as the archiver. It takes the notes from the 'Clipboard' (State)
    and saves them into the permanent 'Audit Log'.
    """
    # Map the technical routes to user-friendly status labels
    status_map = {
        "KEEP_GAP": "Non-Compliant (Gap)", 
        "NO_GAP_HIGH_RISK": "Compliant but Risky", 
        "DROP_GAP": "Out of Scope"
    }
    
    final_status = status_map.get(state.gap_route, "Unknown")

    # We append a dictionary to the list in AppState.audit_log
    # These keys MUST match what you use in your result.html table
    state.audit_log.append({
        "clause": state.requirement,          # From Step 1
        "status": final_status,               # Determined above
        "gap_summary": state.gap_summary,     # From Step 3 (Gap Agent)
        "risk_rating": state.rating,          # From Step 4 (Risk Agent)
        "risk_details": state.risk_statement, # From Step 4 (Risk Agent)
        "recommendation": state.gap_recommendation # From Step 3 (Gap Agent)
    })
    
    return state

# --- STEP 2: TRAFFIC CONTROL (Conditional Logic) ---
def route_after_router(state: AppState):
    """Decides the first path based on the Inspector's initial scan."""
    if state.gap_route == "DROP_GAP": 
        return "FINISH"
    if state.gap_route == "KEEP_GAP": 
        return "AUDIT"
    return "RISK_ONLY"

def route_after_risk(state: AppState):
    """Decides if the final risk is significant enough to be logged."""
    # Matches the RiskRoute Literal ["KEEP_RISK", "DROP_RISK"] in state.py
    return "LOG" if state.risk_route == "KEEP_RISK" else "LOG" # We log both for audit transparency

# --- STEP 3: CONSTRUCTING THE GRAPH ---
builder = StateGraph(AppState)

# Define the Nodes (The "Workstations")
builder.add_node("inspector", router_agent)
builder.add_node("gap_auditor", gap_agent)
builder.add_node("risk_expert", risk_assessment_agent)
builder.add_node("impact_analyst", risk_materiality_agent)
builder.add_node("logger", finalize_and_log)

# --- STEP 4: DEFINING THE FLOW ---
builder.set_entry_point("inspector")

# Branching from the Inspector
builder.add_conditional_edges("inspector", route_after_router, {
    "FINISH": "logger",
    "AUDIT": "gap_auditor",
    "RISK_ONLY": "risk_expert"
})

# Sequential movement: Audit -> Risk Check -> Impact Analysis
builder.add_edge("gap_auditor", "risk_expert")
builder.add_edge("risk_expert", "impact_analyst")

# Final check before closing the file
builder.add_conditional_edges("impact_analyst", route_after_risk, {
    "LOG": "logger"
})

# The logger always leads to the END of the process
builder.add_edge("logger", END)

# Compile the graph into an executable app
app = builder.compile()
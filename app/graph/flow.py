# 1. Framework Imports
from langgraph.graph import StateGraph, END

# 2. Project Imports (Ensuring they match your folder structure)
from app.state import AppState
from app.agents.router_agent import router_agent
from app.agents.gap_agent import gap_agent
from app.agents.risk_agent import risk_assessment_agent

# --- STEP 1: THE RECORDER (Finalizing the State) ---
def finalize_and_log(state: AppState):
    """
    This function acts as the archiver. It maps internal AI state to 
    the specific keys expected by the frontend results table.
    """
    
    # 1. Cleanup for Out-of-Scope items (DROP_GAP)
    if state.gap_route == "DROP_GAP":
        state.gap_status = "Out of Scope"
        state.gap_summary = "Out of Scope"
        state.gap_recommendation = "Out of Scope"
        state.source_ref = "Out of Scope"
        state.rating = "Out of Scope"
        state.risk_statement = "Out of Scope"
        state.recommended_control = "Out of Scope"

    elif state.gap_route == "NO_GAP_HIGH_RISK":
        # If the Risk Agent forgot to set the status, we ensure it's set here
        state.gap_status = state.gap_status or "Compliant but Risky"
        
    elif state.gap_route == "KEEP_GAP":
        state.gap_status = state.gap_status or "Non-Compliant (Gap)"

    # 2. Append to Audit Log using EXACT frontend keys
    # Note: Using .get() or 'or' ensures we never pass 'None' to the UI
    state.audit_log.append({
        "theme": getattr(state, "theme", "General Compliance"), # Default if theme isn't in AppState
        "clause": state.requirement,
        "source_ref": state.source_ref or "Not Explicitly Stated",
        "status": state.gap_status or "Not Applicable",
        "gap_summary": state.gap_summary or "Not Applicable",
        "risk_rating": state.rating or "Not Applicable",
        "risk_statement": state.risk_statement or "Not Applicable",
        "risk_recommendation": state.recommended_control or "Not Applicable"
    })

    return state

# --- STEP 2: CONSTRUCTING THE GRAPH ---
builder = StateGraph(AppState)

# Define the Nodes (The "Workstations")
builder.add_node("router_agent", router_agent)
builder.add_node("gap_auditor", gap_agent)
builder.add_node("risk_expert", risk_assessment_agent)
builder.add_node("logger", finalize_and_log)

# Graph Entry Point
builder.set_entry_point("router_agent")

# Branching from the Router Agent 
builder.add_conditional_edges("router_agent", 
                              lambda state: state.gap_route, 
                             {
                                "DROP_GAP": "logger",
                                "KEEP_GAP": "gap_auditor",
                                "NO_GAP_HIGH_RISK": "risk_expert"
                             }
)

# Sequential movement
builder.add_edge("gap_auditor", "risk_expert")
builder.add_edge("risk_expert", "logger")

# The logger always leads to the END of the process
builder.add_edge("logger", END)

# Compile the graph into an executable app
app = builder.compile()
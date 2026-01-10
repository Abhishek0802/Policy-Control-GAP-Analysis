from langgraph.graph import StateGraph, END
from app.state import AppState
from app.agents.router_agent import route_gap
from app.agents.risk_agent import route_risk

graph = StateGraph(AppState)

graph.add_node("route_gap", route_gap)
graph.add_node("route_risk", route_risk)

graph.set_entry_point("route_gap")

graph.add_edge("route_gap", "route_risk")
graph.add_edge("route_risk", END)

app = graph.compile()

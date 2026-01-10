from app.graph.flow import app
from app.state import AppState

state = AppState(
    requirement="Incident response plan must define SLA",
    evidence="Policy mentions reporting incidents but no SLA"
)

result = app.invoke(state)
print(result)

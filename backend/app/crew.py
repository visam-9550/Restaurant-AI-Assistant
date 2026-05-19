from app.agents import intent_agent
from app.tasks import intent_task
from crewai import Crew


intent_crew = Crew(
    name="Intent Crew",
    agents=[intent_agent],
    tools=[],
    tasks=[intent_task]
)
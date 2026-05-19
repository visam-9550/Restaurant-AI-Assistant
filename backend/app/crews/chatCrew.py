from crewai import Crew, Agent, Task, LLM
from app.agents import chat_agent
from app.tasks import chat_task

chat_crew = Crew(
    name="Chat Crew",
    agents=[chat_agent],
    tools=[],
    tasks=[chat_task]
)
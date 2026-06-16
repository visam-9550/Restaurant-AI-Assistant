from app.agents import intent_agent, faq_agent, menu_agent, menu_parser_agent, order_agent, greeting_agent
from app.tasks import intent_task, faq_task, menu_task, menu_parser_task, order_task, greeting_task
from crewai import Crew


menu_parser_crew = Crew(
    name="Menu Parser Crew",
    agents=[menu_parser_agent],
    tasks=[menu_parser_task],
    cache=False
)

intent_crew = Crew(
    name="Intent Crew",
    agents=[intent_agent],
    tasks=[intent_task],
    cache=False
)

faq_crew = Crew(
    name="FAQ Crew",
    agents=[faq_agent],
    tasks=[faq_task],
    cache=False
)

greeting_crew = Crew(
    name="Greeting Crew",
    agents=[greeting_agent],
    tasks=[greeting_task],
    cache=False
)

menu_crew = Crew(
    name="Menu Crew",
    agents=[menu_agent],
    tasks=[menu_task],
    cache=False
)

order_crew = Crew(
    name="Order Crew",
    agents=[order_agent],
    tasks=[order_task],
    cache=False
)



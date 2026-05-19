from crewai import Task
from app.agents import chat_agent, intent_agent

chat_task = Task(
    agent=chat_agent,
    name="Chat with Restaurant Owner",
    description="Engage in a conversation with a restaurant owner, providing assistance and answering questions related to restaurant management.",
    expected_output="Provide helpful and informative responses to the restaurant owner's questions and requests."
)


intent_task = Task(
    name="Intent Classification",
    agent=intent_agent,
    description="""
    Analyze the following restaurant user query
    and classify the user's intent.

    User Query: {input}

    Return ONLY valid JSON.

    Available intents:

    - order_food
    - modify_cart
    - menu_search
    - menu_price_query
    - faq
    - restaurant_policy
    - track_order
    - cancel_order
    - repeat_order
    - greeting

    Rules:

    - Focus ONLY on identifying the user's goal.
    - Do NOT answer the user.
    - Do NOT explain reasoning.
    - Do NOT extract entities.
    - Return structured JSON only.

    Example Output:

    {{
      "intent": "order_food"
    }}
    """,
    expected_output="JSON object with intent classification"
)
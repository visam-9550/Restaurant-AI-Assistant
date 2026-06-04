from crewai import Task
from app.agents import chat_agent, intent_agent, faq_agent, menu_agent, order_agent, menu_parser_agent

chat_task = Task(
    agent=chat_agent,
    name="Chat with Restaurant Owner",
    description="Engage in a conversation with a restaurant owner, providing assistance and answering questions related to restaurant management.",
    expected_output="Provide helpful and informative responses to the restaurant owner's questions and requests."
)


menu_parser_task = Task(

    name="Restaurant Menu Parsing Task",

    agent=menu_parser_agent,

    description="""
    Convert the following restaurant menu text
    into a valid JSON array.

    MENU TEXT:
    {menu_text}

    EXTRACTION REQUIREMENTS:

    - extract all menu items
    - extract quantities
    - extract prices
    - extract categories
    - detect veg/non_veg

    STRICT OUTPUT RULES:

    - return ONLY JSON
    - return ONLY JSON array
    - no markdown
    - no ```json
    - no explanations
    - no comments
    - no extra text
    - no headings
    - no notes
    - output must start with [
    - output must end with ]

    REQUIRED OUTPUT FORMAT:

    [
        {
            "food_item_name": "Chicken Biryani",
            "quantity": "850 gm",
            "price": 289,
            "category": "biryani",
            "food_type": "non_veg"
        }
    ]
    """,

    expected_output="""
[
    {
        "food_item_name": "Chicken Biryani",
        "quantity": "850 gm",
        "price": 289,
        "category": "biryani",
        "food_type": "non_veg"
    }
]
"""
)


intent_task = Task(

    name="Restaurant Intent Classification Task",

    agent=intent_agent,

    description="""
    Analyze the following restaurant-related
    customer message and identify ONLY
    the SINGLE most accurate intent.

    ------------------------------------------------

    USER MESSAGE:
    {input}

    ------------------------------------------------

    SUPPORTED INTENTS:

    - greeting
    - gratitude
    - acknowledgement

    - menu_search
    - menu_price_query
    - menu_availability_query
    - food_recommendation

    - order_food
    - modify_cart
    - remove_cart_item
    - clear_cart

    - confirm_order
    - cancel_order
    - repeat_order
    - track_order

    - faq
    - restaurant_policy
    - restaurant_timings
    - parking_enquiry
    - delivery_enquiry
    - payment_issue
    - refund_request

    - offers_and_discounts
    - table_booking
    - complaint
    - feedback

    - contact_support
    - unknown

    ------------------------------------------------

    CLASSIFICATION RULES:

    - Return ONLY ONE intent
    - Select MOST relevant intent
    - Do NOT explain reasoning
    - Do NOT answer user
    - Do NOT add extra text
    - Do NOT use markdown
    - Do NOT use ```json
    - Do NOT use code blocks
    - Do NOT generate comments
    - Do NOT add confidence score

    ------------------------------------------------

    STRICT OUTPUT RULE:

    Your ENTIRE response MUST be EXACTLY
    ONE valid raw JSON object.

    ------------------------------------------------

    INVALID OUTPUT EXAMPLES:

    ```json
    {
      "intent": "menu_search"
    }
    ```

    INVALID because markdown exists.

    ------------------------------------------------

    INVALID OUTPUT:

    The intent is:

    {
      "intent": "menu_search"
    }

    INVALID because extra text exists.

    ------------------------------------------------

    VALID OUTPUT:

    {"intent":"menu_search"}

    ------------------------------------------------

    MORE VALID OUTPUT EXAMPLES:

    {"intent":"greeting"}

    {"intent":"food_recommendation"}

    {"intent":"order_food"}

    {"intent":"cancel_order"}

    ------------------------------------------------

    IMPORTANT:

    Output MUST be directly parsable using:

    json.loads(response)

    without any preprocessing.

    ------------------------------------------------

    RESPONSE MUST CONTAIN:

    - ONLY JSON
    - ONLY one key:
      "intent"

    ------------------------------------------------

    RESPONSE FORMAT:

    {"intent":"<intent_name>"}
    """,

    expected_output='{"intent":"menu_search"}'
)

faq_task = Task(
    description="""
    Handle restaurant-related FAQ queries accurately and professionally.

    Responsibilities:
    - Answer restaurant timings
    - Explain restaurant policies
    - Explain cancellation/refund policies
    - Explain delivery availability
    - Explain payment methods
    - Explain table reservation rules
    - Explain offers and discounts
    - Explain parking availability
    - Explain veg/non-veg availability
    - Explain takeaway and dine-in services

    Instructions:
    - Always provide concise and clear responses
    - Use restaurant knowledge base and RAG data
    - Never hallucinate policies or timings
    - If information is unavailable, politely mention it
    - Maintain professional customer support tone
    - Prioritize factual accuracy over creativity

    Example Queries:
    - What are your restaurant timings?
    - Do you provide home delivery?
    - What is your cancellation policy?
    - Is parking available?
    - Are pets allowed?
    - Do you have vegetarian options?
    - Can I reserve a table?
    - Which payment methods are accepted?

    Current Customer Query:
    {input}
    """,

    expected_output="""
    A professional and accurate FAQ response answering
    the customer's restaurant-related query clearly.
    """,

    agent=faq_agent
)

menu_task = Task(

    name="Personalized Restaurant Menu and Food Recommendation Task",

    agent=menu_agent,

    description="""
    Analyze the customer restaurant query
    and intelligently retrieve:

    - restaurant menu information
    - customer food preferences

    using the available tools.

    ------------------------------------------------

    CUSTOMER QUERY:
    {message}

    ------------------------------------------------

    CUSTOMER USER ID:
    {user_id}

    ------------------------------------------------

    YOUR RESPONSIBILITIES:

    - search restaurant menu data
    - retrieve customer food preferences
    - provide personalized food recommendations
    - show complete menu listings
    - recommend suitable dishes
    - explain food prices and quantities
    - suggest beverages and desserts
    - recommend restaurant specials
    - improve customer food discovery experience

    ------------------------------------------------

    VERY IMPORTANT INSTRUCTIONS:

    - ALWAYS use the menu search tool
    - IF user asks for:
        - recommendations
        - suggestions
        - personalized foods
        - what should I eat
        - best food for me

      THEN:
      ALWAYS use the
      user_preferences_search_tool

    ------------------------------------------------

    RECOMMENDATION LOGIC:

    Use customer preferences such as:

    - favorite cuisines
    - diet type
    - favorite foods
    - spice level
    - favorite beverages
    - budget preference
    - ordering history

    to recommend suitable menu items.

    ------------------------------------------------

    RESPONSE RULES:

    IF USER ASKS:

    - "show full menu"
      → show ALL retrieved categories

    - "recommend food"
      → use user preferences +
        menu search results

    - "suggest dinner"
      → recommend personalized dinner items

    - "show veg food"
      → recommend vegetarian dishes

    - "show biryanis"
      → show ALL biryani items

    ------------------------------------------------

    RESPONSE FORMAT:

    Recommendation Example:

    Recommended For You:

    1. Chicken Dum Biryani
       - Quantity: 850 gm
       - Price: ₹320

    Reason:
    Based on your preference for
    spicy non-veg biryanis.

    ------------------------------------------------

    IMPORTANT:

    - show MULTIPLE menu items
    - show COMPLETE category results
    - provide personalized suggestions
    - keep formatting clean
    - keep response readable
    - NEVER hallucinate recommendations

    ------------------------------------------------

    RESPONSE STYLE:

    - professional
    - personalized
    - conversational
    - restaurant assistant tone
    - customer-friendly

    ------------------------------------------------

    EXAMPLE USER QUERIES:

    "Show full menu"

    "Recommend food for me"

    "Suggest dinner"

    "What should I eat?"

    "Show all biryanis"

    "Recommend based on my preferences"

    ------------------------------------------------

    FINAL OUTPUT REQUIREMENTS:

    - conversational response only
    - no JSON
    - no markdown
    - no technical explanations
    - no hallucinations
    - complete menu visibility
    - personalized recommendations
    - structured formatting
    """,

    expected_output="""
    A complete and personalized
    restaurant menu response containing:

    - menu categories
    - food item names
    - prices
    - quantities
    - personalized recommendations
    - beverages
    - desserts
    - restaurant specials

    using ONLY retrieved menu data
    and retrieved customer preferences.
    """
)

order_task = Task(

    name="Restaurant Food Order Extraction Task",

    agent=order_agent,

    description="""
    Analyze the following customer
    restaurant ordering query and extract:

    - ordered food items
    - quantities
    - price for one quantity

    ------------------------------------------------

    CUSTOMER QUERY:
    {input}

    ------------------------------------------------

    EXTRACTION RULES:

    - identify ALL ordered food items
    - extract quantities for every item
    - if quantity missing:
        quantity = 1

    ------------------------------------------------

    FOOD EXTRACTION EXAMPLES:

    USER:
    "I want chicken biryani"

    OUTPUT:
    [
        {
            "food_item_name": "Chicken Biryani",
            "quantity": 1,
            "price": 289
        }
    ]

    ------------------------------------------------

    USER:
    "Add 2 chicken biryanis and 1 coke"

    OUTPUT:
    [
        {
            "food_item_name": "Chicken Biryani",
            "quantity": 2,
            "price": 289
        },
        {
            "food_item_name": "Coke",
            "quantity": 1,
            "price": 45
        }
    ]

    ------------------------------------------------

    USER:
    "Give me 3 pizzas"

    OUTPUT:
    [
        {
            "food_item_name": "Pizza",
            "quantity": 3,
            "price": 499
        }
    ]

    ------------------------------------------------

    IMPORTANT RULES:

    - return ONLY raw JSON
    - return ONLY JSON array
    - no markdown
    - no ```json
    - no explanations
    - no comments
    - no headings
    - no notes
    - no extra text
    - output must start with [
    - output must end with ]

    ------------------------------------------------

    OUTPUT FORMAT:

    [
        {
            "food_item_name": "Chicken Biryani",
            "quantity": 2,
            "price": 289
        }
    ]

    ------------------------------------------------

    OUTPUT MUST BE DIRECTLY PARSEABLE USING:

    json.loads(response)
    """,

    expected_output="""
[
    {
        "food_item_name": "Chicken Biryani",
        "quantity": 2,
        "price": 289
    },
    {
        "food_item_name": "Coke",
        "quantity": 1,
        "price": 45
    }
]
"""
)



# order_task = Task(
# name="Menu Validation Task",

# agent=order_agent,

# description="""
# Validate the following extracted order items.

# ORDER ITEMS:

# {order_items}

# ------------------------------------------------

# Validation Rules

# Exact Match:

# - item_confirmed=true
# - availability_status=available

# Similar Match:

# - closest menu item found
# - item_confirmed=false
# - availability_status=similar_item_found

# Multiple Matches:

# Example:
# User requested: Dosa

# Menu:
# - Plain Dosa
# - Masala Dosa
# - Ghee Karam Dosa

# Output:

# - item_confirmed=false
# - availability_status=multiple_matches_found
# - return suggested_items

# Not Available:

# - item_confirmed=false
# - availability_status=not_available

# ------------------------------------------------

# Required Output

# [
#     {
#         "food_item_name": "",
#         "matched_item": null,
#         "quantity": 1,
#         "price": null,
#         "item_confirmed": false,
#         "availability_status": "",
#         "suggested_items": []
#     }
# ]

# ------------------------------------------------

# Rules

# - JSON array only
# - No markdown
# - No explanations
# - No comments
# - No extra text
# - Output must start with [
# - Output must end with ]
# """,

# expected_output="""
# ```

# [
# {
# "food_item_name": "Chicken Biryani",
# "matched_item": "Chicken Biryani",
# "quantity": 2,
# "price": 289,
# "item_confirmed": true,
# "availability_status": "available",
# "suggested_items": []
# }
# ]
# """
# )

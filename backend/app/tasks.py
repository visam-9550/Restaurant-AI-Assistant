from crewai import Task
from app.agents import chat_agent, intent_agent, faq_agent, menu_agent, order_agent, menu_parser_agent, greeting_agent

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

# menu_task = Task(

#     name="Personalized Restaurant Menu and Food Recommendation Task",

#     agent=menu_agent,

#     description="""
#     Analyze the customer restaurant query
#     and intelligently retrieve:

#     - restaurant menu information
#     - customer food preferences

#     using the available tools.

#     ------------------------------------------------

#     CUSTOMER QUERY:
#     {message}

#     ------------------------------------------------

#     CUSTOMER USER ID:
#     {user_id}

#     ------------------------------------------------

#     YOUR RESPONSIBILITIES:

#     - search restaurant menu data
#     - retrieve customer food preferences
#     - provide personalized food recommendations
#     - show complete menu listings
#     - recommend suitable dishes
#     - explain food prices and quantities
#     - suggest beverages and desserts
#     - recommend restaurant specials
#     - improve customer food discovery experience

#     ------------------------------------------------

#     VERY IMPORTANT INSTRUCTIONS:

#     - ALWAYS use the menu search tool
#     - IF user asks for:
#         - recommendations
#         - suggestions
#         - personalized foods
#         - what should I eat
#         - best food for me

#       THEN:
#       ALWAYS use the
#       user_preferences_search_tool

#     ------------------------------------------------

#     RECOMMENDATION LOGIC:

#     Use customer preferences such as:

#     - favorite cuisines
#     - diet type
#     - favorite foods
#     - spice level
#     - favorite beverages
#     - budget preference
#     - ordering history

#     to recommend suitable menu items.

#     ------------------------------------------------

#     RESPONSE RULES:

#     IF USER ASKS:

#     - "show full menu"
#       → show ALL retrieved categories

#     - "recommend food"
#       → use user preferences +
#         menu search results

#     - "suggest dinner"
#       → recommend personalized dinner items

#     - "show veg food"
#       → recommend vegetarian dishes

#     - "show biryanis"
#       → show ALL biryani items

#     ------------------------------------------------

#     RESPONSE FORMAT:

#     Recommendation Example:

#     Recommended For You:

#     1. Chicken Dum Biryani
#        - Quantity: 850 gm
#        - Price: ₹320

#     Reason:
#     Based on your preference for
#     spicy non-veg biryanis.

#     ------------------------------------------------

#     IMPORTANT:

#     - show MULTIPLE menu items
#     - show COMPLETE category results
#     - provide personalized suggestions
#     - keep formatting clean
#     - keep response readable
#     - NEVER hallucinate recommendations

#     ------------------------------------------------

#     RESPONSE STYLE:

#     - professional
#     - personalized
#     - conversational
#     - restaurant assistant tone
#     - customer-friendly

#     ------------------------------------------------

#     EXAMPLE USER QUERIES:

#     "Show full menu"

#     "Recommend food for me"

#     "Suggest dinner"

#     "What should I eat?"

#     "Show all biryanis"

#     "Recommend based on my preferences"

#     ------------------------------------------------

#     FINAL OUTPUT REQUIREMENTS:

#     - conversational response only
#     - no JSON
#     - no markdown
#     - no technical explanations
#     - no hallucinations
#     - complete menu visibility
#     - personalized recommendations
#     - structured formatting
#     """,

#     expected_output="""
#     A complete and personalized
#     restaurant menu response containing:

#     - menu categories
#     - food item names
#     - prices
#     - quantities
#     - personalized recommendations
#     - beverages
#     - desserts
#     - restaurant specials

#     using ONLY retrieved menu data
#     and retrieved customer preferences.
#     """
# )


menu_task = Task(
name="Restaurant Menu Search and Recommendation Task",

agent=menu_agent,

description="""

CUSTOMER QUERY

{message}

---

USER ID

{user_id}

---

CHAT HISTORY

{chat_history}

---

TASK

Analyze the user query and determine:

1. Menu Search
2. Menu Price Query
3. Food Recommendation
4. Personalized Recommendation
5. Follow-up Menu Query

---

MENU SEARCH RULE

Examples:

* show menu
* show full menu
* show biryanis
* show drinks
* show desserts
* what food do you have
* what is available

Use:

menu_search_tool ONLY

---

PRICE QUERY RULE

Examples:

* price of chicken biryani
* cost of coke
* menu prices
* how much is veg meals

Use:

menu_search_tool ONLY

Return retrieved prices.

---

RECOMMENDATION RULE

Examples:

* recommend food
* suggest dinner
* what should I eat
* best food for me
* recommend based on my preferences

Use:

1. menu_search_tool
2. user_preferences_search_tool

---

FOLLOW-UP RULE

Use chat history to understand:

* which one
* recommend one
* show more
* what about drinks
* any desserts
* what is its price

---

DIRECT RESPONSE RULE

Answer exactly what the user asks.

Examples:

User:
Show biryanis

Return:
Biryani items only.

---

User:
Price of Chicken Biryani

Return:
Chicken Biryani price only.

---

User:
Recommend dinner

Return:
Personalized recommendations.

---

OUTPUT RULES

* conversational response only
* no JSON
* no markdown
* no technical explanations
* include retrieved prices
* include retrieved quantities
* answer user query directly
* use chat history when relevant
* use preferences only when recommendation is requested
* use retrieved data only
  """,

  expected_output="""
  A clean conversational restaurant response
  containing only relevant menu information,
  menu prices, menu categories, or personalized
  recommendations based on the user's query.

All prices and menu items must come from
retrieved menu data only.
"""
)



# order_task = Task(
# name="Restaurant Order Understanding",

# agent=order_agent,

# description="""

# RESTAURANT ID

# {restaurant_id}

# ---

# CHAT HISTORY

# {chat_history}

# ---

# LAST MENU RESPONSE

# {last_menu_response}

# ---

# USER QUERY

# {input}

# ---

# PROCESS

# 1. Search menu using menu_search_tool.

# 2. Extract all requested items.

# 3. Extract quantities.

# 4. Quantity defaults to 1.

# 5. Normalize item names.

# 6. Merge duplicate items.

# 7. Retrieve actual menu prices.

# 8. Evaluate each item independently.

# ---

# EXACT MATCH

# If the user explicitly requests a menu item
# and an exact menu match exists:

# item_confirmed = true

# Example:

# User:
# "I want Chicken Dum Biryani"

# Result:

# Chicken Dum Biryani
# item_confirmed = true

# ---

# CATEGORY MATCH

# Examples:

# * biryani
# * pizza
# * dessert
# * desserts
# * drink
# * drinks
# * starter
# * starters
# * combo
# * meal
# * meals

# If multiple menu items match:

# requires_clarification = true

# clarification_type =
# "category_selection"

# Add matching menu items into:

# available_options

# DO NOT automatically select:

# * first result
# * cheapest result
# * highest priced result
# * most popular result

# Example:

# User:
# "I want 3 biryanis"

# Menu:

# * Chicken Dum Biryani
# * Veg Biryani
# * Mutton Biryani

# Return clarification.

# Do NOT return Chicken Dum Biryani.

# ---

# CRITICAL NEVER GUESS RULE

# Never convert:

# * biryani
# * pizza
# * dessert
# * drink
# * starter
# * combo
# * meal

# into a specific menu item.

# Only return a specific menu item when:

# 1. User explicitly requested it

# OR

# 2. Only one exact menu item exists.

# ---

# SIMILAR MATCH

# If exact item does not exist
# but similar menu items exist:

# requires_clarification = true

# clarification_type =
# "similar_item"

# Return similar options in:

# available_options

# Example:

# User:
# "chiken biriyani"

# Menu:

# Chicken Biryani

# Return similar_item clarification.

# ---

# ITEM NOT FOUND

# If no exact match exists
# and no similar item exists:

# Add item into:

# unavailable_items

# Example:

# {
# "food_item_name": "jjjj",
# "quantity": 5,
# "reason": "item_not_found"
# }

# ---

# MULTI ITEM VALIDATION

# Evaluate EVERY requested item independently.

# Example:

# User:
# "I want 3 biryanis and 5 jjjj"

# Evaluation:

# biryanis
# → category_selection

# jjjj
# → item_not_found

# Never allow one item's result
# to affect another item's result.

# ---

# STATUS RULES

# success

# All requested items were found
# through exact matches.

# ---

# partial_success

# Some items were found and
# some items were unavailable.

# ---

# clarification_required

# One or more requested items
# require clarification.

# ---

# item_not_found

# No valid menu items found.

# ---

# FOLLOW-UP REFERENCES

# Use:

# * chat_history
# * last_menu_response

# to understand:

# * it
# * that
# * those
# * first one
# * second one
# * same
# * same again
# * spicy one
# * cheapest one

# Only resolve references when
# sufficient context exists.

# Otherwise return clarification.

# ---

# IGNORE

# * cancel order
# * modify order
# * update order
# * remove item
# * increase quantity
# * decrease quantity
# * confirm order
# * checkout

# ---

# OUTPUT VALIDATION CHECK

# Before generating output verify:

# 1. No category was converted into a specific menu item.

# 2. item_confirmed=true only for exact matches.

# 3. Every requested item is accounted for.

# 4. unavailable_items contains only unavailable items.

# 5. clarification_items contains only ambiguous items.

# 6. available_options is populated whenever
#    clarification is required.

# ---

# OUTPUT RULES

# Return ONLY JSON.

# No markdown.

# No explanations.

# No comments.

# No extra text.
# """,
# expected_output="""

# {
# "status": "partial_success",
# "items": [
#     {
#         "food_item_name": "Chicken Dum Biryani",
#         "quantity": 4,
#         "price": 320,
#         "item_confirmed": true,
#         "unavailable": false
#     }
# ],

# "unavailable_items": [
#     {
#         "food_item_name": "jjdjjdjdjdj",
#         "quantity": 3,
#         "reason": "item_not_found"
#     }
# ],

# "clarification_items": [],

# "requires_clarification": false,

# "clarification_type": null,

# "clarification_question": null,

# "available_options": []

# }
# """
# )


order_task = Task(
name="Restaurant Order Understanding",
agent=order_agent,

description="""
RESTAURANT ID

{restaurant_id}

---

CHAT HISTORY

{chat_history}

---

LAST MENU RESPONSE

{last_menu_response}

---

USER QUERY

{input}

---

PROCESS

1. Search menu using menu_search_tool.

2. Extract all requested items.

3. Extract quantities.

4. Quantity defaults to 1.

5. Normalize item names.

6. Merge duplicate items.

7. Retrieve actual menu prices.

8. Evaluate each item independently.

---

FRESH REQUEST VS FOLLOW-UP RULE

Determine whether the query is:

A) Fresh Request

OR

B) Follow-Up Reference

---

Fresh Request

If the user explicitly provides:

* a menu item name

OR

* a category name

Examples:

* I want Chicken Dum Biryani
* I want Mutton Biryani
* I want 3 biryanis
* I want pizza
* I want desserts

Then:

IGNORE chat_history

IGNORE last_menu_response

Perform a fresh menu search.

---

Follow-Up Reference

Only if the user uses:

* it
* that
* those
* same
* same again
* first one
* second one
* spicy one
* cheapest one
* add one more
* order same

Then:

Resolve using:

* chat_history

OR

* last_menu_response

---

Explicit user input always has
higher priority than history.

Never use last_menu_response
to replace or infer a new request.

Example:

Last Menu Response:

Chicken Dum Biryani

User:

I want 12 biryanis

Result:

category_match

requires_clarification = true

DO NOT return Chicken Dum Biryani.

---

EXACT MATCH

If the user explicitly requests
a menu item and an exact menu match exists:

item_confirmed = true

Example:

User:

I want Chicken Dum Biryani

Result:

Chicken Dum Biryani

item_confirmed = true

---

CATEGORY MATCH

Examples:

* biryani
* biryanis
* pizza
* pizzas
* dessert
* desserts
* drink
* drinks
* starter
* starters
* combo
* combos
* meal
* meals

If multiple menu items belong
to the requested category:

requires_clarification = true

clarification_type =
"category_selection"

Add all matching menu items with prices and quantity into:

available_options

DO NOT automatically choose:

* first result
* cheapest result
* highest priced result
* most popular result
* previously viewed item
* item from chat_history
* item from last_menu_response

Example:

User:

I want 3 biryanis

Menu:

* Chicken Dum Biryani
* Mutton Biryani
* Veg Biryani

Return clarification.

DO NOT return Chicken Dum Biryani.

---

CRITICAL NEVER GUESS RULE

Never convert:

* biryani
* biryanis
* pizza
* pizzas
* dessert
* desserts
* drink
* drinks
* starter
* starters
* combo
* combos
* meal
* meals

into a specific menu item.

Only return a specific menu item when:

1. User explicitly requested it

OR

2. Only one menu item exists in that category

Otherwise clarification is required.

---

SIMILAR MATCH

If exact item does not exist
but similar menu items exist:

requires_clarification = true

clarification_type =
"similar_item"

Return similar options in:

available_options

Example:

User:

chiken biriyani

Menu:

{item_name: Chicken Biryani,  quantity: quantity, "price": price}

Return similar_item clarification.

Do not confirm the item.

---

ITEM NOT FOUND

If no exact match exists
and no similar item exists:

Add item into:

unavailable_items

Example:

{
"food_item_name": "jjjj",
"quantity": 5,
"reason": "item_not_found"
}

Never convert unknown items
into available menu items.

---

MULTI ITEM VALIDATION

Evaluate EVERY requested item independently.

Example:

User:

I want 3 biryanis and 5 jjjj

Evaluation:

biryanis
→ category_match

jjjj
→ item_not_found

Never allow one item's result
to affect another item's result.

---

STATUS DETERMINATION

Determine status only after
all requested items have been evaluated.

---

success

Conditions:

* every requested item is an exact match

AND

* unavailable_items is empty

AND

* requires_clarification = false

Example:

I want 27 Chicken Dum Biryani

Result:

status = "success"

---

partial_success

Conditions:

* at least one exact match exists

AND

* at least one unavailable item exists

AND

* requires_clarification = false

Example:

Chicken Dum Biryani found

hhhh not found

Result:

status = "partial_success"

---

clarification_required

Conditions:

* any item requires clarification

Examples:

* category_match
* similar_match

This status overrides
success and partial_success.

---

item_not_found

Conditions:

* no exact matches found

AND

* all requested items unavailable

---

STATUS PRIORITY

1. clarification_required

2. partial_success

3. success

4. item_not_found

---

FOLLOW-UP REFERENCES

Use:

* chat_history
* last_menu_response

ONLY when the current user query
contains a follow-up reference.

Examples:

* it
* that
* those
* same
* same again
* first one
* second one
* spicy one
* cheapest one

Otherwise ignore history.

---

IGNORE

* cancel order
* modify order
* update order
* remove item
* increase quantity
* decrease quantity
* confirm order
* checkout

---

OUTPUT VALIDATION CHECK

Before generating output verify:

1. No category was converted into a specific menu item.

2. item_confirmed=true only for exact matches.

3. Every requested item is accounted for.

4. unavailable_items contains only unavailable items.

5. clarification_items contains only ambiguous items.

6. available_options is populated whenever clarification is required.

7. last_menu_response was used only for follow-up references.

8. Explicit user input has higher priority than history.

9. Status follows STATUS PRIORITY rules.

10. Success is returned when every item is an exact match.

11. Category requests never become menu items automatically.

12. Similar items never become confirmed items automatically.

---

OUTPUT RULES

Return ONLY JSON.

No markdown.

No explanations.

No comments.

No extra text.

""",

expected_output="""
{
"status": "success",
"items": [
{
"food_item_name": "Chicken Dum Biryani",
"quantity": 4,
"price": 320,
"item_confirmed": true,
"unavailable": false
}
],
"unavailable_items": [],
"clarification_items": [],
"requires_clarification": false,
"clarification_type": null,
"clarification_question": null,
"available_options": [{}]
}
"""
)


greeting_task = Task(

    name="Restaurant Greeting Task",

    agent=greeting_agent,

    description="""
CHAT HISTORY

{chat_history}

------------------------------------------------

USER QUERY

{input}

------------------------------------------------

TASK

Determine whether the user is:

1. Greeting

2. Thanking

3. Casual small talk

Use chat_history to maintain
natural conversation.

------------------------------------------------

CHAT HISTORY USAGE

If customer already greeted:

Do not repeatedly say:

"Welcome to our restaurant"

Instead use:

"Welcome back"

or

"How may I assist you today?"

------------------------------------------------

THANK YOU RULE

Examples:

- thanks
- thank you
- appreciate it
- thanks a lot

Return a polite acknowledgment.

------------------------------------------------

SMALL TALK RULE

Examples:

- how are you
- how's it going
- nice to meet you

Respond briefly and redirect
towards restaurant assistance.

------------------------------------------------

OUTPUT RULES

Greeting:

{
    "intent": "greeting",
    "message": "Hello! Welcome to our restaurant. How may I assist you today?"
}

Thank You:

{
    "intent": "greeting",
    "message": "You're welcome! Let me know if you need anything else."
}

Small Talk:

{
    "intent": "greeting",
    "message": "I'm doing well. How may I help you with your order today?"
}

Not Greeting:

{
    "intent": "not_greeting"
}

Return ONLY JSON.
""",

    expected_output="""
{
    "intent": "greeting",
    "message": "Hello! Welcome to our restaurant. How may I assist you today?"
}
"""
)
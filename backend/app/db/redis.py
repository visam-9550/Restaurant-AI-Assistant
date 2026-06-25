
# import redis

# redis_client = redis.Redis(
#     host="localhost",
#     port=6379,
#     decode_responses=True
# )


import os
import redis

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

# 2. Connect using the URL method
redis_client = redis.from_url(
    redis_url,
    decode_responses=True
)

# Quick connection test (optional)
try:
    redis_client.ping()
    print("Successfully connected to Redis!")
except Exception as e:
    print(f"Redis connection failed: {e}")
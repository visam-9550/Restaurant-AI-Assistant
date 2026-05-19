from redis import Redis

try:

    redis_client = Redis(

        host="localhost",

        port=6379,

        decode_responses=True

    )

    redis_client.set(

        "name",

        "venkat"

    )

    value = redis_client.get(

        "name"

    )

    print("Redis Working:")

    print(value)

except Exception as e:

    print("Redis Error:")

    print(e)
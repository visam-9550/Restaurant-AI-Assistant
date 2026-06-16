from urllib import response

from crewai.flow.flow import Flow, listen, start, router
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

class SampleFlowModel(BaseModel):
    query: str = ""
    response: str = ""

class SampleFlow(Flow[SampleFlowModel]):
    print("Initializing SampleFlow")
    @start()
    def start_flow(self):
        print("Starting SampleFlow with query:", self.state.query)
        self.state.response = "Flow started. Please provide your query."
        return self.state.response

    @listen(start_flow)
    def handle_query(self):
        print("Handling query:", self.state.query)
        self.state.response = f"Received query: {self.state.query}"
        return self.state.response
    
flow = SampleFlow()
# response = flow.kickoff(inputs={"query": "What is the weather today?"})
# print("Flow response:", response)


# from app.db.redis import redis_client

# redis_client.set("test_key", "Hello, Redis!")
# value = redis_client.get("test_key")
# print("Value from Redis:", value)


import asyncio


async def test_async_function():
    print("Testing async function...")
    await asyncio.sleep(1)
    inputs = {"query": "What is the weather today?"}
    response = await run_in_threadpool(
        flow.kickoff,
        inputs
    )
    # response = flow.kickoff(inputs={"query": "What is the weather today?"})
    print("Flow response:", response)
    print("Async function test completed.")

asyncio.run(test_async_function())
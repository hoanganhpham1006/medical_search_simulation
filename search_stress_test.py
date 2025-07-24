import asyncio
import aiohttp
import time
import random

# Server and test parameters
SERVER_URL = "http://192.168.0.8:10000/search"
CONCURRENT_CLIENTS = 128        # Number of concurrent tasks
TOTAL_REQUESTS = 12800           # Total requests to send
TIMEOUT = aiohttp.ClientTimeout(total=50)

# Sample natural language queries for variation
QUERY_POOL = [
    # "capital of Netherlands city",
    # "largest lake in Africa",
    # "who wrote the iliad",
    # "temperature of the sun's core",
    # "fastest land animal",
    # "deepest ocean trench",
    # "president of the USA in 1990",
    # "population of Tokyo",
    # "how does a black hole form",
    # "tallest building in the world",
    "University of Birmingham laser sprint study"
]

# Generate a randomized query
def generate_query():
    return random.choice(QUERY_POOL)

# Single request coroutine
async def send_query(session, request_id):
    payload = {
        "query": generate_query(),
        "use_reranker": False,
        "preview_char": 512,
        "top_k": 5
    }

    try:
        start = time.time()
        async with session.post(SERVER_URL, json=payload) as response:
            elapsed = time.time() - start
            if response.status == 200:
                await response.text()  # You can parse JSON if needed
                print(f"[{request_id}] ✅ {response.status} in {elapsed:.2f}s")
            else:
                print(f"[{request_id}] ❌ {response.status} in {elapsed:.2f}s")
    except Exception as e:
        print(f"[{request_id}] ❗ Error: {e}")

# Main stress function
async def stress_test():
    connector = aiohttp.TCPConnector(limit=CONCURRENT_CLIENTS)
    async with aiohttp.ClientSession(timeout=TIMEOUT, connector=connector) as session:
        tasks = []
        for i in range(TOTAL_REQUESTS):
            tasks.append(send_query(session, i))
            if len(tasks) >= CONCURRENT_CLIENTS:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)

# Entry point
if __name__ == "__main__":
    asyncio.run(stress_test())
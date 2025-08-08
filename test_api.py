import asyncio
import aiohttp
import json


async def test_api():
    url = "http://localhost:8000/hackrx/run"

    test_data = {
        "documents": "https://example.com/sample-policy.pdf",
        "questions": [
            "Does this policy cover knee surgery?",
            "What is the waiting period for maternity coverage?",
            "What are the coverage limits for cataract surgery?"
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer default_token"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=test_data, headers=headers) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                result = await response.json()
                print(json.dumps(result, indent=2))
            else:
                error_text = await response.text()
                print(f"Error: {error_text}")

if __name__ == "__main__":
    asyncio.run(test_api())

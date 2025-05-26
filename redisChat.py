import asyncio
from main import LowLatencyChatbot


async def test_with_same_session():
    bot = LowLatencyChatbot()
    await bot.initialize()

    user_id = "user139"
    session_id = "765"  # Use the same session ID

    # First question
    response1 = await bot.chat(user_id, "What is the largest animal on land?", session_id)
    print(f"Q1: {response1.response}")

    # Second question (should remember context)
    response2 = await bot.chat(user_id, "Can it swim?", session_id)
    print(f"Q2: {response2.response}")

    response3 = await bot.chat(user_id, "What's their favorite food?", session_id)
    print(f"Q3: {response3.response}")

    await bot.close()


asyncio.run(test_with_same_session())
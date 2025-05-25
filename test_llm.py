from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500,  # Limit tokens for faster response
            timeout=2.0,
            # groq_api_key=Config.GROQ_API_KEY
        )

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)

print(ai_msg.content)
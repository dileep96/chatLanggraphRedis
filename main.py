# requirements.txt
"""
langgraph==0.0.62
langchain==0.1.20
langchain-groq==0.1.5
langchain-community==0.0.38
redis==5.0.4
pydantic==2.7.1
python-dotenv==1.0.1
uvicorn==0.29.0
fastapi==0.111.0
asyncio-mqtt==0.16.1
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict
from uuid import uuid4

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
class Config:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MAX_HISTORY_LENGTH = 20  # Maximum messages before summarization
    SUMMARY_TRIGGER_LENGTH = 15  # When to start considering summarization
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    MAX_RESPONSE_TIME = 2.0  # Maximum acceptable response time in seconds


# Pydantic models
class ChatMessage(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    response_time: float
    tokens_used: Optional[int] = None


class UserState(TypedDict):
    messages: List[BaseMessage]
    user_id: str
    session_id: str
    last_activity: datetime
    summary: Optional[str]
    context: Dict[str, Any]


class RedisMemoryManager:
    """Manages Redis connections and operations for user state and chat history"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

    async def store_user_state(self, user_id: str, session_id: str, state: UserState):
        """Store user state in Redis with TTL"""
        try:
            state_key = f"state:{user_id}:{session_id}"
            # Convert BaseMessage objects to dict for JSON serialization
            serializable_state = {
                "messages": [self._message_to_dict(msg) for msg in state["messages"]],
                "user_id": state["user_id"],
                "session_id": state["session_id"],
                "last_activity": state["last_activity"].isoformat(),
                "summary": state.get("summary"),
                "context": state.get("context", {})
            }

            await self.redis_client.setex(
                state_key,
                Config.SESSION_TIMEOUT,
                json.dumps(serializable_state)
            )
            logger.debug(f"Stored state for user {user_id}, session {session_id}")
        except Exception as e:
            logger.error(f"Failed to store user state: {e}")
            raise

    async def get_user_state(self, user_id: str, session_id: str) -> Optional[UserState]:
        """Retrieve user state from Redis"""
        try:
            state_key = f"state:{user_id}:{session_id}"
            state_data = await self.redis_client.get(state_key)

            if not state_data:
                return None

            state_dict = json.loads(state_data)
            # Convert dict back to BaseMessage objects
            state_dict["messages"] = [
                self._dict_to_message(msg_dict)
                for msg_dict in state_dict["messages"]
            ]
            state_dict["last_activity"] = datetime.fromisoformat(state_dict["last_activity"])

            return state_dict
        except Exception as e:
            logger.error(f"Failed to retrieve user state: {e}")
            return None

    async def store_chat_history(self, user_id: str, session_id: str, messages: List[BaseMessage]):
        """Store chat history in Redis with longer TTL"""
        try:
            history_key = f"history:{user_id}:{session_id}"
            serializable_messages = [self._message_to_dict(msg) for msg in messages]

            await self.redis_client.setex(
                history_key,
                Config.SESSION_TIMEOUT * 24,  # 24 hours for history
                json.dumps(serializable_messages)
            )
            logger.debug(f"Stored chat history for user {user_id}, session {session_id}")
        except Exception as e:
            logger.error(f"Failed to store chat history: {e}")
            raise

    async def get_chat_history(self, user_id: str, session_id: str) -> List[BaseMessage]:
        """Retrieve chat history from Redis"""
        try:
            history_key = f"history:{user_id}:{session_id}"
            history_data = await self.redis_client.get(history_key)

            if not history_data:
                return []

            history_list = json.loads(history_data)
            return [self._dict_to_message(msg_dict) for msg_dict in history_list]
        except Exception as e:
            logger.error(f"Failed to retrieve chat history: {e}")
            return []

    def _message_to_dict(self, message: BaseMessage) -> Dict:
        """Convert BaseMessage to dictionary for JSON serialization"""
        return {
            "type": message.__class__.__name__,
            "content": message.content,
            "timestamp": getattr(message, 'timestamp', datetime.now().isoformat())
        }

    def _dict_to_message(self, msg_dict: Dict) -> BaseMessage:
        """Convert dictionary back to BaseMessage object"""
        message_types = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "SystemMessage": SystemMessage
        }

        message_class = message_types.get(msg_dict["type"], HumanMessage)
        message = message_class(content=msg_dict["content"])
        message.timestamp = msg_dict.get("timestamp", datetime.now().isoformat())
        return message


class ChatSummarizer:
    """Handles chat history summarization to reduce latency"""

    def __init__(self, llm):
        self.llm = llm
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that summarizes chat conversations. 
            Create a concise summary of the conversation history that preserves:
            1. Key topics discussed
            2. Important user preferences or information
            3. Context needed for future responses
            4. Any ongoing tasks or questions

            Keep the summary under 200 words while maintaining essential context."""),
            ("human", "Please summarize this conversation history:\n\n{history}")
        ])

    async def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """Determine if conversation should be summarized"""
        return len(messages) >= Config.SUMMARY_TRIGGER_LENGTH

    async def summarize_history(self, messages: List[BaseMessage]) -> str:
        """Create a summary of the conversation history"""
        try:
            # Format messages for summarization
            history_text = "\n".join([
                f"{msg.__class__.__name__}: {msg.content}"
                for msg in messages[:-Config.MAX_HISTORY_LENGTH // 2]  # Summarize older half
            ])

            # Generate summary
            summary_chain = self.summary_prompt | self.llm
            result = await summary_chain.ainvoke({"history": history_text})

            logger.info("Generated conversation summary")
            return result.content
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Previous conversation context available but summary failed to generate."


class LowLatencyChatbot:
    """Main chatbot class using LangGraph with Redis state management"""

    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500,  # Limit tokens for faster response
            timeout=Config.MAX_RESPONSE_TIME,
            groq_api_key=Config.GROQ_API_KEY
        )
        self.memory_manager = RedisMemoryManager(Config.REDIS_URL)
        self.summarizer = ChatSummarizer(self.llm)
        self.graph = None

    async def initialize(self):
        """Initialize the chatbot"""
        await self.memory_manager.initialize()
        self._build_graph()
        logger.info("Chatbot initialized successfully")

    def _build_graph(self):
        """Build the LangGraph workflow"""

        # Define the graph state
        workflow = StateGraph(UserState)

        # Add nodes
        workflow.add_node("load_context", self._load_context)
        workflow.add_node("check_summarization", self._check_summarization)
        workflow.add_node("summarize", self._summarize_conversation)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("save_state", self._save_state)

        # Define edges
        workflow.set_entry_point("load_context")
        workflow.add_edge("load_context", "check_summarization")

        # Conditional edge for summarization
        workflow.add_conditional_edges(
            "check_summarization",
            self._should_summarize_decision,
            {
                "summarize": "summarize",
                "generate": "generate_response"
            }
        )

        workflow.add_edge("summarize", "generate_response")
        workflow.add_edge("generate_response", "save_state")
        workflow.add_edge("save_state", END)

        # Compile the graph
        self.graph = workflow.compile()

    async def _load_context(self, state: UserState) -> UserState:
        """Load user context and chat history"""
        try:
            logger.info(f"Loading context for user {state['user_id']}, session {state['session_id']}")

            # Load existing state or create new one
            existing_state = await self.memory_manager.get_user_state(
                state["user_id"],
                state["session_id"]
            )

            if existing_state:
                current_msg = state['messages']
                logger.info("Found existing state, merging...")
                # Merge with existing state
                state.update(existing_state)
                # Load full chat history
                history = await self.memory_manager.get_chat_history(
                    state["user_id"],
                    state["session_id"]
                )
                if history:
                    logger.info(f"Loaded {len(history)} messages from history")
                    state["messages"] = history
                    state["messages"].extend(current_msg)
                else:
                    logger.info("No chat history found")
            else:
                logger.info("No existing state found, using initial state")

            state["last_activity"] = datetime.now()
            logger.info(f"Context loaded, total messages: {len(state['messages'])}")
            return state
        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            return state

    async def _check_summarization(self, state: UserState) -> UserState:
        """Check if summarization is needed"""
        state["needs_summarization"] = await self.summarizer.should_summarize(
            state["messages"]
        )
        return state

    def _should_summarize_decision(self, state: UserState) -> str:
        """Decision function for summarization"""
        return "summarize" if state.get("needs_summarization", False) else "generate"

    async def _summarize_conversation(self, state: UserState) -> UserState:
        """Summarize conversation history"""
        try:
            if len(state["messages"]) > Config.SUMMARY_TRIGGER_LENGTH:
                summary = await self.summarizer.summarize_history(state["messages"])
                state["summary"] = summary

                # Keep only recent messages + summary
                recent_messages = state["messages"][-Config.MAX_HISTORY_LENGTH // 2:]
                summary_message = SystemMessage(content=f"Previous conversation summary: {summary}")
                state["messages"] = [summary_message] + recent_messages

            return state
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return state

    async def _generate_response(self, state: UserState) -> UserState:
        """Generate AI response"""
        try:
            # Prepare messages for the LLM
            system_message = SystemMessage(
                content="""You are a helpful AI assistant. Provide concise, relevant responses.
                Use any conversation summary or context to maintain continuity."""
            )

            # Include summary in context if available
            messages = [system_message]
            if state.get("summary"):
                messages.append(SystemMessage(content=f"Context: {state['summary']}"))

            # Add recent conversation history (excluding the current user message we're responding to)
            recent_messages = state["messages"][-Config.MAX_HISTORY_LENGTH:]
            messages.extend(recent_messages)

            logger.info(f"Sending {len(messages)} messages to LLM")

            # Generate response
            response = await self.llm.ainvoke(messages)

            logger.info(f"LLM response received: {response.content[:100]}...")

            # Add AI response to messages
            ai_message = AIMessage(content=response.content)
            ai_message.timestamp = datetime.now().isoformat()
            state["messages"].append(ai_message)

            # Store the response for retrieval
            state["last_response"] = response.content

            return state
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            error_message = AIMessage(content="I apologize, but I'm having trouble processing your request right now.")
            state["messages"].append(error_message)
            state["last_response"] = error_message.content
            return state

    async def _save_state(self, state: UserState) -> UserState:
        """Save state and chat history to Redis"""
        try:
            logger.info(f"Saving state for user {state['user_id']}, session {state['session_id']}")
            logger.info(f"Saving {len(state['messages'])} messages to state and history")

            # Save current state
            await self.memory_manager.store_user_state(
                state["user_id"],
                state["session_id"],
                state
            )
            logger.info("User state saved successfully")

            # Save chat history
            await self.memory_manager.store_chat_history(
                state["user_id"],
                state["session_id"],
                state["messages"]
            )
            logger.info("Chat history saved successfully")

            return state
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            return state

    async def chat(self, user_id: str, message: str, session_id: Optional[str] = None) -> ChatResponse:
        """Main chat method"""
        start_time = time.time()

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid4())

        logger.info(f"Processing chat for user {user_id}, session {session_id}")
        logger.info(f"User message: {message}")

        # Create user message with timestamp
        user_message = HumanMessage(content=message)
        user_message.timestamp = datetime.now().isoformat()

        # Create initial state
        initial_state: UserState = {
            "messages": [user_message],
            "user_id": user_id,
            "session_id": session_id,
            "last_activity": datetime.now(),
            "summary": None,
            "context": {}
        }

        try:
            # Run the graph
            logger.info("Starting graph execution")
            result = await self.graph.ainvoke(initial_state)
            logger.info("Graph execution completed")
            # logger.info(f"Final result keys: {list(result.keys())}")
            # logger.info(f"messages: {result['messages']}")

            response_time = time.time() - start_time

            # Extract the last AI message from the messages list
            final_response = "I'm sorry, I couldn't generate a response."
            if result.get("messages"):
                for message in reversed(result["messages"]):
                    if isinstance(message, AIMessage):
                        final_response = message.content
                        break

            logger.info(f"Final response: {final_response[:100]}...")

            return ChatResponse(
                response=final_response,
                session_id=session_id,
                response_time=response_time
            )

        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            response_time = time.time() - start_time

            return ChatResponse(
                response="I apologize, but I encountered an error processing your message.",
                session_id=session_id,
                response_time=response_time
            )

    async def close(self):
        """Clean up resources"""
        await self.memory_manager.close()


# FastAPI application
app = FastAPI(title="Low-Latency LangGraph Chatbot", version="1.0.0")

# Global chatbot instance
chatbot = None


@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot
    chatbot = LowLatencyChatbot()
    await chatbot.initialize()
    logger.info("Chatbot service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global chatbot
    if chatbot:
        await chatbot.close()
    logger.info("Chatbot service stopped")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Chat endpoint"""
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")

        response = await chatbot.chat(
            user_id=chat_message.user_id,
            message=chat_message.message,
            session_id=chat_message.session_id
        )

        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# CLI interface for testing
async def cli_chat():
    """Command line interface for testing"""
    bot = LowLatencyChatbot()
    await bot.initialize()

    user_id = input("Enter user ID (or press Enter for 'test_user'): ") or "test_user"
    session_id = input("Enter session ID (or press Enter for new session): ") or str(uuid4())

    print(f"Chat session started")
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print("Type 'quit' to exit, 'new' for new session\n")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'new':
                session_id = str(uuid4())
                print(f"New session started: {session_id}\n")
                continue

            response = await bot.chat(user_id, user_input, session_id)
            print(f"Bot: {response.response}")
            print(f"(Response time: {response.response_time:.2f}s)\n")

    finally:
        await bot.close()


# Simple test client that maintains session
async def test_conversation():
    """Test client that demonstrates session continuity"""
    bot = LowLatencyChatbot()
    await bot.initialize()

    user_id = "test_user"
    session_id = str(uuid4())

    print(f"Starting conversation with session: {session_id}\n")

    # First message
    print("=== First Message ===")
    response1 = await bot.chat(user_id, "What is the largest animal on land?", session_id)
    print(f"User: What is the largest animal on land?")
    print(f"Bot: {response1.response}")
    print(f"Time: {response1.response_time:.2f}s\n")

    # Second message (should remember context)
    print("=== Second Message (should remember context) ===")
    response2 = await bot.chat(user_id, "Can it swim?", session_id)
    print(f"User: Can it swim?")
    print(f"Bot: {response2.response}")
    print(f"Time: {response2.response_time:.2f}s\n")

    # Third message
    print("=== Third Message ===")
    response3 = await bot.chat(user_id, "How much does it weigh?", session_id)
    print(f"User: How much does it weigh?")
    print(f"Bot: {response3.response}")
    print(f"Time: {response3.response_time:.2f}s\n")

    await bot.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run CLI interface
        asyncio.run(cli_chat())
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test conversation
        asyncio.run(test_conversation())
    else:
        # Run FastAPI server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
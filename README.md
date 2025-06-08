# Low-Latency LangGraph Chatbot

## Overview

This project implements a highly efficient, low-latency chatbot powered by LangGraph, Groq's LLM API, and Redis for managing user states and chat histories. Designed to be scalable and responsive, it ensures smooth user interactions with built-in summarization capabilities to maintain performance even in lengthy conversations.

## Key Features

* **LangGraph Integration:** Utilizes LangGraph to structure chatbot workflows efficiently, enabling stateful and context-aware conversations.
* **Redis for State Management:** Redis is used to store user states and chat history, ensuring quick retrieval and persistent conversation contexts.
* **Automatic Chat Summarization:** Automatically summarizes chat histories when they exceed a specified length to maintain rapid response times.
* **FastAPI Backend:** Provides RESTful API endpoints for easy integration and deployment.
* **Real-time CLI Interface:** Includes a command-line interface for testing and interacting directly with the chatbot.

## Technology Stack

* **LangGraph:** Workflow and state management for chat applications.
* **Redis:** High-performance storage for conversation states and histories.
* **Groq LLM API:** Generates intelligent and contextually relevant responses.
* **FastAPI & Uvicorn:** Serves API endpoints with efficient async capabilities.
* **Asyncio:** Manages asynchronous operations and concurrency.
* **Pydantic:** Validates and serializes API requests and responses.

## Requirements

Ensure the following dependencies are installed:

```bash
pip install -r requirements.txt
```

## Configuration

Environment variables (`.env` file):

```env
REDIS_URL=redis://localhost:6379
GROQ_API_KEY=your-groq-api-key
```

## Usage

### Run FastAPI Server

Start the server:

```bash
python main.py
```

### API Endpoints

* **Chat Interaction:**

  ```
  POST /chat
  ```

  Request body:

  ```json
  {
    "user_id": "user123",
    "message": "Hello, chatbot!",
    "session_id": "optional-session-id"
  }
  ```

  Response:

  ```json
  {
    "response": "Hello! How can I assist you today?",
    "session_id": "generated-or-provided-session-id",
    "response_time": 1.23
  }
  ```

* **Health Check:**

  ```
  GET /health
  ```

### CLI Testing

Run the chatbot in interactive mode:

```bash
python main.py cli
```

### Automated Testing

Run a test conversation demonstrating session continuity:

```bash
python main.py test
```

## Summarization and Memory

* **Automatic Summarization:**

  * Triggers after the chat reaches 15 messages.
  * Keeps the most recent messages and summarizes older ones.
  * Improves response latency and maintains essential context.

* **Memory Management:**

  * User state is preserved in Redis with a TTL (time-to-live) mechanism, enabling efficient resource use and session continuity.

## Deployment

* Ensure Redis is running and accessible.
* Provide necessary API keys and configurations.
* Deploy using your preferred infrastructure (Docker, cloud services, etc.).


---

This chatbot solution ensures efficient, stateful interactions with robust memory management and response optimization, perfect for scalable chat applications.

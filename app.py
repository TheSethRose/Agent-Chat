# app.py

# Import necessary libraries
import logging
import asyncio
import json
import re
import os
from typing import List, Dict, Any, AsyncIterator, Optional

from pydantic import BaseModel, Field, model_validator
from fastapi.middleware.cors import CORSMiddleware

# Import dotenv to load environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define Pydantic models for ChatMessage and ChatResponse
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="The content of the message")

class ChatResponse(BaseModel):
    message: ChatMessage
    thoughts: Optional[str] = Field(None, description="The agent's thoughts or reasoning process")
    sources: Optional[List[str]] = Field(None, description="Sources of information used in the response")

# Import Phidata classes
from phi.agent import Agent, RunResponse
from phi.vectordb.lancedb import LanceDb
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.calculator import Calculator
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.playground import Playground, serve_playground_app

# Import tools from the new location
from tools.calculator import EnhancedCalculator
from tools.duckduckgo_search import EnhancedDuckDuckGo

# Add this import instead:
from models import get_model, get_embedder

# Set up logging configuration
# To adjust logging level, set the LOG_LEVEL environment variable (e.g., export LOG_LEVEL=DEBUG)
# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
# DEBUG: Detailed information for diagnosing problems
# INFO: Confirmation that things are working as expected
# WARNING: Indication that something unexpected happened
# ERROR: Serious problem that prevented a function from executing
# CRITICAL: Serious error that may prevent the program from continuing
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration for session storage
DB_SESSION_STORAGE_FILE: str = os.getenv('DB_SESSION_STORAGE_FILE', 'agent_chat.db')
VECTOR_DB_URI: str = os.getenv('VECTOR_DB_URI', '/tmp/lancedb')
VECTOR_DB_TABLE_NAME: str = os.getenv('VECTOR_DB_TABLE_NAME', 'agent_chat_vectors')

class AgentChat(Agent):
    """
    AgentChat integrates DuckDuckGo for search, Calculator for calculations,
    and Ollama with Llama 3.2 model for language model analysis.
    """

    # Define tools with default factories
    duckduckgo_tool: EnhancedDuckDuckGo = Field(default_factory=EnhancedDuckDuckGo)
    calculator_tool: EnhancedCalculator = Field(default_factory=EnhancedCalculator)
    llm: Any = Field(default_factory=get_model)
    embedder: Any = Field(default_factory=get_embedder)

    # Agent metadata
    name: str = "AgentChat"
    description: str = (
        "An advanced AI-powered agent with reasoning capabilities and extensive knowledge."
    )
    tools: List[Any] = Field(default_factory=list)
    markdown: bool = True
    show_tool_calls: bool = True
    storage: SqlAgentStorage = Field(
        default_factory=lambda: SqlAgentStorage(table_name="agent_chat", db_file=DB_SESSION_STORAGE_FILE)
    )
    debug_mode: bool = True

    # Context and conversation history for the agent
    context: Dict[str, str] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

    # Embedder and Vector Store
    vector_store: Optional[LanceDb] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def set_tools_and_initialize(cls, values):
        """
        Ensure that tools are properly set and initialize embedder and vector store after tools and LLM are initialized.
        """
        logger.info("Initializing AgentChat tools and components")
        # Add tool instances to the tools list for easy access
        values.tools = [values.duckduckgo_tool, values.calculator_tool]

        # Initialize LanceDb vector store for storing embeddings
        values.vector_store = LanceDb(
            uri=VECTOR_DB_URI,
            table_name=VECTOR_DB_TABLE_NAME,
            embedder=values.embedder,
        )

        logger.debug("Embedder initialized")
        logger.debug("LanceDb vector store initialized")
        logger.info("AgentChat initialization completed")
        return values

    def run_tool(self, tool_name: str, *args, **kwargs):
        """
        Executes the specified tool with provided arguments.
        """
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Starting tool execution: {tool_name}")
        try:
            result = None
            # Select the correct tool based on the tool_name
            if tool_name.lower() == "duckduckgo_search":
                result = self.duckduckgo_tool.duckduckgo_search(*args, **kwargs)
            elif tool_name.lower().startswith("calculator_"):
                method_name = tool_name.split('_')[1]  # This will be 'add', 'subtract', 'multiply', or 'divide'
                calculator_method = getattr(self.calculator_tool, method_name)
                result = calculator_method(*args, **kwargs)
            else:
                raise ValueError(f"Tool '{tool_name}' not found")
            logger.debug(f"Tool execution completed: {tool_name}")
            return result
        except Exception as e:
            logger.error(f"Error during tool execution: {tool_name}", exc_info=True)
            raise

    def perform_calculation(self, num1: float, num2: float, operator: str) -> str:
        """
        Performs a calculation based on the given numbers and operator.

        Args:
            num1 (float): The first number in the calculation.
            num2 (float): The second number in the calculation.
            operator (str): The mathematical operator ('+', '-', '*', '/', or 'x').

        Returns:
            A formatted string with the result of the calculation.
        """
        logger.info(f"Performing calculation: {num1} {operator} {num2}")
        try:
            result = self.calculator_tool.calculate(num1, num2, operator)
            operation_name = {'+': 'addition', '-': 'subtraction', '*': 'multiplication', '/': 'division'}[operator]
            logger.debug(f"Calculation result: {result}")
            return f"The result of the {operation_name} {num1} {operator} {num2} is {result}."
        except ValueError as e:
            logger.error(f"Error during calculation: {str(e)}")
            return f"Error: {str(e)}"

    def process_query(self, query: str, stream: bool = False) -> str:
        logger.info(f"Processing query: '{query}'")
        try:
            calculation = self.extract_calculation_from_query(query)
            if calculation:
                logger.debug("Query identified as a calculation")
                num1, num2, operator = calculation
                response = self.perform_calculation(num1, num2, operator)
            else:
                logger.debug("Query requires search and analysis")
                try:
                    logger.info("Performing DuckDuckGo search")
                    search_results = self.run_tool("duckduckgo_search", query)
                    logger.debug("Summarizing search results")
                    search_summary = self.summarize_search_results(search_results)

                    logger.info("Invoking LLM for analysis")
                    analysis_prompt = ChatMessage(
                        role="user",
                        content=f"""
        Based on the following information related to '{query}', provide a detailed answer:

        {search_summary}

        Answer:
        """
                    )
                    logger.debug(f"LLM Prompt: {analysis_prompt.content}")

                    # Invoke the LLM to generate a response based on the prompt
                    final_answer = self.llm.invoke([analysis_prompt])

                    # Extract the response from the LLM
                    if isinstance(final_answer, ChatResponse):
                        response = final_answer.message.content
                    elif isinstance(final_answer, str):
                        response = final_answer
                    else:
                        response = "An error occurred while generating the response."

                    logger.debug(f"LLM Response:\n{response}")
                    logger.debug("LLM analysis completed")
                except Exception as e:
                    logger.error(f"Error during search and analysis: {e}", exc_info=True)
                    response = f"An error occurred while processing your query: {str(e)}"

            logger.info("Updating conversation history")
            self.conversation_history.append({'query': query, 'answer': response})
            self.conversation_history = self.conversation_history[-5:]

            return response
        except Exception as e:
            logger.error("Error during query processing", exc_info=True)
            return f"An error occurred while processing your query: {str(e)}"

    async def process_query_async(self, query: str, stream: bool = True) -> AsyncIterator[str]:
        logger.info(f"Processing asynchronous query: '{query}'")
        try:
            calculation = self.extract_calculation_from_query(query)
            if calculation:
                logger.debug("Asynchronous query identified as a calculation")
                num1, num2, operator = calculation
                response = await asyncio.to_thread(self.perform_calculation, num1, num2, operator)
                yield response
            else:
                logger.debug("Asynchronous query requires search and analysis")
                logger.info("Performing asynchronous DuckDuckGo search")
                search_results = await asyncio.to_thread(self.run_tool, "duckduckgo_search", query)

                if not search_results:
                    logger.warning("No search results found")
                    yield "I couldn't find any relevant information for your query."
                    return

                logger.debug("Summarizing asynchronous search results")
                search_summary = self.summarize_search_results(search_results)
                if search_summary.startswith("Error:"):
                    logger.error("Error in search summary")
                    yield search_summary
                    return

                logger.info("Invoking LLM for asynchronous analysis")
                analysis_prompt = ChatMessage(
                    role="user",
                    content=f"""
        Based on the following information related to '{query}', provide a detailed answer:

        {search_summary}

        Answer:
        """
                )
                logger.debug(f"LLM Prompt: {analysis_prompt.content}")

                try:
                    # Invoke the LLM asynchronously to get a response
                    final_answer = await asyncio.to_thread(self.llm.invoke, [analysis_prompt])

                    # Extract the response content from LLM output
                    if isinstance(final_answer, ChatResponse):
                        response = final_answer.message.content
                    elif isinstance(final_answer, str):
                        response = final_answer
                    elif isinstance(final_answer, dict):
                        if 'message' in final_answer and isinstance(final_answer['message'], dict):
                            response = final_answer['message'].get('content', '')
                        else:
                            response = final_answer.get('content', '') if 'content' in final_answer else str(final_answer)
                    else:
                        response = "An error occurred while generating the response."

                    logger.debug(f"LLM Response: {response}")
                    logger.debug("Asynchronous LLM analysis completed")
                except Exception as e:
                    logger.error(f"Error during LLM invocation: {str(e)}", exc_info=True)
                    response = f"An error occurred while generating the response: {str(e)}"

            logger.info("Updating conversation history for asynchronous query")
            self.conversation_history.append({'query': query, 'answer': response})
            self.conversation_history = self.conversation_history[-5:]

            yield response
        except Exception as e:
            logger.error("Error during asynchronous query processing", exc_info=True)
            yield f"Error: An unexpected error occurred during processing. Details: {str(e)}"

    async def _arun(self, message: str, stream: bool = False, **kwargs) -> AsyncIterator[RunResponse]:
        """
        Asynchronously executes the agent with the provided message.
        Yields RunResponse chunks for streaming.

        Args:
            message (str): The input message to process.
            stream (bool): Whether to stream the response.
            **kwargs: Additional keyword arguments.

        Yields:
            RunResponse: Chunks of the processed response.
        """
        try:
            async for chunk in self.process_query_async(message, stream=stream):
                yield RunResponse(content=chunk)  # Yield response chunks as they are generated
        except Exception as e:
            logger.error("Error during asynchronous run", exc_info=True)
            yield RunResponse(content="An error occurred during processing.")

    def extract_calculation_from_query(self, query: str) -> Optional[tuple]:
        """
        Extracts numbers and the operation from the query for calculation.

        Args:
            query (str): The input query to extract calculation from.

        Returns:
            Optional[tuple]: A tuple of (num1, num2, operator) if found, else None.
        """
        logger.debug(f"Extracting calculation from query: '{query}'")
        # Define patterns for different operations
        patterns = [
            r"(\d+(?:\.\d+)?)\s*([\+\-\*/x])\s*(\d+(?:\.\d+)?)",  # Standard operators including 'x'
            r"(\d+(?:\.\d+)?)\s*times\s*(\d+(?:\.\d+)?)",  # "times" for multiplication
            r"(\d+(?:\.\d+)?)\s*divided\s*by\s*(\d+(?:\.\d+)?)",  # "divided by" for division
            r"(\d+(?:\.\d+)?)\s*plus\s*(\d+(?:\.\d+)?)",  # "plus" for addition
            r"(\d+(?:\.\d+)?)\s*minus\s*(\d+(?:\.\d+)?)"  # "minus" for subtraction
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    num1, operator, num2 = match.groups()
                    if operator.lower() == 'x':
                        operator = '*'  # Convert 'x' to '*' for multiplication
                else:
                    num1, num2 = match.groups()
                    if "times" in match.group().lower():
                        operator = "*"
                    elif "divided by" in match.group().lower():
                        operator = "/"
                    elif "plus" in match.group().lower():
                        operator = "+"
                    elif "minus" in match.group().lower():
                        operator = "-"
                logger.debug(f"Calculation extracted: {float(num1)} {operator} {float(num2)}")
                return float(num1), float(num2), operator
        logger.debug("No calculation found in query")
        return None

    def summarize_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Summarizes search results into a string for the LLM prompt.

        Args:
            search_results (List[Dict[str, Any]]): The search results to summarize.

        Returns:
            str: A summarized string of the search results.
        """
        logger.debug("Summarizing search results")
        summary = ""

        # Check if search_results is a string (which might be a JSON string)
        if isinstance(search_results, str):
            try:
                # Convert JSON string to list if applicable
                search_results = json.loads(search_results)
            except json.JSONDecodeError:
                logger.error("Failed to parse search results JSON string")
                return "Error: Unable to process search results."

        # Ensure search_results is a list
        if not isinstance(search_results, list):
            logger.error(f"Unexpected search results type: {type(search_results)}")
            return "Error: Unexpected search results format."

        # Loop through each result to build the summary
        for result in search_results:
            if isinstance(result, dict):
                # Extract title if available
                title = result.get('title', 'No Title')
                # Extract snippet if available
                snippet = result.get('snippet', '')
                # Add each result to the summary
                summary += f"- {title}: {snippet}\n"

        logger.debug(f"Search summary created with {len(search_results)} results")
        return summary

# Create the Playground instance with the AgentChat
agent = AgentChat()
playground = Playground(agents=[agent])

# Get the FastAPI app from the Playground instance
app = playground.get_app()

# Add CORS middleware to the FastAPI app to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.phidata.app"],  # Allows access from phidata.app
    allow_credentials=True,  # Allow credentials (cookies, headers, etc.)
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Entry point for running the application
if __name__ == "__main__":
    logger.info("Launching AgentChat playground application")
    try:
        logger.info("Starting FastAPI app with live reloading")
        serve_playground_app("app:app", reload=True)
    except Exception as e:
        logger.critical("Critical error while serving playground app", exc_info=True)


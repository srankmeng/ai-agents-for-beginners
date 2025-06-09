import json
import os

from typing import Annotated

from IPython.display import display, HTML

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField

from openai import AsyncOpenAI

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import FunctionCallContent,FunctionResultContent, StreamingTextContent
from semantic_kernel.functions import kernel_function

load_dotenv()
# Initialize the asynchronous OpenAI client
client = AsyncOpenAI(
    api_key=os.environ["GITHUB_TOKEN"],
    base_url="https://models.inference.ai.azure.com/"
)

# Create the OpenAI Chat Completion Service
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
)

class SearchPlugin:

    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    @kernel_function(
        name="build_augmented_prompt",
        description="Build an augmented prompt using retrieval context or function results.",
    )
    def build_augmented_prompt(self, query: str, retrieval_context: str) -> str:
        return (
            f"Retrieved Context:\n{retrieval_context}\n\n"
            f"User Query: {query}\n\n"
            "First review the retrieved context, if this does not answer the query, try calling an available plugin functions that might give you an answer. If no context is available, say so."
        )
    
    @kernel_function(
        name="retrieve_documents",
        description="Retrieve documents from the Azure Search service.",
    )
    def get_retrieval_context(self, query: str) -> str:
        results = self.search_client.search(query)
        context_strings = []
        for result in results:
            context_strings.append(f"Document: {result['content']}")
        return "\n\n".join(context_strings) if context_strings else "No results found"

class WeatherInfoPlugin:
    """A Plugin that provides the average temperature for a travel destination."""

    def __init__(self):
        # Dictionary of destinations and their average temperatures
        self.destination_temperatures = {
            "maldives": "82°F (28°C)",
            "swiss alps": "45°F (7°C)",
            "african safaris": "75°F (24°C)"
        }

    @kernel_function(description="Get the average temperature for a specific travel destination.")
    def get_destination_temperature(self, destination: str) -> Annotated[str, "Returns the average temperature for the destination."]:
        """Get the average temperature for a travel destination."""
        # Normalize the input destination (lowercase)
        normalized_destination = destination.lower()

        # Look up the temperature for the destination
        if normalized_destination in self.destination_temperatures:
            return f"The average temperature in {destination} is {self.destination_temperatures[normalized_destination]}."
        else:
            return f"Sorry, I don't have temperature information for {destination}. Available destinations are: Maldives, Swiss Alps, and African safaris."

######################################################################

# Initialize Azure AI Search with persistent storage
search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = "travel-documents"

search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)

index_client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=AzureKeyCredential(search_api_key)
)

# Define the index schema
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String)
]

index = SearchIndex(name=index_name, fields=fields)

# Check if index already exists if not, create it
try:
    existing_index = index_client.get_index(index_name)
    print(f"Index '{index_name}' already exists, using the existing index.")
except Exception:
    # Create the index if it doesn't exist
    print(f"Creating new index '{index_name}'...")
    index_client.create_index(index)


# Enhanced sample documents
documents = [
    {"id": "1", "content": "Contoso Travel offers luxury vacation packages to exotic destinations worldwide."},
    {"id": "2", "content": "Our premium travel services include personalized itinerary planning and 24/7 concierge support."},
    {"id": "3", "content": "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage."},
    {"id": "4", "content": "Popular destinations include the Maldives, Swiss Alps, and African safaris."},
    {"id": "5", "content": "Contoso Travel provides exclusive access to boutique hotels and private guided tours."}
]

# Add documents to the index
search_client.upload_documents(documents)

agent = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[SearchPlugin(search_client=search_client), WeatherInfoPlugin()],
    name="TravelAgent",
    instructions="Answer travel queries using the provided tools and context. If context is provided, do not say 'I have no context for that.'",
)

async def main():
    thread: ChatHistoryAgentThread | None = None

    user_inputs = [
        "Can you explain Contoso's travel insurance coverage?",
        "What is the average temperature of the Maldives?",
        "What is a good cold destination offered by Contoso and what is it average temperature?",
    ]

    for user_input in user_inputs:
        html_output = (
            f"<div style='margin-bottom:10px'>"
            f"<div style='font-weight:bold'>User:</div>"
            f"<div style='margin-left:20px'>{user_input}</div></div>"
        )

        agent_name = None
        full_response: list[str] = []
        function_calls: list[str] = []

        # Buffer to reconstruct streaming function call
        current_function_name = None
        argument_buffer = ""

        async for response in agent.invoke_stream(
            messages=user_input,
            thread=thread,
        ):
            thread = response.thread
            agent_name = response.name
            content_items = list(response.items)

            for item in content_items:
                if isinstance(item, FunctionCallContent):
                    if item.function_name:
                        current_function_name = item.function_name

                    # Accumulate arguments (streamed in chunks)
                    if isinstance(item.arguments, str):
                        argument_buffer += item.arguments
                elif isinstance(item, FunctionResultContent):
                    # Finalize any pending function call before showing result
                    if current_function_name:
                        formatted_args = argument_buffer.strip()
                        try:
                            parsed_args = json.loads(formatted_args)
                            formatted_args = json.dumps(parsed_args)
                        except Exception:
                            pass  # leave as raw string

                        function_calls.append(f"Calling function: {current_function_name}({formatted_args})")
                        current_function_name = None
                        argument_buffer = ""

                    function_calls.append(f"\nFunction Result:\n\n{item.result}")
                elif isinstance(item, StreamingTextContent) and item.text:
                    full_response.append(item.text)

        if function_calls:
            html_output += (
                "<div style='margin-bottom:10px'>"
                "<details>"
                "<summary style='cursor:pointer; font-weight:bold; color:#0066cc;'>Function Calls (click to expand)</summary>"
                "<div style='margin:10px; padding:10px; background-color:#f8f8f8; "
                "border:1px solid #ddd; border-radius:4px; white-space:pre-wrap; font-size:14px; color:#333;'>"
                f"{chr(10).join(function_calls)}"
                "</div></details></div>"
            )

        html_output += (
            "<div style='margin-bottom:20px'>"
            f"<div style='font-weight:bold'>{agent_name or 'Assistant'}:</div>"
            f"<div style='margin-left:20px; white-space:pre-wrap'>{''.join(full_response)}</div></div><hr>"
        )

        display(HTML(html_output))
        print(html_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
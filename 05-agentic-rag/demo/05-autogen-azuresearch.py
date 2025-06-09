import os
import time
import asyncio
from typing import List, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from azure.core.credentials import AzureKeyCredential
from autogen_ext.models.azure import AzureAIChatCompletionClient

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField

from dotenv import load_dotenv

load_dotenv()

client = AzureAIChatCompletionClient(
    model="gpt-4o-mini",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": True,
        "family": "unknown",
    },
)

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

# Create the index
# index_client.create_index(index)

# Enhanced sample documents
documents = [
    {"id": "1", "content": "Contoso Travel offers luxury vacation packages to exotic destinations worldwide."},
    {"id": "2", "content": "Our premium travel services include personalized itinerary planning and 24/7 concierge support."},
    {"id": "3", "content": "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage."},
    {"id": "4", "content": "Popular destinations include the Maldives, Swiss Alps, and African safaris."},
    {"id": "5", "content": "Contoso Travel provides exclusive access to boutique hotels and private guided tours."}
]

# Add documents to the index
# search_client.upload_documents(documents)

##################################################################

def get_retrieval_context(query: str) -> str:
    results = search_client.search(query)
    context_strings = []
    for result in results:
        context_strings.append(f"Document: {result['content']}")
    return "\n\n".join(context_strings) if context_strings else "No results found"

def get_weather_data(location: str) -> str:
    """
    Simulates retrieving weather data for a given location.
    In a real-world scenario, this would call a weather API.
    """
    # Simulated weather data for common locations
    weather_database = {
        "new york": {"temperature": 72, "condition": "Partly Cloudy", "humidity": 65, "wind": "10 mph"},
        "london": {"temperature": 60, "condition": "Rainy", "humidity": 80, "wind": "15 mph"},
        "tokyo": {"temperature": 75, "condition": "Sunny", "humidity": 50, "wind": "5 mph"},
        "sydney": {"temperature": 80, "condition": "Clear", "humidity": 45, "wind": "12 mph"},
        "paris": {"temperature": 68, "condition": "Cloudy", "humidity": 70, "wind": "8 mph"},
    }
    
    # Normalize the location string
    location_key = location.lower()
    
    # Check if we have data for this location
    if location_key in weather_database:
        data = weather_database[location_key]
        return f"Weather for {location.title()}:\n" \
               f"Temperature: {data['temperature']}°F\n" \
               f"Condition: {data['condition']}\n" \
               f"Humidity: {data['humidity']}%\n" \
               f"Wind: {data['wind']}"
    else:
        return f"No weather data available for {location}."

##################################################################

# Create agents with enhanced capabilities
assistant = AssistantAgent(
    name="assistant",
    model_client=client,
    system_message=(
        "You are a helpful AI assistant that provides answers using ONLY the provided context. "
        "Do NOT include any external information. Base your answer entirely on the context given below."
    ),
)

class RAGEvaluator:
    def __init__(self):
        self.responses: List[Dict] = []

    def evaluate_response(self, query: str, response: str, context: List[Dict]) -> Dict:
        # Basic metrics: response length, citation count, and a simple relevance score.
        start_time = time.time()
        metrics = {
            'response_length': len(response),
            'source_citations': sum(1 for doc in context if doc["content"] in response),
            'evaluation_time': time.time() - start_time,
            'context_relevance': self._calculate_relevance(query, context)
        }
        self.responses.append({
            'query': query,
            'response': response,
            'metrics': metrics
        })
        return metrics

    def _calculate_relevance(self, query: str, context: List[Dict]) -> float:
        # Simple relevance score: fraction of the documents where the query appears.
        return sum(1 for c in context if query.lower() in c["content"].lower()) / len(context)

##################################################################

async def ask_unified_rag(query: str, evaluator: RAGEvaluator, location: str = None):
    """
    A unified RAG function that combines both document retrieval and weather data
    based on the query and optional location parameter.
    
    Args:
        query: The user's question
        evaluator: The RAG evaluator to measure response quality
        location: Optional location for weather queries
    """
    try:
        # Get context from both sources
        retrieval_context = get_retrieval_context(query)
        
        # If location is provided, add weather data
        weather_context = ""
        if location:
            weather_context = get_weather_data(location)
            weather_intro = f"\nWeather Information for {location}:\n"
        else:
            weather_intro = ""
        
        # Augment the query with both contexts if available
        augmented_query = (
            f"Retrieved Context:\n{retrieval_context}\n\n"
            f"{weather_intro}{weather_context}\n\n"
            f"User Query: {query}\n\n"
            "Based ONLY on the above context, please provide the answer."
        )

        # Send the augmented query as a user message
        start_time = time.time()
        response = await assistant.on_messages(
            [TextMessage(content=augmented_query, source="user")],
            cancellation_token=CancellationToken(),
        )
        processing_time = time.time() - start_time

        # Create combined context for evaluation
        combined_context = documents.copy()  # Start with travel documents
        
        # Add weather as a document if it exists
        if location and weather_context:
            combined_context.append({"id": f"weather-{location}", "content": weather_context})
        
        # Evaluate the response
        metrics = evaluator.evaluate_response(
            query=query,
            response=response.chat_message.content,
            context=combined_context
        )
        
        result = {
            'response': response.chat_message.content,
            'processing_time': processing_time,
            'metrics': metrics,
        }
        
        # Add location to result if provided
        if location:
            result['location'] = location
            
        return result
    except Exception as e:
        print(f"Error processing unified query: {e}")
        return None

async def main():
    evaluator = RAGEvaluator()
    
    # Define user queries similar to the Semantic Kernel example
    user_inputs = [
        # Travel-only queries
        {"query": "Can you explain Contoso's travel insurance coverage?"},
        
        # Weather-only queries 
        {"query": "What's the current weather condition in London?", "location": "london"},
        
        # Combined queries
        {"query": "What is a good cold destination offered by Contoso and what is its temperature?", "location": "london"},
    ]
    
    print("Processing Queries:")
    for query_data in user_inputs:
        query = query_data["query"]
        location = query_data.get("location")
        
        if location:
            print(f"\nProcessing Query for {location}: {query}")
        else:
            print(f"\nProcessing Query: {query}")
        
        # Get the RAG context for printing (similar to the Semantic Kernel example)
        retrieval_context = get_retrieval_context(query)
        weather_context = get_weather_data(location) if location else ""
        
        # Print the RAG context for transparency
        print("\n--- RAG Context ---")
        print(retrieval_context)
        if weather_context:
            print(f"\n--- Weather Context for {location} ---")
            print(weather_context)
        print("-------------------\n")
            
        result = await ask_unified_rag(query, evaluator, location)
        if result:
            print("Response:", result['response'])
            print("\nMetrics:", result['metrics'])
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    import asyncio
    if asyncio.get_event_loop().is_running():
        asyncio.run(main())
    else:
        asyncio.run(main())
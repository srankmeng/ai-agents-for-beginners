import os
from dotenv import load_dotenv
import json

import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from typing import Any, Callable, Set, Dict, List, Optional

load_dotenv()
client = AzureAIChatCompletionClient(
    model="gpt-4o-mini",
    endpoint="https://models.inference.ai.azure.com",
    # To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
    # Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
    credential=AzureKeyCredential(os.environ.get("GITHUB_TOKEN")),
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": True,
        "family": "unknown",
    },
)

from typing import Dict, List, Optional


def vacation_destinations(city: str) -> tuple[str, str]:
    """
    Checks if a specific vacation destination is available
    
    Args:
        city (str): Name of the city to check
        
    Returns:
        tuple: Contains city name and availability status ('Available' or 'Unavailable')
    """
    destinations = {
        "Barcelona": "Available",
        "Tokyo": "Unavailable",
        "Cape Town": "Available",
        "Vancouver": "Available",
        "Dubai": "Unavailable",
    }

    if city in destinations:
        return city, destinations[city]
    else:
        return city, "City not found"

get_vacations = FunctionTool(
    vacation_destinations, description="Search for vacation destinations and if they are available or not."
)

agent = AssistantAgent(
    name="assistant",
    model_client=client,
    tools=[get_vacations],
    system_message="You are a travel agent that helps users find vacation destinations.",
    reflect_on_tool_use=True,
)

async def main():
    result = await client.create([UserMessage(content="What is the capital of France?", source="user")])
    print("Basic test result:", result)

    response = await agent.on_messages(
        [TextMessage(content="I would like to take a trip to Tokyo", source="user")],
        cancellation_token=CancellationToken(),
    )
    print("Agent inner messages:", response.inner_messages)
    print("Agent chat message:", response.chat_message)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen_core import CancellationToken

from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console

load_dotenv()

async def main():
    # Create client inside async context
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

    try:

        # Test basic client functionality
        result = await client.create([UserMessage(content="What is the capital of France?", source="user")])
        print("Basic test result:", result)

        agent = AssistantAgent(
            name="assistant",
            model_client=client,
            tools=[],
            system_message="You are a travel agent that plans great vacations",
        )

        # Define the query
        user_query = "Plan me a great sunny vacation"

        # Execute the agent response
        response = await agent.on_messages(
            [TextMessage(content=user_query, source="user")],
            cancellation_token=CancellationToken(),
        )
        print("Agent response:", response.chat_message.content)

    finally:
        # Properly close the client to avoid cleanup errors
        if hasattr(client, 'close'):
            await client.close()
        elif hasattr(client, '_client') and hasattr(client._client, 'close'):
            await client._client.close()



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
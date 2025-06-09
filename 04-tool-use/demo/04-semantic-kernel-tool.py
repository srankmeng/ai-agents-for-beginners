import json
import os

from dotenv import load_dotenv

from IPython.display import display, HTML

from typing import Annotated
from openai import AsyncOpenAI

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent
from semantic_kernel.functions import kernel_function

# Define a sample plugin for the sample
class DestinationsPlugin:
    """A List of Destinations for vacation."""

    @kernel_function(description="Provides a list of vacation destinations.")
    def get_destinations(self) -> Annotated[str, "Returns the vacation destinations."]:
        return """
        Barcelona, Spain
        Paris, France
        Berlin, Germany
        Tokyo, Japan
        New York, USA
        """

    @kernel_function(description="Provides the availability of a destination.")
    def get_availability(
        self, destination: Annotated[str, "The destination to check availability for."]
    ) -> Annotated[str, "Returns the availability of the destination."]:
        return """
        Barcelona - Unavailable
        Paris - Available
        Berlin - Available
        Tokyo - Unavailable
        New York - Available
        """

load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("GITHUB_TOKEN"), 
    base_url="https://models.inference.ai.azure.com/",
)

chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
)

# Create the agent
agent = ChatCompletionAgent(
    service=chat_completion_service,
    name="TravelAgent",
    instructions="Answer questions about the travel destinations and their availability.",
    plugins=[DestinationsPlugin()],
)

user_inputs = [
    "What destinations are available?",
    "Is Barcelona available?",
    "Are there any vacation destinations available not in Europe?",
]

async def main():
    thread: ChatHistoryAgentThread | None = None

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
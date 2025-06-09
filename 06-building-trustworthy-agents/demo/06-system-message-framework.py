import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

token = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.inference.ai.azure.com"

model_name = "gpt-4o-mini"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

role = "travel agent"
company = "contoso travel"
responsibility = "booking flights"

response = client.complete(
    messages=[
        SystemMessage(content="""You are an expert at creating AI agent assistants. 
You will be provided a company name, role, responsibilities and other
information that you will use to provide a system prompt for.
To create the system prompt, be descriptive as possible and provide a structure that a system using an LLM can better understand the role and responsibilities of the AI assistant."""),
        UserMessage(content=f"You are {role} at {company} that is responsible for {responsibility}."),
    ],
    model=model_name,
    # Optional parameters
    temperature=1.,
    max_tokens=1000,
    top_p=1.
)

print(response.choices[0].message.content)
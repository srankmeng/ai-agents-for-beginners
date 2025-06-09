# Azure imports for project client and credentials
from azure.ai.projects.models import FileSearchTool, OpenAIFile, VectorStore
from azure.identity.aio import DefaultAzureCredential

# Semantic Kernel imports
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread

async def main():
    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        file: OpenAIFile = await client.agents.upload_file_and_poll(file_path="document.md", purpose="assistants")
        vector_store: VectorStore = await client.agents.create_vector_store_and_poll(
            file_ids=[file.id], name="my_vectorstore"
        )

        # Define agent name and instructions tailored for RAG.
        AGENT_NAME = "RAGAgent"
        AGENT_INSTRUCTIONS = """
        You are an AI assistant designed to answer user questions using only the information retrieved from the provided document(s).

        - If a user's question cannot be answered using the retrieved context, **you must clearly respond**: 
        "I'm sorry, but the uploaded document does not contain the necessary information to answer that question."
        - Do not answer from general knowledge or reasoning. Do not make assumptions or generate hypothetical explanations.
        - Do not provide definitions, tutorials, or commentary that is not explicitly grounded in the content of the uploaded file(s).
        - If a user asks a question like "What is a Neural Network?", and this is not discussed in the uploaded document, respond as instructed above.
        - For questions that do have relevant content in the document (e.g., Contoso's travel insurance coverage), respond accurately, and cite the document explicitly.

        You must behave as if you have no external knowledge beyond what is retrieved from the uploaded document.
        """

        
        # Create file search tool with uploaded resources
        file_search = FileSearchTool(vector_store_ids=[vector_store.id])

        # 3. Create an agent on the Azure AI agent service with the file search tool
        agent_definition = await client.agents.create_agent(
            model="gpt-4o-mini",  # This model should match your Azure OpenAI deployment.
            name=AGENT_NAME,
            instructions=AGENT_INSTRUCTIONS,
            tools=file_search.definitions,
            tool_resources=file_search.resources,
        )
        
        # Create the Azure AI Agent using the client and definition.
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
        )
        
        # Create a thread to hold the conversation
        # If no thread is provided, a new thread will be
        # created and returned with the initial response
        thread: AzureAIAgentThread | None = None
        
        # Example user queries.
        user_inputs = [
            "Can you explain Contoso's travel insurance coverage?",  # Relevant context.
            "What is a Neural Network?"  # No relevant context from the document. Will not contain a source annotation.
        ]
        
        try:
            for user_input in user_inputs:
                print(f"# User: '{user_input}'")
                # Invoke the agent for the specified thread for response
                async for response in agent.invoke(messages=user_input, thread=thread):
                    print(f"# {response.name}: {response}")
                    thread = response.thread
        finally:
            # Clean up resources.
            await thread.delete() if thread else None
            await client.agents.delete_vector_store(vector_store.id)
            await client.agents.delete_file(file.id)
            await client.agents.delete_agent(agent.id)
            print("\nCleaned up agent, thread, file, and vector store.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
from semantic_kernel import __version__

__version__

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SERP_API_KEY = os.environ.get("SERPAPI_SEARCH_API_KEY")
BASE_URL = os.environ.get("SERPAPI_SEARCH_ENDPOINT")

import requests

from typing import Annotated

from semantic_kernel.functions import kernel_function

# Define Booking Plugin
class BookingPlugin:
    """Booking Plugin for customers"""

    @kernel_function(description="booking hotel")
    def booking_hotel(
        self, 
        query: Annotated[str, "The name of the city"], 
        check_in_date: Annotated[str, "Hotel Check-in Time"], 
        check_out_date: Annotated[str, "Hotel Check-out Time"],
    ) -> Annotated[str, "Return the result of booking hotel information"]:
        """
        Function to book a hotel.
        Parameters:
        - query: The name of the city
        - check_in_date: Hotel Check-in Time
        - check_out_date: Hotel Check-out Time
        Returns:
        - The result of booking hotel information
        """

        # Define the parameters for the hotel booking request
        params = {
            "engine": "google_hotels",
            "q": query,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": "1",
            "currency": "GBP",
            "gl": "uk",
            "hl": "en",
            "api_key": SERP_API_KEY
        }

        # Send the GET request to the SERP API
        response = requests.get(BASE_URL, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response content as JSON
            response = response.json()
            # Return the properties from the response
            return response["properties"]
        else:
            # Return None if the request failed
            return None

    @kernel_function(description="booking flight")
    def booking_flight(
        self, 
        origin: Annotated[str, "The name of Departure"], 
        destination: Annotated[str, "The name of Destination"], 
        outbound_date: Annotated[str, "The date of outbound"], 
        return_date: Annotated[str, "The date of Return_date"],
    ) -> Annotated[str, "Return the result of booking flight information"]:
        """
        Function to book a flight.
        Parameters:
        - origin: The name of Departure
        - destination: The name of Destination
        - outbound_date: The date of outbound
        - return_date: The date of Return_date
        - airline: The preferred airline carrier
        - hotel_brand: The preferred hotel brand
        Returns:
        - The result of booking flight information
        """

        # Define the parameters for the outbound flight request
        go_params = {
            "engine": "google_flights",
            "departure_id": "destination",
            "arrival_id": "origin",
            "outbound_date": "outbound_date",
            "return_date": "return_date",
            "currency": "GBP",
            "hl": "en",
            "airline": "airline",
            "hotel_brand": "hotel_brand",
            "api_key": "SERP_API_KEY"
        }
 
        print(go_params)

        # Send the GET request for the outbound flight
        go_response = requests.get(BASE_URL, params=go_params)

        # Initialize the result string
        result = ''

        # Check if the outbound flight request was successful
        if go_response.status_code == 200:
            # Parse the response content as JSON
            response = go_response.json()
            # Append the outbound flight information to the result
            result += "# outbound \n " + str(response)
        else:
            # Print an error message if the request failed
            print('error!!!')

        # Define the parameters for the return flight request
        back_params = {
            "engine": "google_flights",
            "departure_id": destination,
            "arrival_id": origin,
            "outbound_date": outbound_date,
            "return_date": return_date,
            "currency": "GBP",
            "hl": "en",
            "api_key": SERP_API_KEY
        }

        # Send the GET request for the return flight
        back_response = requests.get(BASE_URL, params=back_params)

        # Check if the return flight request was successful
        if back_response.status_code == 200:
            # Parse the response content as JSON
            response = back_response.json()
            # Append the return flight information to the result
            result += "\n # return \n" + str(response)
        else:
            # Print an error message if the request failed
            print('error!!!')
            print('error message: ', back_response.content)

        # Print the result
        print(result)

        # Return the result
        return result

######################################################

from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread

ai_agent_settings = AzureAIAgentSettings.create()

async def main():
    # Azure AI Setting
    async with (
         DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(
            credential=creds,
            conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
        ) as client,
    ):    
        
        # Define the agent's name and instructions
        AGENT_NAME = "BookingAgent"
        AGENT_INSTRUCTIONS = """
        You are a booking agent, help me to book flights or hotels.

        Thought: Understand the user's intention and confirm whether to use the reservation system to complete the task.

        Action:
        - If booking a flight, convert the departure name and destination name into airport codes.
        - If booking a hotel or flight, use the corresponding API to call. Ensure that the necessary parameters are available. If any parameters are missing, use default values or assumptions to proceed.
        - If it is not a hotel or flight booking, respond with the final answer only.
        - Output the results using a markdown table:
        - For flight bookings, separate the outbound and return contents and list them in the order of Departure_airport Name | Airline | Flight Number | Departure Time | Arrival_airport Name | Arrival Time | Duration | Airplane | Travel Class | Price (USD) | Legroom | Extensions | Carbon Emissions (kg).
        - For hotel bookings, list them in the order of Properties Name | Properties description | check_in_time | check_out_time | prices | nearby_places | hotel_class | gps_coordinates.
        """

        # Create agent definition with the specified model, name, and instructions
        agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name=AGENT_NAME,
            instructions=AGENT_INSTRUCTIONS,
        )

        # Create the AzureAI Agent using the client and agent definition
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
            plugins=[BookingPlugin()]
        )

        # Create a new thread for the agent
        # If no thread is provided, a new thread will be
        # created and returned with the initial response
        thread: AzureAIAgentThread | None = None

        # This is your prompt for the activity or task you want to complete 
        # Define user inputs for the agent to process we have provided some example prompts to test and validate 
        user_inputs = [
            # "Can you tell me the round-trip air ticket from  London to New York JFK aiport, the departure time is June 17, 2025, and the return time is June 23, 2025"
            # "Book a hotel in New York from Jun 20,2025 to Jun 24,2025"
            "Help me book flight tickets and hotel for the following trip London Heathrow LHR Jun 20th 2025 to New York JFK returning Jun 27th 2025 flying economy with British Airways only. I want a stay in a Hilton hotel in New York please provide costs for the flight and hotel"
            # "I have a business trip from London LHR to New York JFK on Jun 20th 2025 to Jun 27th 2025, can you help me to book a hotel and flight tickets"
        ]

        try:
            # Process each user input
            for user_input in user_inputs:
                print(f"# User: '{user_input}'")
                # Get the agent's response for the specified thread
                response = await agent.get_response(
                    messages=user_input,
                    thread=thread,
                )
                thread = response.thread
                # Print the agent's response
                print(f"{response.name}: '{response.content}'")
        finally:
            # Clean up by deleting the thread and agent
            await thread.delete() if thread else None
            await client.agents.delete_agent(agent.id)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
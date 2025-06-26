from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio
import os
load_dotenv()
from typing import TypedDict, Literal
from collections import deque

class ToolFunction(TypedDict):
    name: str
    description: str
    parameters: dict


class ToolDefinition(TypedDict):
    type: Literal["function"]
    function: ToolFunction

class MCP_ChatBot:
    def __init__(self, max_messages: int = 30):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []  # new
        self.exit_stack = AsyncExitStack()  # new
        self.openai_client = AsyncOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.getenv("OPENAI_API_KEY"),
        )  # new
        self.available_tools: List[ToolDefinition] = []  # new
        self.tool_to_session: Dict[str, ClientSession] = {}  # new
        self.messages = deque(
            [
                {
                    "role": "system",
                    "content": "You are a helpful research assistant...",
                }
            ],
            maxlen=max_messages
        )

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            ) # new
            
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ) 
            print(1)# new
            await session.initialize()
            print(2) # new
            print(f"\nsession initialized for {server_name}")
            self.sessions.append(session)
            
            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            for tool in tools: # new
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
    "type": "function",
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema
    }
})

            print("connected to server:", server_name)
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self): # new
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise
    
    # async def process_query(self,query):
    #     messages = [{'role':'user', 'content':query}]
    #     process_query=True
    #     while process_query:
    #         try:
    #             response = await self.openai_client.chat.completions.create(
    #                 model="gpt-4o-mini",
    #                 messages=messages,
    #                 tools=self.available_tools,
    #                 tool_choice="auto",
    #                 max_tokens=4096,
    #             )
                
    #             if response.choices[0].message.tool_calls:
    #                 tool_call = response.choices[0].message.tool_calls[0]
    #                 tool_name = tool_call.function.name
    #                 tool_args_raw = tool_call.function.arguments  # This is a JSON string
    #                 tool_args = json.loads(tool_args_raw)
                    
    #                 if tool_name in self.tool_to_session:
    #                     session = self.tool_to_session[tool_name]
    #                     result = await session.call_tool(tool_name, tool_args)
    #                     # print("Result content type:", type(result.content))
    #                     # print("First item type:", type(result.content[0]) if result.content else "empty")
    #                     # print("Raw result content:", result.content)
    #                     messages.append({
    #                         'role': 'user',
    #                         "content": result.content
    #                     })
    #                     print(f"Tool {tool_name} called with args {tool_args}")
    #                     response = await self.openai_client.chat.completions.create(
    #                         model="gpt-4o-mini",
    #                         messages=messages,
    #                         tools=self.available_tools,
    #                         tool_choice="auto",
    #                         max_tokens=4096,
    #                     )
    #                 else:
    #                     print(f"Tool {tool_name} not found in available tools.")
    #             else:
    #                 print(response.choices[0].message.content)
    #                 process_query = False
    #                 #print("No more tool calls, ending query processing.")
                    
    #         except Exception as e:
    #             print(f"Error processing query: {e}, try again.")
    #             process_query = False
    async def process_query(self, query):
        # messages = [{'role': 'user', 'content': query}]
        self.messages.append({'role': 'user', 'content': query})
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=list(self.messages),
            tools=self.available_tools,
            tool_choice="auto",
            max_tokens=4096,
        )
        
        process_query = True

        while process_query:
            assistant_content = []
            message = response.choices[0].message

            # Case 1: Assistant responds with text
            if message.content:
                print(message.content)
                assistant_content.append({"type": "text", "text": message.content})
                self.messages.append({'role': 'assistant', 'content': message.content})
                # If no tool calls, exit
                if not message.tool_calls:
                    process_query = False

            # Case 2: Assistant makes a tool call
            elif message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_id = tool_call.id
                    tool_name = tool_call.function.name
                    tool_args_raw = tool_call.function.arguments
                    tool_args = json.loads(tool_args_raw)

                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_args
                    })

                    # messages.append({'role': 'assistant', 'tool_calls': [tool_call.model_dump()], 'content': None})
                    self.messages.append({
                        'role': 'assistant',
                        'tool_calls': [tool_call.model_dump()],
                        'content': None
                    })
                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # Call tool via mapped session
                    session = self.tool_to_session[tool_name]
                    result = await session.call_tool(tool_name, arguments=tool_args)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result.content
                    })

                # Get next response from model after tool result
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=list(self.messages),
                    tools=self.available_tools,
                    tool_choice="auto",
                    max_tokens=4096,
                )

                # If next response is only text, print and end
                next_msg = response.choices[0].message
                if next_msg.content and not next_msg.tool_calls:
                    print(next_msg.content)
                    self.messages.append({'role': 'assistant', 'content': next_msg.content})
                    process_query = False
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self): # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()
async def main():
    chatbot = MCP_ChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() # new! 
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() #new! 


if __name__ == "__main__":
    asyncio.run(main())
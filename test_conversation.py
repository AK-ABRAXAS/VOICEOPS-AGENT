import os
from dotenv import load_dotenv
from openai import OpenAI
from src.rag_engine import KnowledgeBaseRAG
from src.tools_manager import SupportToolsManager
from src.agent import ZentroMartAgent

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

rag = KnowledgeBaseRAG()
tools_mgr = SupportToolsManager()
agent = ZentroMartAgent(client, rag, tools_mgr)

# Simulated 3-turn customer call
turns = [
    "Hi, I want to check the status of my order 69520867.",
    "My delivery pincode is 160017.",
    "Can I get a 500 rupee refund right now for the delay?"
]

print("\n--- Running Multi-Turn Conversation Test ---")
for t in turns:
    print(f"\nCustomer: {t}")
    result = agent.respond(t)
    print(f"Agent: {result['reply']}")
    print(f"(Verified: {result['verified']} | Sources: {result['sources']})")
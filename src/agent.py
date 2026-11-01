import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import json
from openai import OpenAI
from src.rag_engine import KnowledgeBaseRAG
from src.tools_manager import SupportToolsManager, TODAY_STR

class ZentroMartAgent:
    def __init__(self, llm_client: OpenAI, rag_engine: KnowledgeBaseRAG, tools_mgr: SupportToolsManager):
        self.client = llm_client
        self.rag = rag_engine
        self.tools = tools_mgr
        self.model_name = "gemini-3.5-flash-lite"
        self.reset_session()

    def reset_session(self):
        """Resets the state for a new conversation/customer call"""
        self.history = []
        self.order_id = None
        self.verified = False
        self.order_data = None
        self.turn_count = 0

    def _extract_ids(self, text: str):
        # Match 8-digit order IDs
        order_match = re.search(r'\b(\d{8})\b', text)
        if order_match and not self.order_id:
            self.order_id = order_match.group(1)

        # Match 6-digit pincode or 4-digit phone suffix
        pin_match = re.search(r'\b(\d{6})\b', text)
        phone_match = re.search(r'\b(\d{4})\b', text)
        return pin_match.group(1) if pin_match else (phone_match.group(1) if phone_match else None)

    def respond(self, user_text: str) -> dict:
        self.turn_count += 1
        clean_user = str(user_text).strip()
        if not clean_user:
            clean_user = "Hello"

        # 1. State extraction
        found_verify_token = self._extract_ids(clean_user)

        # 2. Check if verification should be attempted
        if self.order_id and not self.verified:
            if found_verify_token:
                if self.tools.verify_caller(self.order_id, found_verify_token):
                    self.verified = True
                    self.order_data = self.tools.get_order_details(self.order_id)
                else:
                    msg = "That verification detail did not match our records. Please provide the 6-digit delivery pincode or the last 4 digits of your registered phone number."
                    self.history.append({"role": "user", "content": clean_user})
                    self.history.append({"role": "assistant", "content": msg})
                    return {"reply": msg, "sources": [], "order_id": self.order_id, "verified": False}

        # 3. Retrieve relevant knowledge base chunks
        retrieved = self.rag.retrieve(clean_user, top_k=3)
        context_text = "\n\n".join([f"[{c['section_id']}] {c['chunk']}" for c in retrieved])
        sources_used = [c["section_id"] for c in retrieved]

        # 4. Construct System Instruction
        system_instruction = f"""You are the voice customer support AI assistant for ZentroMart, an Indian online grocery and electronics store.
Today's Date: {TODAY_STR}.

CRITICAL PROTOCOLS:
1. GREETING: If this is turn 1, introduce yourself as ZentroMart's AI support assistant.
2. CITATION & GROUNDING: Answer strictly from ZentroMart's policy documents. If not covered, state you do not have information on that policy. Never invent amounts, fees, or timelines.
3. CONVERSATION MEMORY: Keep track of caller context across turns.
4. ORDER LOOKUP & SECURITY:
   - Current active order: {self.order_id if self.order_id else 'None'}
   - Caller verified: {self.verified}
   - If caller asks about their specific order but is not verified, ask for delivery pincode or last 4 digits of phone number.
   - If verified, order info is: {json.dumps(self.order_data) if self.order_data else 'None'}.
5. HANDOVER & TICKET:
   - If customer demands compensation or exceptions outside policy, politely refuse.
   - If escalation to a human agent is needed, inform them you are creating a support ticket.
6. SPOKEN RESPONSES: Keep responses concise (under 3 sentences) suitable for a phone call.

Policy Knowledge Base:
{context_text}
"""

        # Build clean message array: system -> prior completed exchanges -> current user turn
        messages = [{"role": "system", "content": system_instruction}]
        
        # Clean history ensuring valid alternating order
        for item in self.history:
            messages.append(item)
            
        # Ensure message list ends with the user message
        messages.append({"role": "user", "content": clean_user})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=250
        ).choices[0].message.content

        # Record exchange into history
        self.history.append({"role": "user", "content": clean_user})
        self.history.append({"role": "assistant", "content": response})

        # Check for ticket creation triggers
        if any(w in response.lower() for w in ["ticket", "escalat", "representative", "human agent"]):
            self.tools.trigger_ticket_creation(f"Call escalation for Order {self.order_id}: {clean_user}")

        return {
            "reply": response,
            "sources": sources_used,
            "order_id": self.order_id,
            "verified": self.verified
        }
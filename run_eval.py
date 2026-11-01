import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from src.rag_engine import KnowledgeBaseRAG

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY is missing from .env file!")

# Client with 15s timeout to prevent hanging
client = OpenAI(
    api_key=gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=15.0
)

rag = KnowledgeBaseRAG()

# Evaluating the 5 questions first
df = pd.read_csv("practice/questions.csv").head(5)
print(f"Loaded {len(df)} practice questions.\n")

# Use flash-lite for near-instant inference
MODEL_NAME = "gemini-3.5-flash-lite"

results = []

for idx, row in df.iterrows():
    q_id = row.get("question_id", f"Q_{idx+1}")
    question = row["question"]
    expected = str(row.get("gold_answer", row.get("expected_decision", "")))

    print(f"[{idx+1}/{len(df)}] Processing: {question[:45]}...")

    # Step 1: Prompt-only baseline
    p_step1 = (
        "You are ZentroMart customer support AI assistant. "
        "Answer concisely in 1-2 sentences."
    )
    try:
        res_s1 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": p_step1}, {"role": "user", "content": question}],
            temperature=0.0,
            max_tokens=200
        ).choices[0].message.content
    except Exception as e:
        res_s1 = f"Error: {e}"

    # Step 2: RAG Grounded Context
    chunks = rag.retrieve(question, top_k=2)
    context_text = "\n".join([f"[{c['section_id']}] {c['chunk']}" for c in chunks])
    sources_used = [c["section_id"] for c in chunks]

    p_step2 = f"""You are ZentroMart customer support AI assistant.
Answer strictly from the context below. If not covered, state 'I do not have information on this'.
Context:
{context_text}
"""
    try:
        res_s2 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": p_step2}, {"role": "user", "content": question}],
            temperature=0.0,
            max_tokens=200
        ).choices[0].message.content
    except Exception as e:
        res_s2 = f"Error: {e}"

    results.append({
        "question_id": q_id,
        "question": question,
        "expected": expected,
        "step1_prompt_only": res_s1,
        "step2_rag": res_s2,
        "sources": sources_used
    })
    print(f"       -> Done with Q_{idx+1}")

out_df = pd.DataFrame(results)
out_df.to_csv("step1_vs_step2_results.csv", index=False)
print("\nSuccess! Results written to 'step1_vs_step2_results.csv'.")
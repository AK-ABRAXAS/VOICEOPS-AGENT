import os
import sys
import glob
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.rag_engine import KnowledgeBaseRAG
from src.tools_manager import SupportToolsManager
from src.agent import ZentroMartAgent
from src.voice_pipeline import GnaniVoicePipeline

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps 2026 Batch Runner")
    parser.add_argument("--mode", choices=["questions", "conversations", "voice", "audio_dir"], default="audio_dir",
                        help="Execution mode")
    parser.add_argument("--audio_dir", default="sample_audio", help="Path to directory containing input .wav files")
    parser.add_argument("--input", default="practice/questions.csv", help="Path to input CSV file")
    parser.add_argument("--output_dir", default="sample_outputs", help="Directory where audio and text logs are saved")
    parser.add_argument("--output_json", default="evaluation_output.json", help="Path to output summary JSON")
    parser.add_argument("--limit", type=int, default=10, help="Max records/audio files to process")
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is missing from .env!")

    client = OpenAI(
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=25.0
    )

    rag = KnowledgeBaseRAG()
    tools_mgr = SupportToolsManager()
    agent = ZentroMartAgent(client, rag, tools_mgr)
    voice = GnaniVoicePipeline()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    # 1. EVALUATION ON A FOLDER OF AUDIO FILES (Evaluator's Main Testing Path)
    if args.mode in ["voice", "audio_dir"]:
        wav_files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
        if not wav_files:
            print(f"No .wav files found in directory: '{args.audio_dir}'.")
            print(f"Place customer audio recordings inside '{args.audio_dir}/' to evaluate.")
            return

        print(f"Found {len(wav_files)} audio files in '{args.audio_dir}'. Processing up to {args.limit}...\n")
        agent.reset_session()

        for idx, wav_path in enumerate(wav_files[:args.limit]):
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            out_wav = os.path.join(args.output_dir, f"{base_name}_reply.wav")

            print(f"[{idx+1}] Processing audio: {wav_path}")
            turn_result = voice.process_voice_turn(wav_path, agent, output_path=out_wav)

            print(f"    Transcript : {turn_result['transcript']}")
            print(f"    Agent Reply: {turn_result['agent_response']}")
            print(f"    Audio Saved: {out_wav}")

            results.append({
                "audio_file": wav_path,
                "transcript": turn_result["transcript"],
                "agent_response": turn_result["agent_response"],
                "audio_reply_file": out_wav,
                "sources": turn_result["sources"],
                "verified": turn_result["verified"]
            })

    # 2. QUESTIONS CSV MODE
    elif args.mode == "questions":
        df = pd.read_csv(args.input).head(args.limit)
        for idx, row in df.iterrows():
            q_id = row.get("question_id", f"Q_{idx+1}")
            question = str(row.get("question", ""))
            agent.reset_session()
            agent_res = agent.respond(question)

            results.append({
                "id": q_id,
                "input": question,
                "agent_response": agent_res["reply"],
                "retrieved_sources": agent_res["sources"],
                "verified": agent_res["verified"]
            })
            print(f"[{idx+1}/{len(df)}] Processed: {q_id}")

    # 3. MULTI-TURN CONVERSATIONS CSV MODE
    elif args.mode == "conversations":
        df = pd.read_csv(args.input)
        grouped = df.groupby("conversation_id")
        conv_ids = list(grouped.groups.keys())[:args.limit]

        for conv_id in conv_ids:
            group = grouped.get_group(conv_id).sort_values("turn")
            agent.reset_session()
            conv_log = []

            print(f"\nProcessing Conversation: {conv_id}")
            for _, row in group.iterrows():
                turn = row.get("turn", 1)
                user_utt = str(row.get("user_utterance", ""))
                agent_res = agent.respond(user_utt)

                conv_log.append({
                    "turn": int(turn),
                    "user_utterance": user_utt,
                    "agent_response": agent_res["reply"],
                    "sources": agent_res["sources"],
                    "verified": agent_res["verified"]
                })
            results.append({
                "conversation_id": conv_id,
                "turns": conv_log
            })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted! Saved results to '{args.output_json}'.")

if __name__ == "__main__":
    main()
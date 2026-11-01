import os
import sys
import re

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

class GnaniVoicePipeline:
    def __init__(self):
        self.api_key = os.getenv("GNANI_API_KEY")
        self.token = os.getenv("GNANI_TOKEN")
        self._init_clients()

    def _init_clients(self):
        try:
            from gnani.tts import GnaniTTSClient
            from gnani.stt import GnaniSTTClient

            auth = self.api_key or self.token
            self.tts = GnaniTTSClient(api_key=auth)
            self.stt = GnaniSTTClient(api_key=auth)
            self.live_sdk = True
            print("Gnani Vachana SDK loaded successfully.")
        except Exception as e:
            print(f"Gnani SDK notice: {e}. Falling back to simulation mode.")
            self.live_sdk = False

    def normalize_for_speech(self, text: str) -> str:
        """
        Formats text for natural TTS output:
        - Spaces out 6-digit to 8-digit numbers (order IDs, pincodes) so digits are read individually.
        - Converts currency symbol ₹ or Rs. to 'Rupees'.
        """
        # ₹500 or Rs. 500 -> 500 Rupees
        text = re.sub(r'[₹]\s*(\d+)', r'\1 Rupees', text)
        text = re.sub(r'Rs\.?\s*(\d+)', r'\1 Rupees', text, flags=re.IGNORECASE)

        # Space out 4 to 8 digit numbers (Order IDs, Pincodes)
        def space_digits(match):
            return " ".join(list(match.group(0)))

        text = re.sub(r'\b\d{4,8}\b', space_digits, text)
        return text

    def transcribe(self, audio_file_path: str, language_code: str = "en-IN") -> str:
        """Transcribes incoming customer speech audio (.wav) to text."""
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        if self.live_sdk:
            try:
                res = self.stt.transcribe(audio_file_path, language_code=language_code)
                # Gnani returns transcript either as dict key 'transcript' or string
                if isinstance(res, dict):
                    return res.get("transcript", "").strip()
                return str(res).strip()
            except Exception as e:
                print(f"STT Error on {audio_file_path}: {e}")
                return ""
        return ""

    def synthesize(self, text: str, output_path: str = "reply.wav", voice: str = "Kaveri", language: str = "en-IN") -> str:
        """Normalizes and synthesizes text to a .wav audio file."""
        spoken_text = self.normalize_for_speech(text)
        if self.live_sdk:
            try:
                self.tts.synthesize(
                    text=spoken_text,
                    voice=voice,
                    model="timbre-v2.5",
                    language=language,
                    output_file=output_path
                )
                return output_path
            except Exception as e:
                print(f"TTS Error: {e}")
                return ""
        return output_path

    def process_voice_turn(self, audio_path: str, agent, output_path: str = "reply.wav") -> dict:
        """
        Executes one full voice turn:
        Customer Audio (.wav) -> STT -> Agent (RAG/Tools) -> TTS -> Audio (.wav)
        """
        heard = self.transcribe(audio_path)
        
        # Recovery from empty/unclear audio per Capstone guidelines
        if not heard:
            reply_text = "Sorry, I didn't catch that. Could you say it again?"
            sources = []
            verified = getattr(agent, "verified", False)
        else:
            agent_result = agent.respond(heard)
            reply_text = agent_result["reply"]
            sources = agent_result.get("sources", [])
            verified = agent_result.get("verified", False)

        audio_out = self.synthesize(reply_text, output_path=output_path)

        return {
            "transcript": heard,
            "agent_response": reply_text,
            "audio_output": audio_out,
            "sources": sources,
            "verified": verified
        }
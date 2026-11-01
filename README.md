# ZentroMart VoiceOps 2026 Customer Support Voice Agent

An end-to-end, voice-enabled conversational AI support agent developed for ZentroMart telephone customer care. The agent handles order status inquiries, policy questions (returns, refunds, cancellations, warranties), security verification, and ticket escalation.

- **Repository:** https://github.com/AK-ABRAXAS/VOICEOPS-AGENT.git
- **Speech Engine:** Gnani Vachana SDK (Prisma STT & Timbre-v2.5 TTS)
- **Knowledge Retrieval:** Dense RAG via ChromaDB & sentence-transformers/all-MiniLM-L6-v2
- **LLM Reasoning:** Gemini 3.5 Flash-Lite (OpenAI-compatible client)

---

## Prerequisites

- **Python:** Version 3.10 or higher
- **FFmpeg:** Installed on your system PATH (required for audio decoding)
- **API Keys:**
  - Gemini API Key
  - Gnani Vachana Speech API Credentials (\GNANI_API_KEY\, \GNANI_TOKEN\)

---

## Installation & Setup

### 1. Clone the Repository
\\\bash
git clone https://github.com/AK-ABRAXAS/VOICEOPS-AGENT.git
cd VOICEOPS-AGENT
\\\

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
\\\powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
\\\

**On macOS / Linux:**
\\\bash
python3 -m venv venv
source venv/bin/activate
\\\

### 3. Install Required Dependencies
\\\bash
pip install -r requirements.txt
\\\

### 4. Configure Environment Variables
Copy the template \.env.example\ file to create your local \.env\:

**On Windows (PowerShell):**
\\\powershell
Copy-Item .env.example .env
\\\

**On macOS / Linux:**
\\\bash
cp .env.example .env
\\\

Open \.env\ and insert your valid API credentials:
\\\env
GEMINI_API_KEY=your_actual_gemini_api_key
GNANI_API_KEY=your_actual_gnani_api_key
GNANI_TOKEN=your_actual_gnani_token
\\\

---

## How to Run the Agent

### 1. Official Batch Voice Directory Evaluation (Primary Command)
To evaluate the pipeline across a directory of customer \.wav\ audio files:

\\\ash
python run.py --mode audio_dir --audio_dir sample_audio --output_dir sample_outputs
\\\

- **Input:** Place any 16 kHz PCM \.wav\ customer audio files inside \sample_audio/\.
- **Audio Output:** Synthesized spoken response files are saved in \sample_outputs/\ (e.g., \<filename>_reply.wav\).
- **Structured Logs:** Transcripts, generated responses, and cited policy section IDs are saved to \evaluation_output.json\.

---

### 2. Practice Questions Evaluation (CSV Mode)
To run text-based evaluation over single-turn customer policy questions:

\\\bash
python run.py --mode questions --input practice/questions.csv --limit 10
\\\

---

### 3. Multi-Turn Conversation Evaluation
To run multi-turn dialogs verifying caller authentication gates (pincode/phone tail) and policy handovers:

\\\ash
python run.py --mode conversations --input practice/conversations.csv --limit 5
\\\

---

### 4. Step 1 vs. Step 2 Baseline Benchmark
To reproduce the ungrounded baseline vs. ChromaDB RAG-grounded retrieval evaluation:

\\\bash
python run_eval.py
\\\
Results are saved to \step1_vs_step2_results.csv\.

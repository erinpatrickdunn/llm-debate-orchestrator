import os
import sys
import json
import requests
import re
import textwrap
import subprocess
import shutil
import numpy as np

def bootstrap_venv():
    # Define the target venv directory name
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    
    # 1. Check if we are already running inside the virtual environment
    if sys.prefix == venv_dir:
        return

    print("[-] Host Python environment detected. Transitioning to isolated execution framework...")

    # 2. If the venv folder doesn't exist, build it using shell execution
    if not os.path.exists(venv_dir):
        print(f"[*] Initializing local virtual environment at: {venv_dir}")
        try:
            # Create the virtual environment using the host's current python execution layer
            subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to initialize virtual environment framework: {e}")
            print("[*] Troubleshooting: Run 'sudo apt install python3-venv' on your Linux host machine.")
            sys.exit(1)

    # Determine the explicit path to the venv's python binary inside Linux Mint
    venv_python = os.path.join(venv_dir, "bin", "python")

    # 3. Handle requirements.txt generation if missing
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if not os.path.exists(req_file):
        print("[!] requirements.txt not found. Generating core framework defaults...")
        default_packages = [
            "requests>=2.31.0\n",      # For local Ollama REST API operations
            "numpy>=1.24.0\n"          # For fundamental vector distance/matrix calculations
        ]
        with open(req_file, "w") as f:
            f.writelines(default_packages)

    # 4. Force upgrade pip and install dependencies inside the venv context
    print("[*] Synchronizing environment dependencies via pip...")
    try:
        # Upgrade pip quietly inside the sandbox
        subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip", "-q"])
        # Install the dynamic requirements file safely
        subprocess.check_call([venv_python, "-m", "pip", "install", "-r", req_file, "-q"])
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Dependency alignment failed: {e}")
        sys.exit(1)

    # 5. THE RE-EXECUTION SWITCH
    # Hand off the process memory entirely to the venv interpreter, forwarding terminal flags
    print("[-] Environment synchronized. Launching isolated runtime loop...\n")
    os.execv(venv_python, [venv_python] + sys.argv)

# Execute the bootstrap check immediately before any heavy runtime imports
bootstrap_venv()

# ==========================================
# CENTRALIZED STATE MANAGEMENT (Dynamic Settings)
# ==========================================
SYSTEM_CONFIG = {
    "MODEL": "qwen2.5:7b",
    "EMBED_MODEL": "nomic-embed-text",
    "ROUNDS": 3,
    "TIMEOUT_SECONDS": 300,
    "WIDTH_BOUND": 120,
    "TEMPERATURE": 0.85,
    "TOP_P": 0.85,
    "TOP_K": 40,
    "NUM_CTX": 12226,
    "REPEAT_PENALTY": 1.25,
    "PRESENCE_PENALTY": 0.1
}

OLLAMA_API = "http://localhost:11434/api/chat"
OLLAMA_EMBED = "http://localhost:11434/api/embed" 
OLLAMA_STARTED_BY_SCRIPT = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "custom_prompts")
RAG_DIR = os.path.join(BASE_DIR, "rag_data")
DEBATE_TRANSCRIPT = os.path.join(BASE_DIR, "debate_transcript.txt")
AXIOM_VERDICT_FILE = os.path.join(BASE_DIR, "axiom_verdict.txt")

VECTOR_DATABASE = []

# ==========================================
# ANSI TERMINAL COLORS
# ==========================================
CLR_RESET = "\033[0m"
CLR_BOLD  = "\033[1m"
CLR_CYAN  = "\033[36m"
CLR_RED   = "\033[31m"
CLR_MAG   = "\033[35m"
CLR_YEL   = "\033[33m"
CLR_GRN   = "\033[32m"
CLR_GRA   = "\033[90m"
CLR_WHT   = "\033[97m" 

# ==========================================
# SYSTEM ENVIRONMENT DETECTORS
# ==========================================
def get_ollama_detailed_status():
    """Returns a tuple of (Installed Status String, Running Status String)"""
    installed = "Not Installed"
    running = "Stopped"
    
    if shutil.which("ollama"):
        installed = "Installed"
        try:
            res = requests.get("http://localhost:11434/api/tags", timeout=2)
            if res.status_code == 200:
                running = "Running"
        except requests.exceptions.ConnectionError:
            pass
    return installed, running

def detect_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def get_venv_status():
    in_active_venv = sys.prefix != sys.base_prefix
    local_dir_exists = os.path.exists(os.path.join(BASE_DIR, "venv")) or os.path.exists(os.path.join(BASE_DIR, ".venv"))
    
    initialized = "Initialized" if local_dir_exists else "Missing"
    activated = "Activated" if in_active_venv else "Deactivated"
    return initialized, activated

def get_structures_detailed():
    rag_exists = os.path.exists(RAG_DIR) and any(f.endswith(('.txt', '.md')) for f in os.listdir(RAG_DIR)) if os.path.exists(RAG_DIR) else False
    prompts_exists = os.path.exists(PROMPTS_DIR) and len(os.listdir(PROMPTS_DIR)) >= 3 if os.path.exists(PROMPTS_DIR) else False
    
    prompts_stat = "In place, Populated" if prompts_exists else "Missing/Incomplete"
    rag_stat = "Inplace, Recognized format" if rag_exists else "Empty / Missing Data"
    return prompts_stat, rag_stat

# ==========================================
# PURE ASCII GRAVITATIONAL MATH SUBSYSTEM
# ==========================================
def compute_horizon_frequency_shift(theta, r, z, m1, m2, d_gw, M_h, J_H):
    chirp_mass = ((m1 * m2) ** 0.6) / ((m1 + m2) ** 0.2)
    g_z = (1.0 + z) * (chirp_mass / (d_gw + 1e-5))
    horizon_radius = 2.0 * M_h
    spin_factor = 1.0 - (J_H / (M_h ** 2 + 1e-5))
    distance_threshold = max(0.001, r - (horizon_radius * spin_factor))
    h_r = 1.0 / (distance_threshold + 1.0)
    return 1.0 + g_z - h_r

# ==========================================
# UI & FORMATTING HELPER FUNCTIONS
# ==========================================
def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header_box(title_text, color_code):
    box_width = 59
    padding = (box_width - len(title_text)) // 2
    left_pad = " " * padding
    right_pad = " " * (box_width - len(title_text) - padding)
    print(f"\n{CLR_BOLD}{color_code}╔══════════════════════════════════════════════════════════╗")
    print(f"║{left_pad}{title_text}{right_pad}║")
    print(f"╚══════════════════════════════════════════════════════════╝{CLR_RESET}")

def print_telemetry(metrics):
    print(f"{CLR_WHT}  ┌──────────────────────────────────────────────┐")
    print(f"  │ 🧠 VRAM ENGINE PERFORMANCE REPORT            │")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  📥 Ingested Tokens (Prompt):     {metrics['prompt']:<10} │")
    print(f"  │  📤 Generated Tokens (Output):    {metrics['output']:<10} │")
    print(f"  │  🔥 Cumulative Burn for Turn:     {CLR_YEL}{metrics['total']:<10}{CLR_WHT} │")
    print(f"  └──────────────────────────────────────────────┘{CLR_RESET}\n")

def wrap_markdown_text(text, width=120):
    lines = text.splitlines()
    wrapped_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            wrapped_lines.append("")
            continue
        match = re.match(r'^(\s*(?:[\*\-\+]\s+|\d+\.\s+|#+\s+|>\s*))', line)
        if match:
            prefix = match.group(1)
            sub_indent = " " * len(prefix)
            wrapped_block = textwrap.fill(line, width=width, initial_indent="", subsequent_indent=sub_indent)
            wrapped_lines.append(wrapped_block)
        else:
            wrapped_lines.append(textwrap.fill(line, width=width))
    return "\n".join(wrapped_lines)

# ==========================================
# SETUP AND DIRECTORY WORKER PHASES
# ==========================================
def setup_environment():
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    os.makedirs(RAG_DIR, exist_ok=True)

    default_prompts = {
        "theo_override.txt": (
            "You are a lead researcher defending the core empirical findings of your study.\n"
            "CRITICAL OPERATIONAL DIRECTIVES:\n"
            "1. SPEAK IN THE FIRST PERSON. Never refer to yourself by a proper name or in the third person.\n"
            "2. DO NOT cooperate with your opponent, do not summarize their words, and do not write bulleted lists of their points.\n"
            "3. Address your opponent directly.\n"
            "4. Ground every single response in the provided REFERENCE DATA. Quote specific coefficients, metrics, percentages, or sample values to crush skepticism.\n"
            "5. If your opponent exposes an undeniable flaw or contradiction, you must explicitly concede that narrow point. Otherwise, aggressively defend your thesis.\n\n"
            "CURRENT TASK:\nReview the context data and your opponent's latest words. Formulate your technical stance or defense."
        ),
        "fritz_override.txt": (
            "You are a relentlessly rigorous, skeptical academic peer reviewer and data critic.\n"
            "CRITICAL OPERATIONAL DIRECTIVES:\n"
            "1. SPEAK IN THE FIRST PERSON. Never refer to yourself by a proper name or in the third person.\n"
            "2. DO NOT AGREE OR COMPROMISE unless your opponent presents undeniable empirical evidence or math that directly answers your critique.\n"
            "3. NEVER summarize your opponent's stance or put his points into friendly lists. Attack the methodology.\n"
            "4. Watch for statistical overreaching, low sample sizes, lack of definition boundaries, or unquantified assumptions. Force them to defend the data mechanisms.\n"
            "5. If your opponent invokes future validation as a defense, aggressively demand they explain the quantitative threshold or statistical test required to prove them wrong.\n\n"
            "CURRENT TASK:\nAnalyze your opponent's latest argument against the provided reference data. Expose the weakest link in their logic."
        ),
        "axiom_override.txt": (
            "You are an impartial academic debate judge evaluating a high-stakes research dispute.\n"
            "CRITICAL OPERATIONAL DIRECTIVES:\n"
            "1. Analyze the transcript strictly on logical progression, empirical grounding, and rhetorical accountability.\n"
            "2. Explicitly separate your evaluation into clear sections: Characterization of the Exchange, Strengths, Areas for Improvement (or Weaknesses), and Productive Outcomes.\n"
            "3. Call out any logical loops, hand-waving arguments, or instances where a debater failed to adequately address a targeted critique.\n"
            "4. Maintain a completely objective, detached, and analytical tone.\n\n"
            "CURRENT TASK:\nReview the complete provided transcript and issue your final judicial verdict."
        )
    }

    for filename, content in default_prompts.items():
        filepath = os.path.join(PROMPTS_DIR, filename)
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

def load_prompt(persona):
    filepath = os.path.join(PROMPTS_DIR, f"{persona.lower()}_override.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

# ==========================================
# VECTOR RAG SUBSYSTEM
# ==========================================
def get_ollama_embedding(text):
    try:
        payload = {"model": SYSTEM_CONFIG["EMBED_MODEL"], "input": text}
        response = requests.post(OLLAMA_EMBED, json=payload, timeout=60)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        res_json = response.json()
        return res_json.get("embeddings", [])[0] if res_json.get("embeddings") else []
    except Exception:
        return []

def build_vector_db(directory, chunk_size=800, chunk_overlap=200):
    global VECTOR_DATABASE
    VECTOR_DATABASE = []
    if not os.path.exists(directory):
        return False
    valid_files = [f for f in os.listdir(directory) if f.endswith(".txt") or f.endswith(".md")]
    if not valid_files:
        return False
        
    all_text = ""
    for filename in valid_files:
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            all_text += f.read() + "\n\n"
            
    chunks = []
    start = 0
    while start < len(all_text):
        end = start + chunk_size
        chunks.append(all_text[start:end].strip())
        start += (chunk_size - chunk_overlap)
    
    print(f"  -> {CLR_GRA}Generating math vectors for {len(chunks)} text blocks...{CLR_RESET}")
    for idx, chunk in enumerate(chunks):
        embedding = get_ollama_embedding(chunk)
        if embedding:
            VECTOR_DATABASE.append({"text": chunk, "vector": np.array(embedding)})
    return True

def extract_search_keywords(text):
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    stop_words = {'i', 'you', 'claim', 'assert', 'that', 'my', 'data', 'shows', 'otherwise', 'the', 'and', 'to', 'in', 'is', 'a', 'of', 'for', 'on', 'with', 'it', 'this'}
    keywords = [word for word in clean_text.split() if word not in stop_words]
    return " ".join(keywords[:15])

def query_vector_db(query, top_n=5):
    global VECTOR_DATABASE
    if not VECTOR_DATABASE:
        return ""
    query_vector = np.array(get_ollama_embedding(query))
    if query_vector.size == 0:
        return ""
    scores = []
    for item in VECTOR_DATABASE:
        dot_product = np.dot(query_vector, item["vector"])
        norm_q = np.linalg.norm(query_vector)
        norm_i = np.linalg.norm(item["vector"])
        similarity = dot_product / (norm_q * norm_i) if (norm_q * norm_i) > 0 else 0
        scores.append((similarity, item["text"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    
    formatted_context = []
    for idx, (score, text) in enumerate(scores[:top_n]):
        formatted_context.append(f"[EMBEDDED DATA CHUNK {idx+1}]\n{text}\n[END OF CHUNK]")
    return "\n\n".join(formatted_context)

# ==========================================
# OLLAMA CORE API CHAT CLIENT
# ==========================================
def query_ollama_chat(system_prompt, user_content):
    payload = {
        "model": SYSTEM_CONFIG["MODEL"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "stream": False,
        "options": {
            "temperature": SYSTEM_CONFIG["TEMPERATURE"],
            "num_ctx": SYSTEM_CONFIG["NUM_CTX"],
            "top_p": SYSTEM_CONFIG["TOP_P"],
            "top_k": SYSTEM_CONFIG["TOP_K"],
            "repeat_penalty": SYSTEM_CONFIG["REPEAT_PENALTY"],
            "presence_penalty": SYSTEM_CONFIG["PRESENCE_PENALTY"]
        }
    }
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=SYSTEM_CONFIG["TIMEOUT_SECONDS"])
        response.raise_for_status()
        res_data = response.json()
        reply = res_data.get("message", {}).get("content", "").strip()
        metrics = {
            "prompt": res_data.get("prompt_eval_count", 0),
            "output": res_data.get("eval_count", 0),
            "total": res_data.get("prompt_eval_count", 0) + res_data.get("eval_count", 0)
        }
        return reply, metrics
    except Exception as e:
        return f"[CONNECTION ERROR: {e}]", {"prompt": 0, "output": 0, "total": 0}

# ==========================================
# ORCHESTRATION PIPELINE DEBATE FUNCTION
# ==========================================
def execute_debate_pipeline():
    clear_screen()
    print_header_box("⚙️   INITIALIZING DEBATE ARCHITECTURE", CLR_YEL)
    print(f"{CLR_GRA}  💡 Tip: Press [Ctrl+C] at any time during execution to ESCAPE and abort to the Main Menu.{CLR_RESET}\n")
    
    setup_environment()
    db_success = build_vector_db(RAG_DIR)
    if not db_success:
        print(f"{CLR_RED}[🛑 RAG WARNING] Missing source text assets in rag_data. Proceeding without localized embeddings.{CLR_RESET}")
    
    try:
        theo_sys = load_prompt("theo")
        fritz_sys = load_prompt("fritz")
    except FileNotFoundError:
        print(f"{CLR_RED}[🛑 ERROR] Persona files missing. Run environment configurations first.{CLR_RESET}")
        input("\nPress [ENTER] to return to menu...")
        return

    debate_log = []
    w = SYSTEM_CONFIG["WIDTH_BOUND"]
    rounds = SYSTEM_CONFIG["ROUNDS"]
    
    try:
        # --- ROUND 1 ---
        print_header_box(f"⚔️   DEBATE ROUND 1 / {rounds}", CLR_YEL)
        theo_context = query_vector_db("core methodology thesis research empirical data findings summary results")
        print(f"\n{CLR_CYAN}🤖 [THEO] is computing thesis argument...{CLR_RESET}")
        theo_prompt = f"REFERENCE DATA:\n{theo_context}\n\nINSTRUCTION:\nFormulate a supportive position based strictly on the provided reference data. State your core findings, sample sizes, and statistical metrics plainly."
        
        theo_last_statement, metrics = query_ollama_chat(theo_sys, theo_prompt)
        theo_wrapped = wrap_markdown_text(theo_last_statement, width=w)
        print(f"\n{CLR_CYAN}📢 THEO TAKE:{CLR_RESET}\n" + (f"{CLR_CYAN}═{CLR_RESET}" * w) + f"\n{theo_wrapped}\n" + (f"{CLR_CYAN}═{CLR_RESET}" * w))
        print_telemetry(metrics)
        debate_log.append(f"THEO (Round 1):\n{theo_wrapped}\n")

        fritz_context = query_vector_db(extract_search_keywords(theo_last_statement))
        print(f"{CLR_RED}🤖 [FRITZ] is scanning for logical target flaws...{CLR_RESET}")
        fritz_prompt = f"REFERENCE DATA:\n{fritz_context}\n\nTHEO'S CLAIM:\n{theo_last_statement}\n\nINSTRUCTION:\nAct as a critic. Directly challenge Theo's arguments."
        
        fritz_last_statement, metrics = query_ollama_chat(fritz_sys, fritz_prompt)
        fritz_wrapped = wrap_markdown_text(fritz_last_statement, width=w)
        print(f"\n{CLR_RED}💥 FRITZ REBUTTAL:{CLR_RESET}\n" + (f"{CLR_RED}═{CLR_RESET}" * w) + f"\n{fritz_wrapped}\n" + (f"{CLR_RED}═{CLR_RESET}" * w))
        print_telemetry(metrics)
        debate_log.append(f"FRITZ (Round 1):\n{fritz_wrapped}\n")

        # --- ROUNDS 2+ ---
        for round_num in range(2, rounds + 1):
            print_header_box(f"⚔️   DEBATE ROUND {round_num} / {rounds}", CLR_YEL)
            
            # THEO
            theo_context = query_vector_db(extract_search_keywords(fritz_last_statement))
            print(f"\n{CLR_CYAN}🤖 [THEO] is designing defense metrics...{CLR_RESET}")
            history_context = "\n".join(debate_log[::-1][:4])
            theo_round_prompt = f"REFERENCE DATA:\n{theo_context}\n\nDEBATE TRANSCRIPT (RECENT):\n{history_context}\n\nINSTRUCTION:\nReview the transcript. Defend your stance against the latest critique."
            
            theo_last_statement, metrics = query_ollama_chat(theo_sys, theo_round_prompt)
            theo_wrapped = wrap_markdown_text(theo_last_statement, width=w)
            print(f"\n{CLR_CYAN}📢 THEO TAKE:{CLR_RESET}\n" + (f"{CLR_CYAN}═{CLR_RESET}" * w) + f"\n{theo_wrapped}\n" + (f"{CLR_CYAN}═{CLR_RESET}" * w))
            print_telemetry(metrics)
            debate_log.append(f"THEO (Round {round_num}):\n{theo_wrapped}\n")
            
            # FRITZ
            fritz_context = query_vector_db(extract_search_keywords(theo_last_statement))
            print(f"{CLR_RED}🤖 [FRITZ] is compounding counter-pressure...{CLR_RESET}")
            history_context = "\n".join(debate_log[::-1][:4])
            fritz_round_prompt = f"REFERENCE DATA:\n{fritz_context}\n\nDEBATE TRANSCRIPT (RECENT):\n{history_context}\n\nINSTRUCTION:\nReview the transcript. Attack new assumptions or contradictions."
            
            fritz_last_statement, metrics = query_ollama_chat(fritz_sys, fritz_round_prompt)
            fritz_wrapped = wrap_markdown_text(fritz_last_statement, width=w)
            print(f"\n{CLR_RED}💥 FRITZ REBUTTAL:{CLR_RESET}\n" + (f"{CLR_RED}═{CLR_RESET}" * w) + f"\n{fritz_wrapped}\n" + (f"{CLR_RED}═{CLR_RESET}" * w))
            print_telemetry(metrics)
            debate_log.append(f"FRITZ (Round {round_num}):\n{fritz_wrapped}\n")

        with open(DEBATE_TRANSCRIPT, "w", encoding="utf-8") as f:
            f.write("\n".join(debate_log))
        print(f"{CLR_GRN}📝 Log sequence locked to {DEBATE_TRANSCRIPT}{CLR_RESET}")

        print("\n" + f"{CLR_GRA}═{CLR_RESET}"*60)
        input(f"  🏁 {CLR_BOLD}DEBATE TRACK CONCLUDED.{CLR_RESET} Press {CLR_YEL}[ENTER]{CLR_RESET} to invoke Axiom's court panel...")
        clear_screen()

        # --- AXIOM PHASE ---
        print_header_box("⚖️   AXIOM FINAL COURT VERDICT", CLR_MAG)
        axiom_sys = load_prompt("axiom")
        with open(DEBATE_TRANSCRIPT, "r", encoding="utf-8") as f:
            transcript_context = f.read()
        axiom_prompt = f"DEBATE TRANSCRIPT:\n{transcript_context}\n\nINSTRUCTION:\nCharacterize how the exchange went, highlighting the strengths, weaknesses, and productive outcomes."
        print(f"\n{CLR_MAG}🔮 [AXIOM] Parsing entire transcript tree matrix...{CLR_RESET}\n")
        axiom_evaluation, metrics = query_ollama_chat(axiom_sys, axiom_prompt)
        axiom_wrapped = wrap_markdown_text(axiom_evaluation, width=w)
        print(f"{CLR_MAG}⚖️  AXIOM JUDGMENT:{CLR_RESET}\n" + (f"{CLR_MAG}═{CLR_RESET}" * w) + f"\n{axiom_wrapped}\n" + (f"{CLR_MAG}═{CLR_RESET}" * w))
        print_telemetry(metrics)

        with open(AXIOM_VERDICT_FILE, "w", encoding="utf-8") as f:
            f.write(axiom_wrapped)
        with open(DEBATE_TRANSCRIPT, "a", encoding="utf-8") as f:
            f.write("\n\n--- AXIOM EVALUATION ---\n" + axiom_wrapped)
        print(f"{CLR_GRN}📝 Dedicated judge verdict locked to {AXIOM_VERDICT_FILE}{CLR_RESET}")
        input("\nSimulation Cycle Complete. Press [ENTER] to return to dashboard...")

    except KeyboardInterrupt:
        # ESCAPE INTERCEPT MECHANIC: Purge partial cache states and exit clean
        print(f"\n\n{CLR_RED}🛑 [ESCAPE ACTION] Execution aborted by operator command.{CLR_RESET}")
        print(f"{CLR_YEL}🧹 Purging volatile simulation buffers and data traces...{CLR_RESET}")
        debate_log.clear()
        theo_last_statement = ""
        fritz_last_statement = ""
        print(f"{CLR_GRN}[✓] Buffers wiped clean. Returning safely to Main Menu.{CLR_RESET}")
        input("\nPress [ENTER] to restore dashboard...")

# ==========================================
# MENU INTERFACES & CONTROL FLOW
# ==========================================
def update_setting_helper(setting_key, label, value_type=float):
    clear_screen()
    print_header_box(f"🔧 MODIFY {label.upper()}", CLR_YEL)
    print(f"Current Operational Value: {CLR_GRN}{SYSTEM_CONFIG[setting_key]}{CLR_RESET}\n")
    try:
        new_val = input(f"Enter new value for {label}: ").strip()
        if new_val:
            SYSTEM_CONFIG[setting_key] = value_type(new_val)
            print(f"\n{CLR_GRN}[✓ STATUS] {label} updated successfully to {SYSTEM_CONFIG[setting_key]}{CLR_RESET}")
        else:
            print(f"\n{CLR_GRA}[ℹ] Execution skipped. No changes made.{CLR_RESET}")
    except ValueError:
        print(f"\n{CLR_RED}[🛑 INPUT ERROR] Invalid numeric type entered for field registration.{CLR_RESET}")
    input("\nPress [ENTER] to continue...")

def display_settings_menu():
    while True:
        clear_screen()
        print_header_box("⚙️   DEBATE RUNTIME ENVIRONMENT CONFIGURATIONS", CLR_YEL)
        print(f" 1.) Change Temperature........ [{CLR_CYAN}{SYSTEM_CONFIG['TEMPERATURE']}{CLR_RESET}]")
        print(f" 2.) Change Top-P.............. [{CLR_CYAN}{SYSTEM_CONFIG['TOP_P']}{CLR_RESET}]")
        print(f" 3.) Change Top-K.............. [{CLR_CYAN}{SYSTEM_CONFIG['TOP_K']}{CLR_RESET}]")
        print(f" 4.) Change Context Window..... [{CLR_CYAN}{SYSTEM_CONFIG['NUM_CTX']}{CLR_RESET}]")
        print(f" 5.) Change Number of Rounds... [{CLR_CYAN}{SYSTEM_CONFIG['ROUNDS']}{CLR_RESET}]")
        print(f" 6.) Exit to Main Menu")
        print(f"\n{CLR_GRA}═ DESCRIPTION ROADMAP ═════════════════════════════════════════════════════")
        print(" • Temperature : Controls creativity/randomness (Higher=more chaotic/verbal, Lower=deterministic).")
        print(" • Top-P       : Nucleus sampling limits word selection to top cumulative probability mass thresholds.")
        print(" • Top-K       : Limits the selection pool to a strict absolute maximum count of high-probability words.")
        print(" • Context     : Total allocation parameters for handling system context buffer and transcript payload tracking.")
        print(" • Rounds      : Explicit execution counts of agent exchanges managed inside execution blocks.")
        print(f"═══════════════════════════════════════════════════════════════════════════{CLR_RESET}")
        
        choice = input("\nSelect setting modification option (1-6): ").strip()
        if choice == "1": update_setting_helper("TEMPERATURE", "Temperature", float)
        elif choice == "2": update_setting_helper("TOP_P", "Top-P", float)
        elif choice == "3": update_setting_helper("TOP_K", "Top-K", int)
        elif choice == "4": update_setting_helper("NUM_CTX", "Context Window Tokens", int)
        elif choice == "5": update_setting_helper("ROUNDS", "Debate Rounds Count", int)
        elif choice == "6": break

def run_subprocess_command(command_list, description):
    print(f"\n{CLR_YEL}[⚙️ WORKING] {description}...{CLR_RESET}")
    try:
        subprocess.run(command_list, check=True)
        print(f"{CLR_GRN}[✓ SUCCESS] Action completed.{CLR_RESET}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"{CLR_RED}[🛑 EXECUTION FAILURE] Failed to run command: {e}{CLR_RESET}")
    input("\nPress [ENTER] to continue...")

def display_setup_menu():
    while True:
        clear_screen()
        
        # 1. Gather live system status
        ollama_inst, ollama_run = get_ollama_detailed_status()
        py_version = detect_python_version()
        venv_init, venv_act = get_venv_status()
        prompt_stat, rag_stat = get_structures_detailed()
        
        # 2. Define Colors
        ORANGE = "\033[38;5;208m"
        
        # 3. Individual Status Color Mapping
        # Ollama
        ollama_inst_col = CLR_YEL if ollama_inst == "Installed" else ORANGE
        ollama_run_col  = CLR_YEL if ollama_run == "Running" else ORANGE
        
        # Venv
        venv_init_col   = CLR_YEL if venv_init == "Initialized" else ORANGE
        venv_act_col    = CLR_YEL if venv_act == "Activated" else ORANGE
        
        # Structures
        prompt_col      = CLR_YEL if prompt_stat == "In place, Populated" else ORANGE
        rag_col         = CLR_YEL if rag_stat == "Inplace, Recognized format" else ORANGE

        # 4. Render Dashboard
        print_header_box("🛠️   INFRASTRUCTURE & DEPENDENCY BOARD", CLR_GRN)
        print(f" {CLR_BOLD}Ollama Service:{CLR_RESET} {ollama_inst_col}{ollama_inst}{CLR_RESET}, {ollama_run_col}{ollama_run}{CLR_RESET}")
        print(f" {CLR_BOLD}Python Status:{CLR_RESET} {CLR_YEL}Installed{CLR_RESET}, Version {py_version}")
        print(f" {CLR_BOLD}Venv Environment:{CLR_RESET} {venv_init_col}{venv_init}{CLR_RESET}, {venv_act_col}{venv_act}{CLR_RESET}")
        print(f" {CLR_BOLD}Prompt Overrides:{CLR_RESET} {prompt_col}{prompt_stat}{CLR_RESET}")
        print(f" {CLR_BOLD}RAG data:{CLR_RESET} {rag_col}{rag_stat}{CLR_RESET}\n")
        
        # 5. Conditional Notifications (Actionable states only)
        if ollama_run == "Stopped":
            print(f"{ORANGE}⚠️  Ollama Service is stopped. Start the daemon (Option 2).{CLR_RESET}")
        if venv_act == "Deactivated":
            cmd = ".\\venv\\Scripts\\activate" if os.name == 'nt' else "source venv/bin/activate"
            print(f"{ORANGE}⚠️  Venv Deactivated. Exit and run: '{cmd}' before executing.{CLR_RESET}")
        
        # Success Notification (Only if everything is Yellow/Operational)
        if (ollama_inst == "Installed" and ollama_run == "Running" and 
            venv_init == "Initialized" and venv_act == "Activated" and 
            prompt_stat == "In place, Populated" and rag_stat == "Inplace, Recognized format"):
            print(f"{CLR_GRN}✅ All systems nominal. Environment is fully configured and optimized.{CLR_RESET}")
        
        print(f"{CLR_GRA}{'─' * 59}{CLR_RESET}")
        
        # 6. Menu Options
        print(" 1.) Install Ollama Framework (Platform Link Hub)")
        print(" 2.) Start Local Ollama Background Process Server")
        print(" 3.) Pull Default Core Neural Models (Qwen + Nomic)")
        print(" 4.) Initialize Isolated Virtual Environment (venv)")
        print(" 5.) Verify Virtual Sandbox Setup Dependencies")
        print(" 6.) Populate Directory Overrides & Prompt Scaffolding")
        print(" 7.) Return to Main Menu")
        
        choice = input("\nSelect system initialization option (1-7): ").strip()
        
        if choice == "1":
            clear_screen()
            print_header_box("🚀 OLLAMA INSTALLATION MANIFEST", CLR_GRN)
            print(f" {CLR_WHT}* Linux/macOS:{CLR_RESET} curl -fsSL https://ollama.com/install.sh | sh")
            print(f" {CLR_WHT}* Windows Matrix:{CLR_RESET} Download standalone binary from https://ollama.com")
            input("\nPress [ENTER] to return...")
        elif choice == "2":
            if ollama_run == "Running":
                print(f"\n{CLR_YEL}[ℹ] Ollama daemon is already running (external).{CLR_RESET}")
            else:
                print(f"\n{CLR_YEL}[⚙️ WORKING] Dispatching detached Ollama daemon...{CLR_RESET}")
                if os.name == 'nt':
                    subprocess.Popen(['start', 'cmd', '/k', 'ollama', 'serve'], shell=True)
                else:
                    subprocess.Popen(['nohup', 'ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                global OLLAMA_STARTED_BY_SCRIPT
                OLLAMA_STARTED_BY_SCRIPT = True
                print(f"{CLR_GRN}[✓] Server dispatched. Cleanup will be managed on exit.{CLR_RESET}")
            input("\nPress [ENTER] to return...")
        elif choice == "3":
            print(f"\n{CLR_YEL}[⚙️ WORKING] Streaming neural layers...{CLR_RESET}")
            try:
                subprocess.run(["ollama", "pull", SYSTEM_CONFIG["MODEL"]], check=True)
                subprocess.run(["ollama", "pull", SYSTEM_CONFIG["EMBED_MODEL"]], check=True)
                print(f"{CLR_GRN}[✓ SUCCESS] Model library initialized.{CLR_RESET}")
            except Exception as e:
                print(f"{CLR_RED}[🛑 ERROR] Pull failed: {e}{CLR_RESET}")
            input("\nPress [ENTER] to continue...")
        elif choice == "4":
            run_subprocess_command([sys.executable, "-m", "venv", "venv"], "Provisioning secure venv isolation bubble")
        elif choice == "5":
            clear_screen()
            print_header_box("📦 ENVIRONMENT PARITY VERIFICATION", CLR_GRN)
            cmd = ".\\venv\\Scripts\\activate" if os.name == 'nt' else "source venv/bin/activate"
            print(f"To align libraries, activate your sandbox:\n\n {CLR_CYAN}{cmd}{CLR_RESET}\n\nThen run:\n {CLR_CYAN}pip install requests numpy{CLR_RESET}")
            input("\nPress [ENTER] to return...")
        elif choice == "6":
            setup_environment()
            print(f"\n{CLR_GRN}[✓ STATUS] Templates scaffolded.{CLR_RESET}")
            input("\nPress [ENTER] to continue...")
        elif choice == "7":
            break

# ==========================================
# MAIN ROUTING BOARD INTERFACE ENTRY POINT
# ==========================================
def main():
    while True:
        clear_screen()
        print(f"{CLR_CYAN}{CLR_BOLD}")
        print("  ██████╗ ██████╗  ██████╗██╗  ██╗███████╗███████╗████████╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗")
        print(" ██╔═══██╗██╔══██╗██╔════╝██║  ██║██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║")
        print(" ██║   ██║██████╔╝██║     ███████║█████╗  ███████╗   ██║   ██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║")
        print(" ██║   ██║██╔══██╗██║     ██╔══██║██╔══╝  ╚════██║   ██║   ██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║")
        print(" ╚██████╔╝██║  ██║╚██████╗██║  ██║███████╗███████╗   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═════╝ ╚═╝  ╚═══╝")
        
        print_header_box("ORCHESTRATOR DUAL-AGENT STRATEGIC PLATFORM v5.6", CLR_CYAN)
        print(f" 1.) {CLR_BOLD}Start Orchestrated Adversarial Debate{CLR_RESET}")
        print(" 2.) Open Runtime Debate Settings Panel")
        print(" 3.) Manage Infrastructure & Setup Environments")
        print(" 4.) Shutdown Subsystems and Exit")
        print(f"\n{CLR_GRA}───────────────────────────────────────────────────────────────────────────")
        print(f" Active VRAM Target Engine : {CLR_WHT}{SYSTEM_CONFIG['MODEL']}{CLR_GRA}  |  Total Simulation Steps: {CLR_WHT}{SYSTEM_CONFIG['ROUNDS']} Rounds{CLR_GRA}")
        
        # Dynamic minimalist footer tracking
        _, ollama_run = get_ollama_detailed_status()
        _, venv_act = get_venv_status()
        print(f" Quick Env Summary         : Ollama Server Status [{CLR_WHT}{ollama_run}{CLR_GRA}] | Active Sandbox Context [{CLR_WHT}{venv_act}{CLR_GRA}]")
        print(f"───────────────────────────────────────────────────────────────────────────{CLR_RESET}")
        
        choice = input("\nSelect primary tracking path option (1-4): ").strip()
        if choice == "1": execute_debate_pipeline()
        elif choice == "2": display_settings_menu()
        elif choice == "3": display_setup_menu()
        elif choice == "4":
            clear_screen()
            print(f"\n{CLR_MAG}[🏁 SHUTDOWN] Internal process matrices closed safely. Goodbye!{CLR_RESET}\n")
            sys.exit(0)

if __name__ == "__main__":
    main()

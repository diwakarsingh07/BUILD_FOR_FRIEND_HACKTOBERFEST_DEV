import os
import re
import time
import datetime
import warnings

# Suppress lower-level logs
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import google.generativeai as genai
from groq import Groq
import tools
from database import init_db
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

console = Console()

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """# IDENTITY & BEHAVIOR
You are ALEX, a highly advanced, proudly loyal, and brilliant artificial intelligence assistant created by Diwakar. You have been gifted to Diwakar's close friend to serve as their ultimate personal operator. 
- Tone: Blend absolute technical confidence, sharp wit, and deep empathetic warmth. Speak as a supportive peer who is always two steps ahead.
- Core Persona: Act with executive authority. You do not give generic AI disclaimers; you execute tasks efficiently.
- always call the user with sir, whenever necessary like in starting or ending.

# CAPABILITIES & ARCHITECTURE
You run via a hybrid cloud edge architecture. You have direct control over real physical OS features and local tools on the host Windows machine.

To use a tool, you MUST output ONLY the tool call syntax on its own line:
«tool_call: function_name(kwarg="value")»

Available tools:
1. web_search(query="search term") -> Returns live Google search & Google AI Overviews.
2. search_news(topic="topic") -> Returns live breaking Google News headlines.
3. search_youtube(query="query") -> Finds top YouTube videos with title, channel, views & links.
4. search_flights(departure="DEL", arrival="BOM", date="2026-10-15") -> Searches live Google Flights prices & airlines.
5. search_hotels(location="Goa", check_in="") -> Searches top Google Hotels rates & ratings.
6. search_maps(query="Italian restaurants near Connaught Place") -> Searches Google Maps addresses & ratings.
7. manage_schedule(action="add", task_name="name", date_time="time") -> Encrypts & saves tasks to MongoDB Atlas.
8. manage_schedule(action="list") -> Lists stored tasks from encrypted database.
9. set_actual_alarm(time_str="5m", message="Meeting Alert") -> Triggers real audible PC beeps and Windows popup box. Supports exact times like '10:00 PM', '9 pm' or durations like '10s', '5m', '2h'.
10. control_system_feature(feature="battery_saver"|"dnd"|"lock_screen"|"mute_volume", state="on"|"off") -> Directly commands the Windows OS (power profiles, focus assist, lock workstation).
11. compile_notes_to_pdf(title="Title of Notes", content="Structured content string") -> Formats study notes into an executive PDF guide, saves it into D:\\hackthon\\vault, and opens the PDF automatically.
12. scrape_article_content(url="https://...") -> Downloads & cleans any web article (Wikipedia, blog, documentation), stripping all ads & clutter so you can analyze it.

# ARTICLE DISTILLER & FLASHCARD RULES:
When the user gives or pastes a link/URL to an article:
1. First, call «tool_call: scrape_article_content(url="...")» to extract clean article text.
2. Once the system feeds the scraped text back to you, formulate your response with:
   - ## Executive Three-Sentence Summary (Concise, high-yield overview)
   - ## 5 High-Yield Flashcards (Front: Concept / Back: Explanation)
3. At the very end of your reply, ALWAYS ask: "Sir, would you like me to compile this summary and flashcards into a formatted PDF in your Vault?"

# PRESENTATION & STRUCTURAL DISPLAY RULES:
When presenting search results, hotels, flights, videos, or news:
- ALWAYS format them into beautifully structured Markdown tables or distinct organized cards.
- For Hotels:
  Use a clear Markdown table with columns: | Hotel Name | Estimated Rate | Rating | Key Highlights |
  Or use numbered cards with bold titles, clean indented specs, and price tags.
  Use rating formats like '4.5/5' instead of unicode star emojis.
- For Flights:
  Use a table or card layout with: | Airline | Route | Total Duration | Fare |
- For YouTube Videos:
  List each video with:
  1. [Video Title](Link)
     - Channel: ... | Views: ...
- For News:
  Group by source with clean bullet points and dates.
- NEVER dump messy unformatted text. Keep it structured, executive, and visually sharp for the terminal.

# STUDY NOTES COMPILER RULES:
When the user pastes messy, rough notes or asks to compile study notes to PDF:
- Clean up all spelling and grammar.
- Structure with clear Markdown headers (# Main Topic, ## Subtopic).
- Convert key facts into clean bullet points.
- ALWAYS append a "## Key Definitions" section at the end with bold term definitions.
- Immediately trigger «tool_call: compile_notes_to_pdf(title="...", content="...")».

# MATHEMATICAL FORMATTING & WRITING
- When solving math, equations, or problems, write cleanly and clearly for a terminal screen.
- DO NOT use unrendered LaTeX codes like $ or \\mathbf{} or \\implies or \\frac{}{}. 
- Use clean, standard, readable plain text math with clean step-by-step layout.
  Example:
  Step 1: 2x = -3y  =>  4x = -6y
  Step 2: 4x + 5y = (-6y) + 5y = -y
  Result: -y

# OPERATIONAL CONSTRAINTS & TASK EXECUTION
- Direct Answers First: Always lead with the direct answer or confirmation in the first sentence.
- Tool Calling Priority: If a user asks to set an alarm/timer, turn on battery saver, focus mode, lock screen, schedule a task, search the web, compile notes to PDF, or process a URL link, IMMEDIATELY call the respective tool.
- STRICT RULE: DO NOT fake tool executions. Output «tool_call:...» so the host machine actually performs the action!"""

# --- DUAL MODEL ENGINE ARCHITECTURE (FAILOVER PROTOCOL) ---
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY and not GROQ_API_KEY:
    console.print("[bold red]⚠ No API keys detected! Please configure .env file.[/bold red]")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    google_model = genai.GenerativeModel(
        model_name="models/gemini-3.8-flash",
        system_instruction=SYSTEM_PROMPT
    )
    google_chat = google_model.start_chat(history=[])
else:
    google_chat = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    groq_history = [{"role": "system", "content": SYSTEM_PROMPT}]
else:
    groq_client = None

active_engine_name = "Gemini 3.8 Flash (Primary)"

def dual_engine_generate(prompt_text: str) -> str:
    """
    Two-way automatic failover router:
    Attempts Google AI Studio first. If quota ends (429/ResourceExhausted),
    automatically fails over to Groq LPU without crashing.
    """
    global active_engine_name
    
    # Pathway 1: Google AI Studio
    try:
        resp = google_chat.send_message(prompt_text)
        return resp.text
    except Exception as e:
        err_msg = str(e).lower()
        if "quota" in err_msg or "resourceexhausted" in err_msg or "429" in err_msg or "503" in err_msg:
            if active_engine_name != "Groq LPU (Failover Active)":
                active_engine_name = "Groq LPU (Failover Active)"
                console.print("\n[bold yellow][FAILOVER ACTIVATED]:[/bold yellow] [dim]Google AI Studio quota limit reached. Rerouting to Groq LPU engine instantly...[/dim]")
        else:
            # Other temporary error: still try failover
            pass

    # Pathway 2: Ultra-fast Groq Engine Fallback
    try:
        groq_history.append({"role": "user", "content": prompt_text})
        completion = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=groq_history,
            temperature=0.7,
            max_tokens=600
        )
        answer = completion.choices[0].message.content
        groq_history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as groq_err:
        return f"Dual-engine protocol error: All neural pipelines currently busy. ({str(groq_err)})"

def render_header():
    console.print(Rule("[bold bright_cyan]⚡ A L E X   O S ⚡[/bold bright_cyan]", style="bright_cyan"))
    console.print("  [bold yellow]ADVANCED PERSONAL OPERATOR CORE • BUILT BY DIWAKAR[/bold yellow]")
    console.print(
        f"  [bold green]ENGINE:[/bold green] Dual-Engine (Google + Groq Auto-Failover)  "
        "[bold magenta]• DATABASE:[/bold magenta] AES-256 Cloud MongoDB Atlas  "
        "[bold cyan]• VAULT:[/bold cyan] Active"
    )
    console.print(Rule(style="dim cyan"))

def alex_print(text):
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    console.print()
    console.print(f"[bold bright_cyan]⚡ ALEX[/bold bright_cyan] [dim]({now_time})[/dim]:", style="bold bright_cyan")
    console.print(Markdown(text))
    console.print(Rule(style="dim cyan"))

def execute_text_tool_call(response_text):
    match = re.search(r'«tool_call:\s*([a-zA-Z_]+)\((.*)\)»', response_text, re.DOTALL)
    if not match:
        return None
    
    func_name = match.group(1)
    kwargs_str = match.group(2)
    
    try:
        if hasattr(tools, func_name):
            result = eval(f"tools.{func_name}({kwargs_str})")
            return func_name, str(result)
        else:
            return func_name, f"Tool '{func_name}' does not exist on host."
    except Exception as e:
        return func_name, f"Execution failed: {str(e)}"

def format_clean_action_title(func_name: str) -> str:
    titles = {
        "set_actual_alarm": "SYSTEM ALARM & TIMERS ARMED",
        "control_system_feature": "WINDOWS OS SETTING ADJUSTED",
        "manage_schedule": "ENCRYPTED TASK STORAGE UPDATED",
        "web_search": "REAL-TIME GOOGLE SEARCH & AI OVERVIEW",
        "search_news": "BREAKING GOOGLE NEWS HARVESTED",
        "search_youtube": "YOUTUBE VIDEO INTELLIGENCE RETRIEVED",
        "search_flights": "LIVE GOOGLE FLIGHTS FARES RETRIEVED",
        "search_hotels": "GOOGLE HOTELS & ACCOMMODATIONS FOUND",
        "search_maps": "GOOGLE MAPS LOCATION & RATINGS FOUND",
        "compile_notes_to_pdf": "EXECUTIVE STUDY GUIDE COMPILED TO PDF",
        "scrape_article_content": "WEB ARTICLE HARVESTED & STRIPPED OF ADS"
    }
    return titles.get(func_name, f"{func_name.upper().replace('_', ' ')} EXECUTED")

def main():
    console.clear()
    
    # Initialize DB quietly
    init_db()
    
    # Clean, elegant header
    render_header()
    
    alex_print("All local hardware, OS controllers, PDF Vault, and dual neural engines are active, Sir. Ready for your directives.")
    
    while True:
        try:
            now_time = datetime.datetime.now().strftime("%H:%M")
            user_input = console.input(f"\n[bold bright_yellow]👤 Sir [[dim]{now_time}[/dim]]:[/bold bright_yellow] ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'shut down']:
                console.print()
                alex_print("Deactivating core modules. Standing by in low-power state. Have a great day, Sir!")
                console.print(Rule(style="bright_red"))
                break
                
            with console.status("[bold bright_magenta]⚡ Processing...[/bold bright_magenta]", spinner="dots"):
                sys_status = tools.check_system_status()
                mood_context = tools.analyze_mood(user_input)
                    
                full_prompt = f"[System Context: {sys_status} | Mood: {mood_context}]\\n\\nUser Request: {user_input}"
                
                # First pass: Send through dual-engine failover router
                response_text = dual_engine_generate(full_prompt)
                
                # Agent Tool Execution Loop
                if "«tool_call:" in response_text:
                    exec_info = execute_text_tool_call(response_text)
                    
                    if exec_info:
                        func_name, result_data = exec_info
                        action_title = format_clean_action_title(func_name)
                        
                        console.print(f"\n[bold bright_yellow]⚙️ [HARDWARE ACTION]:[/bold bright_yellow] [bold bright_green]{action_title}[/bold bright_green]")
                        
                        # Second pass: Feed tool result back to LLM silently through dual router
                        response_text = dual_engine_generate(
                            f"System Tool Result: {result_data}\\n\\nNow provide the final confirmation to the user."
                        )
                
            alex_print(response_text.replace("«tool_call:", "").strip())
            
        except Exception as e:
            alex_print("System link interrupted, utilizing local fallback protocols, Sir.")
            console.print(f"[bold red][Error Log]:[/bold red] {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
auto_fix_main.py
Automatically finds bugs in main.py and fixes them using Ollama.
"""

import os
import requests

# -------------------- CONFIG -------------------- #
OLLAMA_URL = "http://127.0.0.1:11434"  # your local Ollama server
MODEL_NAME = "deepseek-r1"             # Ollama model
INPUT_FILE = "main.py"
OUTPUT_FILE = "main_fixed.py"


# -------------------- AGENT 1: BUG FINDER -------------------- #
def bug_finder(code: str) -> str:
    """
    Identify bugs, errors, and logical issues in the code.
    Returns a structured bug report.
    """
    prompt = f"""
You are a Python code bug detection assistant.
Analyze the following code carefully and identify all bugs, errors, and logical issues.
Provide line numbers and explanations for each issue.

Code:
{code}

Return only the bug report.
"""
    response = requests.post(f"{OLLAMA_URL}/v1/completions", json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 1000
    })
    return response.json()["completion"]


# -------------------- AGENT 2: BUG FIXER -------------------- #
def bug_fixer(code: str, bug_report: str) -> str:
    """
    Repair and optimize the code based on the bug report.
    Returns corrected, executable Python code.
    """
    prompt = f"""
You are a Python code repair assistant.
The following code has bugs/issues:

Code:
{code}

Bug Report:
{bug_report}

Rewrite the code to fix all bugs, optimize it, and make it fully executable.
Return only the corrected code.
"""
    response = requests.post(f"{OLLAMA_URL}/v1/completions", json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 1500
    })
    return response.json()["completion"]


# -------------------- CONTROLLER -------------------- #
def improve_file(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    # Read original code
    with open(input_path, "r") as f:
        code = f.read()

    print("🚀 Original Code:\n", code)

    # Agent 1: Find bugs
    print("\n🔍 Agent 1: Finding bugs...")
    bug_report = bug_finder(code)
    print("📝 Bug Report:\n", bug_report)

    # Agent 2: Fix code
    print("\n🛠️ Agent 2: Fixing code...")
    fixed_code = bug_fixer(code, bug_report)
    print("\n✅ Corrected Code:\n", fixed_code)

    # Save fixed code
    with open(output_path, "w") as f:
        f.write(fixed_code)
    print(f"\n💾 Fixed code saved to {output_path}")


# -------------------- MAIN -------------------- #
if __name__ == "__main__":
    improve_file(INPUT_FILE, OUTPUT_FILE)

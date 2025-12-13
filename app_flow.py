import os
import re
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langdetect import detect, LangDetectException

# --- CONFIGURATION ---
BERT_API_URL = "https://abhinavdread-prompt-analyzer.hf.space/score"
HF_TOKEN = os.environ.get("HF_TOKEN") 

# WEIGHTS: We trust the LLM heavily (70%)
WEIGHT_BERT = 30
WEIGHT_LLM = 70
TOTAL_WEIGHT = WEIGHT_BERT + WEIGHT_LLM

# --- 1. HEURISTIC FILTERS (Basic Sanitization) ---
def is_valid_prompt(text: str) -> dict:
    # A. Length Check
    if len(text.split()) < 3:
        return {"valid": False, "reason": "Prompt is too short."}
    
    # B. Repetition Check (Spam protection)
    words = text.lower().split()
    if len(words) > 0:
        most_common = max(set(words), key=words.count)
        if words.count(most_common) / len(words) > 0.5:
             return {"valid": False, "reason": "Detected repetitive spam."}

    # C. Language Check
    try:
        lang = detect(text)
        if lang != 'en':
            return {"valid": False, "reason": f"Detected non-English text ({lang})."}
    except LangDetectException:
        pass 

    return {"valid": True, "reason": "Pass"}

# --- 2. CLOUD BERT TOOL ---
@tool
def get_quality_score(text: str) -> float:
    """Sends the prompt to the cloud-hosted BERT model for scoring."""
    try:
        response = requests.post(BERT_API_URL, json={"prompt": text})
        if response.status_code == 200:
            return response.json().get("score", 0.0) 
        return 0.0
    except:
        return 0.0

# --- 3. CLOUD LLM SETUP ---
llm_engine = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1, # Low temp for consistent scoring
    huggingfacehub_api_token=HF_TOKEN
)
llm = ChatHuggingFace(llm=llm_engine)

# --- 4. THE PROBABILISTIC SYSTEM PROMPT ---
# We treat Intent as a score. "Act as..." gets 100. "Shakespeare" gets 0.
SYSTEM_PROMPT = """
You are a Master Prompt Engineer. Rate this prompt on 7 metrics (0-100).

INPUT PROMPT: "{user_prompt}"

SCORING GUIDELINES:
1. Intent Strength: 
   - 100 = Explicit instruction ("Act as...", "Write a...", "Fix code...", "Imagine..."). 
   - 0 = Pure content (Poem, Story, Statement) with NO request.

2. Clarity: Is the goal unambiguous?
3. Specificity: detailed vs vague?
4. Context: Is background provided?
5. Constraints: Are limits defined?
6. Complexity: Does it require reasoning?
7. Role Definition: Does it assign a persona? (e.g. "Act as an expert").

OUTPUT FORMAT (Strictly Numbers Only):
Intent_Strength: [Score]
Clarity: [Score]
Specificity: [Score]
Context: [Score]
Constraints: [Score]
Complexity: [Score]
Role_Definition: [Score]
"""

def analyze_prompt_flow(user_prompt: str):
    print(f"\n🚀 Received: '{user_prompt[:30]}...'")

    # STEP 0: SANITY CHECK
    check = is_valid_prompt(user_prompt)
    if not check["valid"]:
        return {
            "bert_score": 0, "llm_score": 0, "final_score": 0,
            "status": "REJECTED", "msg": check['reason']
        }

    # STEP A: BERT SCORE
    bert_score = get_quality_score.invoke(user_prompt)
    
    # STEP B: LLM SCORING
    prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    try:
        chain = prompt_template | llm
        response = chain.invoke({"user_prompt": user_prompt})
        content = response.content

        # Extract Scores
        def extract_score(key, text):
            match = re.search(fr"{key}:\s*(\d+(\.\d+)?)", text)
            return float(match.group(1)) if match else 0.0

        intent_score = extract_score("Intent_Strength", content)
        role_score = extract_score("Role_Definition", content)
        
        # Calculate Average of Quality Metrics
        quality_metrics = ["Clarity", "Specificity", "Context", "Constraints", "Complexity", "Role_Definition"]
        total_quality = sum(extract_score(m, content) for m in quality_metrics)
        llm_quality_average = total_quality / len(quality_metrics)

        # --- THE LOGIC TRAPDOOR (FIXED) ---
        # 1. Reject if Intent is non-existent (Shakespeare case)
        if intent_score < 20:
             return {
                "bert_score": bert_score,
                "llm_score": llm_quality_average,
                "final_score": llm_quality_average, 
                "status": "REJECTED",
                "msg": "Input appears to be content rather than an instruction."
            }

        # 2. Boost Score if Role Definition is present (Etymologist case)
        # If the user used "Act as...", we boost the final LLM score
        if role_score > 80:
            llm_quality_average = min(100, llm_quality_average * 1.1)

        # Calculate Final Hybrid Score
        final_score = ((bert_score * WEIGHT_BERT) + (llm_quality_average * WEIGHT_LLM)) / TOTAL_WEIGHT
        
        return {
            "bert_score": bert_score,
            "llm_score": round(llm_quality_average, 1),
            "final_score": round(final_score, 2),
            "status": "ACCEPTED"
        }
    except Exception as e:
        return {"status": "ERROR", "msg": str(e)}
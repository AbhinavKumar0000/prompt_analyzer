import os
import re
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- CONFIGURATION ---
# 1. Your Custom BERT Endpoint (Live on Hugging Face Spaces)
BERT_API_URL = "https://abhinavdread-prompt-analyzer.hf.space/score"


HF_TOKEN = os.environ.get("HF_TOKEN") 

WEIGHT_BERT = 65
WEIGHT_LLM = 35
TOTAL_WEIGHT = WEIGHT_BERT + WEIGHT_LLM

@tool
def get_quality_score(text: str) -> float:
    """Sends the prompt to the cloud-hosted BERT model for scoring."""
    try:
        response = requests.post(BERT_API_URL, json={"prompt": text})
        
        if response.status_code == 200:
            # FIX: Your API returns {"score": 95.5}, so we must use "score" here
            return response.json().get("score", 0.0) 
        else:
            print(f"Cloud BERT Error {response.status_code}: {response.text}")
            return 0.0
    except Exception as e:
        print(f"Connection Error: {e}")
        return 0.0

# --- 2. INITIALIZE CLOUD LLM (Mistral-7B) ---
print("⚡ Connecting to Hugging Face Cloud LLM...")

try:
    # We use Mistral-7B-Instruct-v0.2 because it is free, fast, and smart
    llm_engine = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,  # Deterministic output
        temperature=0.1,
        huggingfacehub_api_token=HF_TOKEN
    )
    llm = ChatHuggingFace(llm=llm_engine)
    print("Connected to Cloud LLM!")
except Exception as e:
    print(f"LLM Config Error: {e}")
    print("TIP: Did you paste your HF Token in the script?")
    exit()

# --- 3. METRICS RUBRIC ---
METRIC_RUBRIC = """
1. Clarity and Ambiguity
2. Specificity and Information Density
3. Contextual Priming
4. Constraint Adherence
5. Reasoning Complexity
6. Goal Alignment
"""

def analyze_prompt_flow(user_prompt: str):
    print(f"\nSending to Cloud BERT: '{user_prompt[:30]}...'")
    
    # STEP A: Get Cloud Score from your custom model
    bert_score = get_quality_score.invoke(user_prompt)
    print(f"BERT Score: {bert_score}/100")

    # STEP B: Gatekeeper (Reject low quality immediately)
    if bert_score < 30:
        return {
            "bert_score": bert_score,
            "llm_score": 0,
            "final_score": bert_score,
            "status": "REJECTED"
        }
    
    # STEP C: Cloud LLM Evaluation
    print("Running Cloud LLM Analysis (Mistral-7B)...")
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are a scoring engine. Rate the prompt on these 6 metrics (0-100).
    RUBRIC: {rubric}
    PROMPT: "{user_prompt}"
    
    OUTPUT FORMAT (Strictly numbers only):
    Metric 1: [Score]/100
    Metric 2: [Score]/100
    Metric 3: [Score]/100
    Metric 4: [Score]/100
    Metric 5: [Score]/100
    Metric 6: [Score]/100
    LLM_Average: [Score]/100
    """)
    
    try:
        chain = prompt_template | llm
        response = chain.invoke({
            "rubric": METRIC_RUBRIC,
            "user_prompt": user_prompt
        })
        
        content = response.content
        match = re.search(r"LLM_Average:\s*(\d+(\.\d+)?)", content)
        
        llm_score = float(match.group(1)) if match else bert_score

        final_score = ((bert_score * WEIGHT_BERT) + (llm_score * WEIGHT_LLM)) / TOTAL_WEIGHT
        
        return {
            "bert_score": bert_score,
            "llm_score": llm_score,
            "final_score": round(final_score, 2),
            "status": "ACCEPTED"
        }
    except Exception as e:
        print(f"LM Inference Failed: {e}")
        return {"status": "ERROR", "msg": str(e)}

if __name__ == "__main__":
    # Test Prompt
    test_prompt = "I want you to act as a stand-up comedian. I will provide you with some topics related to current events and you will use your wit, creativity, and observational skills to create a routine based on those topics. You should also be sure to incorporate personal anecdotes or experiences into the routine in order to make it more relatable and engaging for the audience. My first request is 'I want an humorous take on politics.'"
    
    result = analyze_prompt_flow(test_prompt)

    if result.get("status") == "ACCEPTED":
        print("\n" + "="*30)
        print("      PROMPT QUALITY REPORT      ")
        print("="*30)
        print(f"Cloud BERT Score      : {result['bert_score']}/100")
        print(f"Cloud LLM Score       : {result['llm_score']}/100")
        print("-" * 30)
        print(f"🏆 FINAL HYBRID SCORE : {result['final_score']}/100")
        print("="*30 + "\n")
    else:
        print(f"\n❌ Request Rejected: {result.get('status')} (Score: {result.get('bert_score', 0)})")
from flask import Flask, render_template, request, jsonify
from app_flow import analyze_prompt_flow

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({'error': 'Please enter a prompt.'}), 400

    try:
        # Call your existing AI pipeline
        result = analyze_prompt_flow(prompt)
        
        # If rejected by Gatekeeper
        if result.get("status") == "REJECTED":
            return jsonify({
                "success": True,
                "final_score": result['final_score'],
                "bert_score": result['bert_score'],
                "llm_score": 0,
                "status": "REJECTED",
                "msg": "Prompt is too vague or low quality."
            })
            
        # If Accepted
        return jsonify({
            "success": True,
            "final_score": result['final_score'],
            "bert_score": result['bert_score'],
            "llm_score": result['llm_score'],
            "status": "ACCEPTED"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from pyngrok import ngrok
# IMPORTANT: Replace 'YOUR_NGROK_AUTH_TOKEN' with your actual ngrok authentication token.
# You can get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok.set_auth_token('YOUR_NGROK_AUTH_TOKEN')
from flask import Flask, request, render_template_string
from transformers import pipeline
from pyngrok import ngrok

app = Flask(__name__)

# cache the pipelines so they load only once
pipelines_cache = {}

def get_pipeline(task):
    """Lazy-load the pipeline for each task."""
    if task not in pipelines_cache:
        if task == "sentiment-analysis":
            pipelines_cache[task] = pipeline("sentiment-analysis",
                                             model="distilbert-base-uncased-finetuned-sst-2-english")
        elif task == "summarization":
            pipelines_cache[task] = pipeline("summarization",
                                             model="sshleifer/distilbart-cnn-12-6")
        elif task == "zero-shot-classification":
            pipelines_cache[task] = pipeline("zero-shot-classification",
                                             model="facebook/bart-large-mnli")
    return pipelines_cache[task]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""
    selected_task = "sentiment-analysis"
    candidate_labels = ""

    if request.method == "POST":
        selected_task = request.form.get("task")
        input_text = request.form.get("input_text", "")
        candidate_labels = request.form.get("labels", "")

        if input_text.strip():
            nlp = get_pipeline(selected_task)
            if selected_task == "sentiment-analysis":
                result = nlp(input_text)
            elif selected_task == "summarization":
                result = nlp(input_text, max_length=80, min_length=20, do_sample=False)
            elif selected_task == "zero-shot-classification":
                labels = [lbl.strip() for lbl in candidate_labels.split(",") if lbl.strip()]
                result = nlp(input_text, candidate_labels=labels)

    html = '''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Hugging Face NLP Playground (Colab)</title>
      <style>
        body{font-family:Arial;background:#f5f7fa;padding:30px}
        form{max-width:700px;margin:auto;background:#fff;padding:20px;
             border-radius:10px;box-shadow:0 0 10px rgba(0,0,0,0.1)}
        textarea,select,input{width:100%;margin:8px 0;padding:10px;font-size:15px}
        button{background:#007bff;color:white;border:none;padding:10px 16px;
               border-radius:6px;cursor:pointer}
        button:hover{background:#0056b3}
        .result{margin-top:20px;background:#eef;padding:15px;border-radius:8px}
      </style>
    </head>
    <body>
      <h2 align="center">Hugging Face NLP Playground</h2>
      <form method="post">
        <label>Task:</label>
        <select id="task" name="task" onchange="toggleLabels()">
          <option value="sentiment-analysis" {% if selected_task=='sentiment-analysis' %}selected{% endif %}>Sentiment Analysis</option>
          <option value="summarization" {% if selected_task=='summarization' %}selected{% endif %}>Summarization</option>
          <option value="zero-shot-classification" {% if selected_task=='zero-shot-classification' %}selected{% endif %}>Zero-Shot Classification</option>
        </select>

        <label>Enter Text:</label>
        <textarea name="input_text" placeholder="Type or paste your text here...">{{ input_text }}</textarea>

        <div id="labels_div" style="display:none;">
          <label>Candidate Labels (comma separated):</label>
          <input type="text" name="labels" value="{{ candidate_labels }}" placeholder="e.g. sports, politics, tech">
        </div>

        <button type="submit">Run</button>

        {% if result %}
        <div class="result">
          <h4>Result:</h4>
          <pre>{{ result | tojson(indent=2) }}</pre>
        </div>
        {% endif %}
      </form>

      <script>
        function toggleLabels(){
          const t=document.getElementById("task").value;
          document.getElementById("labels_div").style.display=(t==="zero-shot-classification")?"block":"none";
        }
        window.onload=toggleLabels;
      </script>
    </body>
    </html>
    '''
    return render_template_string(html, result=result,
                                  input_text=input_text,
                                  selected_task=selected_task,
                                  candidate_labels=candidate_labels)

# Create an ngrok tunnel to make the app accessible
public_url = ngrok.connect(5000)
print("🌐 Public URL →", public_url)

# Run the Flask app
app.run(port=5000)

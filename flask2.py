# app.py -- single-file Flask + HF pipelines with better error handling
from flask import Flask, request, render_template_string
import traceback
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Lazy-loaded pipelines
pipelines = {}

# Recommended smaller models to reduce memory/timeout issues
MODEL_CHOICES = {
    "sentiment-analysis": "distilbert-base-uncased-finetuned-sst-2-english",
    # lightweight summarization model (smaller than bart-large-cnn)
    "summarization": "sshleifer/distilbart-cnn-12-6",
    # zero-shot classification model (still moderately sized)
    "zero-shot-classification": "facebook/bart-large-mnli"
}

def get_pipeline(task):
    """Return a huggingface pipeline instance for the task (cached)."""
    from transformers import pipeline  # import here to isolate import errors
    if task in pipelines:
        return pipelines[task]
    model = MODEL_CHOICES.get(task)
    if not model:
        raise ValueError("Unknown task: " + str(task))
    # device=-1 forces CPU (safer for many users). If you have GPU, change device to 0.
    logging.info(f"Loading pipeline for task {task} using model {model} ...")
    try:
        if task == "sentiment-analysis":
            p = pipeline("sentiment-analysis", model=model, device=-1)
        elif task == "summarization":
            p = pipeline("summarization", model=model, device=-1)
        elif task == "zero-shot-classification":
            p = pipeline("zero-shot-classification", model=model, device=-1)
        else:
            raise ValueError("Unsupported task")
    except Exception:
        logging.error("Failed to load pipeline:\n" + traceback.format_exc())
        raise
    pipelines[task] = p
    logging.info("Pipeline loaded.")
    return p

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HF NLP Single-file App</title>
  <style>
    body{font-family:Arial,Helvetica,sans-serif;background:#f2f4f7;padding:24px}
    .card{max-width:900px;margin:20px auto;background:#fff;padding:18px;border-radius:10px;
          box-shadow:0 6px 18px rgba(0,0,0,0.06)}
    textarea{width:100%;min-height:140px;padding:10px;font-size:15px;border-radius:6px;border:1px solid #d6d9df}
    select,input[type=text]{width:100%;padding:8px;margin-top:6px;margin-bottom:12px;border-radius:6px}
    button{background:#0069ff;color:white;border:none;padding:10px 14px;border-radius:8px;cursor:pointer}
    pre{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto}
    .error{background:#ffecec;color:#900;padding:10px;border-radius:6px;margin-top:10px}
    label{font-weight:600}
  </style>
</head>
<body>
  <div class="card">
    <h2>Hugging Face NLP (single-file Flask)</h2>
    <form method="post">
      <label for="task">Task</label>
      <select id="task" name="task" onchange="toggleLabels()">
        <option value="sentiment-analysis" {{ 'selected' if selected_task=='sentiment-analysis' else '' }}>Sentiment Analysis</option>
        <option value="summarization" {{ 'selected' if selected_task=='summarization' else '' }}>Summarization</option>
        <option value="zero-shot-classification" {{ 'selected' if selected_task=='zero-shot-classification' else '' }}>Zero-Shot Classification</option>
      </select>

      <label for="input_text">Input text</label>
      <textarea id="input_text" name="input_text" placeholder="Paste your text here...">{{ input_text }}</textarea>

      <div id="labels_div" style="display:none">
        <label for="labels">Candidate labels (comma separated)</label>
        <input id="labels" name="labels" value="{{ candidate_labels }}" placeholder="e.g. sports, politics, tech">
      </div>

      <label for="model_override">(Optional) Override model name</label>
      <input id="model_override" name="model_override" value="{{ model_override }}" placeholder="Optional HF model id">

      <button type="submit">Run</button>
    </form>

    {% if busy %}
      <p><em>Processing... check server console for progress messages.</em></p>
    {% endif %}

    {% if result %}
      <h3>Result</h3>
      <pre>{{ result }}</pre>
    {% endif %}

    {% if error %}
      <div class="error">
        <strong>Error:</strong>
        <pre>{{ error }}</pre>
        <p><em>Check server console for full trace. Common fixes: run <code>pip install -U transformers torch</code>, ensure internet access for model download, or use a smaller model.</em></p>
      </div>
    {% endif %}

    <hr>
    <p><strong>Notes:</strong> models load lazily on first run and may take time (and download bandwidth). If you get an out-of-memory error, try smaller models or run on a machine with more RAM.</p>
  </div>

<script>
function toggleLabels() {
  const t = document.getElementById("task").value;
  document.getElementById("labels_div").style.display = (t === "zero-shot-classification") ? "block" : "none";
}
window.onload = toggleLabels;
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    input_text = ""
    selected_task = "sentiment-analysis"
    candidate_labels = ""
    model_override = ""

    if request.method == "POST":
        selected_task = request.form.get("task") or selected_task
        input_text = request.form.get("input_text", "").strip()
        candidate_labels = request.form.get("labels", "").strip()
        model_override = request.form.get("model_override", "").strip()

        if not input_text:
            error = "Please provide some input text."
        else:
            try:
                # Optionally override which model to use
                if model_override:
                    # temporarily set chosen model
                    MODEL_CHOICES[selected_task] = model_override

                nlp = get_pipeline(selected_task)
                # Run appropriate pipeline
                if selected_task == "sentiment-analysis":
                    out = nlp(input_text)
                elif selected_task == "summarization":
                    # safe defaults for short text
                    out = nlp(input_text, max_length=120, min_length=20, do_sample=False)
                elif selected_task == "zero-shot-classification":
                    labels = [l.strip() for l in candidate_labels.split(",") if l.strip()]
                    if not labels:
                        raise ValueError("Zero-shot requires at least one comma-separated label.")
                    out = nlp(input_text, candidate_labels=labels)
                else:
                    raise ValueError("Unsupported task: " + str(selected_task))
                # convert result to a readable string
                result = out
            except Exception as e:
                logging.error("Error during pipeline execution:\n" + traceback.format_exc())
                error = str(e) + "\n\n" + traceback.format_exc().splitlines()[-10:]  # short trace

    # render template
    return render_template_string(HTML,
                                  result=result,
                                  error=error,
                                  input_text=input_text,
                                  selected_task=selected_task,
                                  candidate_labels=candidate_labels,
                                  model_override=model_override,
                                  busy=False)

if __name__ == "__main__":
    print("Starting app on http://127.0.0.1:5000")
    # debug True helps show server errors during development
    app.run(debug=True, host="127.0.0.1", port=5000)


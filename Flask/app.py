from flask import Flask, render_template, request
from qa_engine import get_answer
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        file = request.files["document"]
        question = request.form["question"]

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        answer = get_answer(file_path, question)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)

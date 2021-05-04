from flask import Flask, render_template, request
app = Flask(__name__)

from utils import translate_text

@app.route('/')
def render_index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    print(translate_text(request.form['english']))
    return request.form['english']

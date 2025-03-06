from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Gaming Performance Analyzer - Dashboard"

if __name__ == '__main__':
    app.run(debug=True)

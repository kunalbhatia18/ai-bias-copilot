from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})  # Allow requests only from your frontend

@app.route('/')
def home():
    return jsonify({'message': 'Bias Co-Pilot API Running 🚀'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# source venv/bin/activate  # On Windows: venv\Scripts\activate       
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
from bias_copilot import analyze_file  # Import from your Day 5/6 script

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})  # Allow React frontend

# Configure logging for Flask
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

@app.route('/')
def home():
    """Basic health check endpoint."""
    logging.info("Home endpoint accessed")
    return jsonify({'message': 'Bias Co-Pilot API Running 🚀'})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a CSV file for bias and mitigation."""
    try:
        if 'file' not in request.files:
            logging.error("No file provided in request")
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            logging.error("Invalid file type: %s", file.filename)
            return jsonify({'success': False, 'error': 'File must be a CSV'}), 400
        
        # Save temp file and analyze
        temp_path = 'temp_upload.csv'
        file.save(temp_path)
        logging.info("Received file: %s", file.filename)
        
        result, df = analyze_file(temp_path)
        
        # Clean up temp file (optional, could keep for debugging)
        import os
        os.remove(temp_path)
        
        logging.info("Analysis completed successfully")
        return jsonify({
            'success': True,  # Added success field
            'before': {
                'impact': float(result['before']['impact']),
                'males': float(result['before']['males'] * 100),  # Convert to percentage
                'females': float(result['before']['females'] * 100)  # Convert to percentage
            },
            'after': {
                'impact': float(result['after']['impact']),
                'males': float(result['after']['males'] * 100),  # Convert to percentage
                'females': float(result['after']['females'] * 100)  # Convert to percentage
            }
        })
    except Exception as e:
        logging.error("Analysis failed: %s", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting Flask API on port 5000")
    app.run(debug=True, port=5000)
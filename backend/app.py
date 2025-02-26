# app.py 

from flask import Flask, request, jsonify, Response
from flask_cors import CORS # type: ignore
import logging
import sys
import os
from bias_copilot import BiasAnalyzer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

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
        
        temp_path = 'temp_upload.csv'
        file.save(temp_path)
        logging.info("Received file: %s", file.filename)
        
        analyzer = BiasAnalyzer()
        result = analyzer.analyze(temp_path)
        os.remove(temp_path)
        
        logging.info("Analysis completed successfully")
        return jsonify({
            'success': True,
            'before': {
                'gender': {
                    'impact': result['before']['gender']['disparate_impact'],
                    'males': result['before']['gender']['positive_rate_privileged'] * 100,
                    'females': result['before']['gender']['positive_rate_unprivileged'] * 100
                },
                'race': {
                    'impact': result['before']['race']['disparate_impact'],
                    'privileged': result['before']['race']['positive_rate_privileged'] * 100,
                    'unprivileged': result['before']['race']['positive_rate_unprivileged'] * 100
                },
                'age': {
                    'impact': result['before']['age_bin']['disparate_impact'],
                    'old': result['before']['age_bin']['positive_rate_privileged'] * 100,
                    'young': result['before']['age_bin']['positive_rate_unprivileged'] * 100
                },
                'accuracy': result['before_accuracy']
            },
            'after': {
                'gender': {
                    'impact': result['after']['gender']['disparate_impact'],
                    'males': result['after']['gender']['positive_rate_privileged'] * 100,
                    'females': result['after']['gender']['positive_rate_unprivileged'] * 100
                },
                'race': {
                    'impact': result['after']['race']['disparate_impact'],
                    'privileged': result['after']['race']['positive_rate_privileged'] * 100,
                    'unprivileged': result['after']['race']['positive_rate_unprivileged'] * 100
                },
                'age': {
                    'impact': result['after']['age_bin']['disparate_impact'],
                    'old': result['after']['age_bin']['positive_rate_privileged'] * 100,
                    'young': result['after']['age_bin']['positive_rate_unprivileged'] * 100
                },
                'accuracy': result['after_accuracy']
            }
        })
    except Exception as e:
        logging.error("Analysis failed: %s", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/mitigate', methods=['POST'])
def mitigate():
    """Mitigate bias in a CSV file and return reweighted dataset."""
    try:
        if 'file' not in request.files:
            logging.error("No file provided in request")
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            logging.error("Invalid file type: %s", file.filename)
            return jsonify({'success': False, 'error': 'File must be a CSV'}), 400
        
        temp_path = 'temp_upload.csv'
        file.save(temp_path)
        logging.info("Received file for mitigation: %s", file.filename)
        
        analyzer = BiasAnalyzer()
        result = analyzer.analyze(temp_path)
        reweighted_csv = result['reweighted_data'].to_csv(index=False)
        os.remove(temp_path)
        
        logging.info("Mitigation completed successfully")
        return Response(reweighted_csv, mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=reweighted_dataset.csv"})
    except Exception as e:
        logging.error("Mitigation failed: %s", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting Flask API on port 5000")
    app.run(debug=True, port=5000)
#!/usr/bin/env python3
"""
Melbourne Commute Planner - Flask Application Server
Run this script to serve the application locally using Flask
"""

import os
import sys
import webbrowser
import time
from datetime import datetime
from threading import Timer

import requests

from prediction import find_nearby_free_slots_by_location

# Try Flask first, fall back to simple HTTP server if not available
try:
    from flask import Flask, render_template, request, jsonify
    from flask_cors import CORS

    USE_FLASK = True
except ImportError:
    USE_FLASK = False
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import mimetypes

# Configuration
PORT = 5000
HOST = 'localhost'
# Api key
GOOGLE_API_KEY = 'AIzaSyBE2C9ZB9XCENHh8ujnbWxcxIBNzd5Zh84'


def open_browser():
    """Open the default web browser after a short delay"""
    time.sleep(1.5)
    url = f"http://{HOST}:{PORT}"
    print(f"\n🌐 Opening browser at {url}")
    webbrowser.open(url)


# ============= FLASK APPLICATION =============
if USE_FLASK:
    app = Flask(__name__)
    CORS(app)  # Enable CORS for API calls

    # Configuration (替换成你自己的)
    app.config['SECRET_KEY'] = 'goz0A14haOY1EKir7ZYx4v4gxQ='
    app.config['GOOGLE_API_KEY'] = GOOGLE_API_KEY


    @app.route('/')
    def index():
        """Render the main page"""
        # Check if template exists
        template_path = os.path.join(app.template_folder, 'index.html')
        if not os.path.exists(template_path):
            return f"""
            <html>
            <body style="font-family: Arial; padding: 40px;">
                <h1>⚠️ Template Not Found</h1>
                <p>The file <code>templates/index.html</code> does not exist.</p>
                <p>Please ensure your HTML file is saved in:</p>
                <code>{template_path}</code>
                <hr>
                <p><small>Current working directory: {os.getcwd()}</small></p>
            </body>
            </html>
            """, 404

        return render_template('index.html', api_key=app.config['GOOGLE_API_KEY'])


    @app.route('/plan', methods=['POST'])
    def plan_route():
        """Process route planning request (optional endpoint for future expansion)"""
        try:
            data = request.json
            origin = data.get('origin')
            destination = data.get('destination')
            avoid_tolls = data.get('avoidTolls', False)
            avoid_highways = data.get('avoidHighways', False)

            # This endpoint can be used for server-side processing if needed
            # For now, just return success as the client handles everything
            return jsonify({
                'status': 'success',
                'message': 'Route calculated successfully',
                'origin': origin,
                'destination': destination
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400


    @app.route("/available_parking", methods=['POST'])
    def available_parking():
        """
        POST JSON: { "lat": -37.8136, "lng": 144.9631}
        returns: Array of all parking locations
        """
        data = request.get_json(silent=True) or {}
        try:
            lat = round(float(data.get("lat")), 4)
            lng = round(float(data.get("lng")))
        except (TypeError, ValueError):
            return jsonify({"error": "lat/lng required as numbers"}), 400

        result = find_nearby_free_slots_by_location(
            lat,
            lng,
            current_time=datetime.now(),
            top_k=5,
            radius_m=2000)

        all_coords = []
        if len(result) >= 1 and result[0] != 0:
            for kerbside_id in result:
                real_time_api = f"https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/on-street-parking-bay-sensors/records?where=kerbsideid%3D{kerbside_id}&limit=1"
                r = requests.get(real_time_api)
                data = r.json()
                try:
                    slot = data["results"][0]
                    coords = slot["location"]
                    all_coords.append(coords)
                except IndexError:
                    all_coords

        return jsonify({
            'status': 'success',
            "coords": all_coords,
        }), 200


    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'server': 'Flask',
            'api_key_configured': bool(app.config['GOOGLE_API_KEY'])
        })


    def run_flask():
        """Run the Flask application"""
        print("""
╔════════════════════════════════════════════════════════╗
║     🚗 Melbourne Commute Planner - Flask Server 🚗      ║
╚════════════════════════════════════════════════════════╝
        """)

        # Check if template exists
        template_path = os.path.join('templates', 'index.html')
        if not os.path.exists(template_path):
            print(f"⚠️  Warning: {template_path} not found!")
            print(f"   Please ensure your HTML file is saved in the templates/ directory")
            print(f"   Current directory: {os.getcwd()}\n")
        else:
            print(f"✅ Found template: {template_path}")

        print(f"✅ Server starting...")
        print(f"📍 Server Address: http://{HOST}:{PORT}")
        print(f"🔑 Google Maps API Key: Configured ✓")
        print(f"\n📋 Quick Instructions:")
        print(f"   1. The browser will open automatically")
        print(f"   2. Enter any Melbourne address or landmark")
        print(f"   3. Select your route preferences")
        print(f"   4. Click 'Calculate Best Route'\n")
        print(f"⚠️  Press Ctrl+C to stop the server\n")
        print(f"{'=' * 60}\n")

        # Open browser automatically
        Timer(0.5, open_browser).start()

        try:
            app.run(host='0.0.0.0', port=PORT, debug=False)
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped by user")
            sys.exit(0)

# ============= SIMPLE HTTP SERVER FALLBACK =============
else:
    class TemplateHTTPRequestHandler(SimpleHTTPRequestHandler):
        """Custom HTTP handler that serves templates directory"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory='.', **kwargs)

        def do_GET(self):
            """Handle GET requests"""
            if self.path == '/' or self.path == '':
                self.path = '/templates/index.html'

            # Check if requesting root and template exists
            if self.path == '/templates/index.html':
                template_path = os.path.join(os.getcwd(), 'templates', 'index.html')
                if not os.path.exists(template_path):
                    self.send_error(404, f"Template not found: {template_path}")
                    return

            return SimpleHTTPRequestHandler.do_GET(self)

        def end_headers(self):
            """Add CORS headers"""
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            SimpleHTTPRequestHandler.end_headers(self)

        def guess_type(self, path):
            """Properly handle MIME types"""
            mimetype = super().guess_type(path)
            if path.endswith('.js'):
                return 'application/javascript'
            if path.endswith('.css'):
                return 'text/css'
            return mimetype

        def log_message(self, format, *args):
            """Suppress default logging for cleaner output"""
            if "GET /" in format % args or "POST /" in format % args:
                print(f"📍 {format % args}")


    def run_simple_server():
        """Run simple HTTP server as fallback"""
        print("""
╔════════════════════════════════════════════════════════╗
║  🚗 Melbourne Commute Planner - Simple HTTP Server 🚗   ║
║         (Install Flask for better functionality)        ║
╚════════════════════════════════════════════════════════╝
        """)

        print("⚠️  Flask not installed. Using simple HTTP server.")
        print("   For better functionality, install Flask:")
        print("   pip install flask flask-cors\n")

        # Check if template exists
        template_path = os.path.join('templates', 'index.html')
        if not os.path.exists(template_path):
            print(f"❌ Error: {template_path} not found!")
            print(f"   Please ensure your HTML file is saved in:")
            print(f"   {os.path.join(os.getcwd(), 'templates', 'index.html')}")
            sys.exit(1)
        else:
            print(f"✅ Found template: {template_path}")

        try:
            # Create server
            server = HTTPServer((HOST, PORT), TemplateHTTPRequestHandler)

            print(f"✅ Server started successfully!")
            print(f"📍 Server Address: http://{HOST}:{PORT}")
            print(f"🔑 Google Maps API Key: {GOOGLE_API_KEY[:10]}...")
            print(f"\n⚠️  Press Ctrl+C to stop the server\n")
            print(f"{'=' * 60}\n")

            # Open browser automatically
            Timer(0.5, open_browser).start()

            # Start server
            server.serve_forever()

        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped by user")
            sys.exit(0)
        except OSError as e:
            if e.errno == 48 or 'Address already in use' in str(e):
                print(f"\n❌ Error: Port {PORT} is already in use!")
                print(f"   Try closing other applications or changing the PORT variable")
            else:
                print(f"\n❌ Error starting server: {e}")
            sys.exit(1)


# ============= MAIN FUNCTION =============
def main():
    """Main function to run the appropriate server"""
    # Check Python version
    if sys.version_info < (3, 6):
        print("❌ Error: Python 3.6 or higher is required")
        sys.exit(1)

    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created templates/ directory")
        print("   Please save your index.html file in this directory\n")

    # Check which server to run
    if USE_FLASK:
        try:
            run_flask()
        except Exception as e:
            print(f"❌ Error running Flask server: {e}")
            print("Falling back to simple HTTP server...")
            run_simple_server()
    else:
        run_simple_server()


# ============= REQUIREMENTS CHECK =============
def check_requirements():
    """Check and display missing requirements"""
    requirements = []

    try:
        import flask
    except ImportError:
        requirements.append('flask')

    try:
        import flask_cors
    except ImportError:
        requirements.append('flask-cors')

    if requirements:
        print("\n📦 Optional packages not installed:")
        print(f"   pip install {' '.join(requirements)}")
        print("   (The app will still work with basic functionality)\n")
        time.sleep(2)


if __name__ == "__main__":
    check_requirements()
    try:
        main()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

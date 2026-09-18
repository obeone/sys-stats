#!/usr/bin/env python3

import logging
import os

import coloredlogs
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from .collectors import collect_stats

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.png')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'favicon.png', mimetype='image/png')


@app.route('/stats', methods=['GET'])
def get_stats():
    """Serve the ``/stats`` payload consumed by the web UI and the Rich CLI.

    A thin route: it only parses the ``limit`` query parameter and hands off
    to :func:`sys_stats.collectors.collect_stats` for the actual collection
    work, then jsonifies the result unchanged.

    Returns
    -------
    flask.Response
        JSON response whose top-level keys are the public contract of the
        project; adding keys is a minor bump, renaming one is a major bump.
    """
    limit_str = request.args.get("limit", "5")
    try:
        limit = int(limit_str)
    except ValueError:
        limit = 5

    return jsonify(collect_stats(limit=limit))

def main() -> None:
    """Console-script entry point: run the Flask metrics server.

    Honours the ``FLASK_DEBUG``, ``HOST`` and ``PORT`` environment variables.
    """
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=debug_mode)


if __name__ == '__main__':
    main()

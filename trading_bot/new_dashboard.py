"""Minimal futuristic dashboard blueprint and server."""

from __future__ import annotations

from flask import Blueprint, Flask, jsonify, render_template


dashboard_bp = Blueprint("dashboard", __name__, template_folder="templates")


@dashboard_bp.route("/")
def landing():
    """Serve the new cyberpunk landing page."""

    return render_template("dashboard_hero.html")


def create_app() -> Flask:
    """Build a lightweight Flask app that only serves the new dashboard."""

    app = Flask(__name__)
    app.register_blueprint(dashboard_bp)

    @app.get("/api/health")
    def api_health():
        return jsonify({"ok": True})

    return app


def start_dashboard(host: str, port: int) -> None:
    """Launch the minimal dashboard server."""

    app = create_app()
    app.run(host=host, port=port, use_reloader=False)

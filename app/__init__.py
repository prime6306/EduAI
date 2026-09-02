"""
Application factory. Builds and configures the Flask app, registers
extensions and blueprints, and preloads heavy ML artifacts once at startup
(Flask 3 removed `before_first_request`, so we preload here instead — the
effect the spec asks for, "loaded once at startup, not per request", is the
same either way).
"""
try:
    import dlib  # noqa: F401
    import face_recognition  # noqa: F401
except Exception:
    pass

import logging
import os

from flask import Flask, render_template, request, jsonify

from app.config import Config
from app.extensions import db, login_manager, jwt, csrf, mail
from flask_wtf.csrf import CSRFError


def create_app(config_class: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    _configure_logging(app)
    _init_extensions(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _register_context_processors(app)
    _ensure_indexes(app)
    _preload_models(app)
    _init_scheduler(app)

    return app


def _configure_logging(app: Flask) -> None:
    level = logging.DEBUG if app.config["DEBUG"] else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _init_extensions(app: Flask) -> None:
    db.init_app(app)

    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to continue."
    login_manager.login_message_category = "info"

    jwt.init_app(app)
    csrf.init_app(app)
    mail.init_app(app)

    # Ensure upload dir exists.
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    from app.auth.utils import load_user  # noqa: F401  (registers user_loader)


def _register_blueprints(app: Flask) -> None:
    from app.auth.routes import auth_bp
    from app.dashboard.routes import dashboard_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(dashboard_bp, url_prefix="/dashboard")

    # Feature modules — each registers its own url_prefix internally.
    # Wrapped defensively so a module still under construction can't take
    # down the whole app; a warning is logged instead.
    module_imports = [
        ("app.modules.nlp.routes", "nlp_bp"),
        ("app.modules.rag.routes", "rag_bp"),
        ("app.modules.attendance.routes", "attendance_bp"),
        ("app.modules.attendance.routes", "attendance_api_bp"),
        ("app.modules.attendance.routes", "students_bp"),
        ("app.modules.attendance.routes", "students_api_bp"),
        ("app.modules.attendance.corrections_routes", "corrections_bp"),
        ("app.modules.dropout.routes", "dropout_bp"),
        ("app.modules.wellness.routes", "wellness_bp"),
        ("app.modules.plagiarism.routes", "plagiarism_bp"),
        ("app.modules.analytics.routes", "analytics_bp"),
        ("app.modules.announcements.routes", "announcements_bp"),
        ("app.modules.question_paper.routes", "question_paper_bp"),
        ("app.modules.tests.routes", "tests_bp"),
        ("app.modules.report_cards.routes", "report_cards_bp"),
        ("app.modules.syllabus.routes", "syllabus_bp"),
        ("app.modules.co_po.routes", "co_po_bp"),
        ("app.modules.digest.routes", "digest_bp"),
        ("app.modules.admin.audit_routes", "admin_bp"),
        ("app.modules.interview.routes", "interview_bp"),
        ("app.modules.interview.routes", "interview_api_bp"),
    ]
    registered = set()
    for module_path, bp_name in module_imports:
        if module_path in registered:
            continue
        try:
            mod = __import__(module_path, fromlist=[bp_name])
            for attr in dir(mod):
                if attr.endswith("_bp"):
                    app.register_blueprint(getattr(mod, attr))
            registered.add(module_path)
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Module '%s' not loaded: %s", module_path, exc)

    from flask import redirect, url_for

    @app.route("/")
    def index():
        return redirect(url_for("dashboard.home"))


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(403)
    def forbidden(e):
        return render_template("errors/403.html"), 403

    @app.errorhandler(500)
    def server_error(e):
        app.logger.exception("Unhandled server error on %s", request.path)
        return render_template("errors/500.html"), 500

    @app.errorhandler(CSRFError)
    def csrf_error(e):
        # Return JSON for AJAX/API requests so fetch() callers get a parseable
        # error response instead of an HTML 400 page that breaks resp.json().
        if request.is_json or request.headers.get("X-CSRFToken") or request.headers.get("X-CSRF-Token"):
            return jsonify({"error": "Session expired — please refresh the page and try again."}), 400
        return render_template("errors/404.html"), 400  # fallback for non-AJAX


def _register_context_processors(app: Flask) -> None:
    from datetime import datetime, timedelta

    @app.context_processor
    def inject_globals():
        return {
            "now": datetime.utcnow(),
            "app_name": "EduAI",
        }

    # Available in every template without needing to pass it explicitly —
    # several pages (digest, planner) do date arithmetic like
    # `(some_date - timedelta(days=1))` for display formatting.
    app.jinja_env.globals["timedelta"] = timedelta


def _init_scheduler(app: Flask) -> None:
    """
    Weekly Digest scheduling (Module 18): a Sunday 23:00 cron job inside
    this Flask process, plus a startup catch-up run in case the server was
    offline when the job would have fired. Both paths are idempotent
    (digest_service checks for an existing digest before generating one),
    so it's safe for this to run on every boot.

    Skipped entirely in tests, when explicitly disabled via config, or if
    APScheduler isn't installed — none of which should take down the app.
    """
    if app.config.get("TESTING") or not app.config.get("DIGEST_SCHEDULER_ENABLED", True):
        return

    # Avoid double-starting the scheduler under Flask's debug reloader,
    # which forks a child process — only the reloaded child should run it.
    if app.config.get("DEBUG") and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        app.logger.warning("APScheduler not installed — weekly digest will not auto-run. "
                            "Use POST /api/digest/send-now to generate one manually.")
        return

    from app.modules.digest import digest_service

    try:
        digest_service.run_weekly_job(app)
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Startup digest catch-up check failed (will retry next boot): %s", exc)

    try:
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            func=lambda: digest_service.run_weekly_job(app),
            trigger="cron", day_of_week="sun", hour=23, minute=0,
            id="weekly_digest", replace_existing=True,
        )
        scheduler.start()
        app.extensions["digest_scheduler"] = scheduler

        import atexit
        atexit.register(lambda: scheduler.shutdown(wait=False))
        app.logger.info("Weekly digest scheduler started — runs every Sunday 23:00.")
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Could not start the digest scheduler: %s", exc)


def _ensure_indexes(app: Flask) -> None:
    """
    Best-effort index creation for collections added by the infra features
    (rate limiting, audit trail). Wrapped defensively — an unreachable
    Mongo at boot must not crash the app, matching every other startup
    step here.
    """
    try:
        from app.utils.rate_limiter import ensure_indexes as ensure_rate_limit_indexes
        ensure_rate_limit_indexes()

        db.audit_log.create_index([("timestamp", -1), ("actor_id", 1)], name="ts_actor")
        db.audit_log.create_index([("action", 1)], name="action")

        from app.modules.interview.store import ensure_indexes as ensure_interview_indexes
        ensure_interview_indexes()
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Could not ensure indexes (will retry next boot): %s", exc)


def _preload_models(app: Flask) -> None:
    """
    Load heavyweight, rarely-changing artifacts once, into module-level
    singletons, so per-request code never pays model-load cost:
      - face encodings cache (ENcodedFile.p)
      - anti-spoof ensemble (pretrained .pkl — never trained by this app)
      - dropout classifiers (auto-trained on first run if absent)
    Each loader is optional/best-effort at import time — a missing model
    file logs a warning rather than crashing startup, since e.g. the
    anti-spoof .pkl is supplied manually by the operator.
    """
    with app.app_context():
        try:
            # Import face_recognition (dlib) eagerly in the main thread.
            # On Windows, dlib's native DLL uses thread-local storage that
            # fails to initialise when first loaded inside a Flask worker
            # thread.  Importing here forces the DLL to initialise once on
            # the main thread; all later lazy imports inside request handlers
            # then simply reuse the already-loaded module with no DLL error.
            import face_recognition  # noqa: F401
            from app.modules.attendance.face_engine import load_face_encodings
            load_face_encodings(app)
        except ImportError as exc:
            app.logger.warning("face_recognition / dlib could not be imported: %s", exc)
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Face encodings not preloaded: %s", exc)

        try:
            from app.modules.attendance.antispoof import load_antispoof_model
            load_antispoof_model(app)
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Anti-spoof model not preloaded: %s", exc)

        try:
            from app.modules.dropout.model import ensure_models_trained
            ensure_models_trained(app)
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Dropout models not preloaded: %s", exc)

"""
Shared extension instances. Instantiated here (unbound), attached to the app
in the factory (app/__init__.py). Import `db` from here anywhere routes need
Mongo collections, e.g. `from app.extensions import db; db.users.find_one(...)`.
"""
import logging
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from flask_login import LoginManager
from flask_jwt_extended import JWTManager
from flask_wtf import CSRFProtect
from flask_mail import Mail

logger = logging.getLogger("eduai")

login_manager = LoginManager()
jwt = JWTManager()
csrf = CSRFProtect()
mail = Mail()


class MongoWrapper:
    """
    Thin lazy wrapper around a pymongo Database so the rest of the app can do
    `db.users.find_one(...)` without worrying about connection lifecycle.
    If Mongo is unreachable, attribute access raises a clear RuntimeError
    instead of hanging or crashing the whole app at import time.
    """

    def __init__(self):
        self._client = None
        self._db = None

    def init_app(self, app):
        uri = app.config["MONGODB_URI"]
        db_name = app.config["MONGODB_DB"]
        try:
            self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self._db = self._client[db_name]
            # Cheap ping — don't let a bad URI silently defer failure to first query.
            self._client.admin.command("ping")
            logger.info("Connected to MongoDB database '%s'", db_name)
        except ServerSelectionTimeoutError as exc:
            logger.warning(
                "Could not reach MongoDB at startup (%s). "
                "The app will still boot; requests touching the DB will fail "
                "until MONGODB_URI is reachable.", exc
            )
            # Keep the client — pymongo retries lazily on first real operation.

    def __getattr__(self, name):
        if self._db is None:
            raise RuntimeError(
                "MongoDB is not initialised. Call db.init_app(app) in the app factory."
            )
        return getattr(self._db, name)

    def __getitem__(self, name):
        if self._db is None:
            raise RuntimeError(
                "MongoDB is not initialised. Call db.init_app(app) in the app factory."
            )
        return self._db[name]

    @property
    def raw(self):
        return self._db

    @property
    def client(self):
        return self._client


db = MongoWrapper()

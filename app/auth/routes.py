from datetime import datetime

import bcrypt
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from flask_jwt_extended import create_access_token, set_access_cookies, unset_jwt_cookies

from app.extensions import db
from app.utils.audit import log_action
from .forms import RegisterForm, LoginForm, ProfileForm
from .utils import User

auth_bp = Blueprint("auth", __name__, template_folder="../templates/auth")


def _hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt())


def _check_password(plain: str, hashed) -> bool:
    if isinstance(hashed, str):
        hashed = hashed.encode("utf-8")
    return bcrypt.checkpw(plain.encode("utf-8"), hashed)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.home"))

    form = RegisterForm()
    if form.validate_on_submit():
        existing = db.users.find_one({"email": form.email.data.lower().strip()})
        if existing:
            flash("An account with that email already exists.", "danger")
            return render_template("auth/register.html", form=form)

        student_id = (form.student_id.data or "").strip()
        if form.role.data == "student" and student_id:
            id_taken = db.users.find_one({"student_id": student_id, "role": "student"})
            if id_taken:
                flash("That Student ID is already registered.", "danger")
                return render_template("auth/register.html", form=form)

        doc = {
            "name": form.name.data.strip(),
            "email": form.email.data.lower().strip(),
            "password_hash": _hash_password(form.password.data),
            "role": form.role.data,
            "branch": form.branch.data,
            "year": form.year.data,
            "student_id": student_id,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }
        result = db.users.insert_one(doc)
        doc["_id"] = result.inserted_id

        user = User(doc)
        login_user(user)
        log_action("auth.registered", {"role": user.role})
        flash(f"Welcome to EduAI, {user.name.split()[0]}!", "success")
        return redirect(url_for("dashboard.home"))

    return render_template("auth/register.html", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.home"))

    form = LoginForm()
    if form.validate_on_submit():
        doc = db.users.find_one({"email": form.email.data.lower().strip()})
        if not doc or not _check_password(form.password.data, doc["password_hash"]):
            flash("Incorrect email or password.", "danger")
            return render_template("auth/login.html", form=form)

        user = User(doc)
        login_user(user, remember=form.remember.data)
        db.users.update_one({"_id": doc["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
        log_action("auth.login", {})

        token = create_access_token(identity=user.id)
        resp = redirect(url_for("dashboard.home"))
        set_access_cookies(resp, token)
        flash(f"Welcome back, {user.name.split()[0]}.", "success")
        return resp

    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    log_action("auth.logout", {})
    logout_user()
    resp = redirect(url_for("auth.login"))
    unset_jwt_cookies(resp)
    flash("You've been signed out.", "info")
    return resp


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    form = ProfileForm(
        name=current_user.name,
        branch=current_user.branch,
        year=current_user.year,
        student_id=current_user.student_id,
    )
    if form.validate_on_submit():
        db.users.update_one(
            {"_id": current_user.to_doc()["_id"]},
            {
                "$set": {
                    "name": form.name.data.strip(),
                    "branch": form.branch.data,
                    "year": form.year.data,
                    "student_id": (form.student_id.data or "").strip(),
                }
            },
        )
        log_action("auth.profile_updated", {})
        flash("Profile updated.", "success")
        return redirect(url_for("auth.profile"))

    return render_template("auth/profile.html", form=form)

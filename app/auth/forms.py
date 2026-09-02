from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    PasswordField,
    SelectField,
    BooleanField,
    SubmitField,
)
from wtforms.validators import DataRequired, Email, Length, EqualTo, Optional

BRANCH_CHOICES = [
    ("ECE", "Electronics & Communication (ECE)"),
    ("CS", "Computer Science (CS)"),
    ("IT", "Information Technology (IT)"),
    ("ME", "Mechanical (ME)"),
    ("CE", "Civil (CE)"),
    ("EE", "Electrical (EE)"),
]

YEAR_CHOICES = [
    ("1", "1st Year"),
    ("2", "2nd Year"),
    ("3", "3rd Year"),
    ("4", "4th Year"),
]

ROLE_CHOICES = [
    ("student", "Student"),
    ("teacher", "Teacher"),
]


class RegisterForm(FlaskForm):
    name = StringField("Full Name", validators=[DataRequired(), Length(max=120)])
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=180)])
    password = PasswordField(
        "Password", validators=[DataRequired(), Length(min=6, message="Minimum 6 characters.")]
    )
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")],
    )
    role = SelectField("Role", choices=ROLE_CHOICES, validators=[DataRequired()])
    branch = SelectField("Branch", choices=BRANCH_CHOICES, validators=[DataRequired()])
    year = SelectField("Year", choices=YEAR_CHOICES, validators=[DataRequired()])
    student_id = StringField("Student ID", validators=[Optional(), Length(max=40)])
    submit = SubmitField("Create Account")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Remember me")
    submit = SubmitField("Sign In")


class ProfileForm(FlaskForm):
    name = StringField("Full Name", validators=[DataRequired(), Length(max=120)])
    branch = SelectField("Branch", choices=BRANCH_CHOICES, validators=[DataRequired()])
    year = SelectField("Year", choices=YEAR_CHOICES, validators=[DataRequired()])
    student_id = StringField("Student ID", validators=[Optional(), Length(max=40)])
    submit = SubmitField("Save Changes")

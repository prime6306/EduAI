# Security Policy

## Supported Versions

The latest version of EduAI is actively maintained.

| Version | Supported |
|--------|-----------|
| Latest | ✅ |
| Older versions | ❌ |

Users are encouraged to always use the latest version of the repository.

---

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

You can report vulnerabilities by:

- Opening a **private security advisory** on GitHub
- Emailing the maintainer

Email: priyanshukashyap178@gmail.com

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

We will respond as soon as possible.

---

## Responsible Disclosure

Please do **not publicly disclose vulnerabilities** until they have been reviewed and addressed.

Responsible reporting helps protect users of the platform.

---

## Security Best Practices

When deploying EduAI:

- Do not expose MongoDB publicly
- Use environment variables for API keys
- Restrict model access if using private APIs
- Enable authentication for production APIs
- Avoid logging sensitive user data

---

## Data Privacy

EduAI may process educational data such as:

- academic performance
- attendance
- assignments

Developers should ensure compliance with local data protection policies when deploying the platform.

---

## Dependencies

Keep dependencies updated to avoid vulnerabilities.

Recommended practices:

- Regularly update Python packages
- Use `pip-audit` or `safety` to check dependency vulnerabilities
- Avoid using outdated ML libraries

---

## Security Scope

This repository focuses on **research and educational use**.

Production deployments should implement additional security measures including:

- authentication
- rate limiting
- secure database configuration
- encrypted communication

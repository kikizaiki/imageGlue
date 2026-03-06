"""ID generation utilities."""
import secrets
import string


def generate_job_id() -> str:
    """Generate unique job ID."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(12))

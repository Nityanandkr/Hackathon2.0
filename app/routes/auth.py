from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.auth.jwt import create_access_token, validate_token

router = APIRouter()

# ── In-memory user store ──────────────────────────────────────────────
# Seeded with a default admin account
_users: dict[str, dict] = {
    "admin": {"password": "password123", "role": "admin"}
}


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class SignupRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str


def _validate_credentials(username: str, password: str):
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if username in _users:
        raise HTTPException(status_code=409, detail="Username already exists")


@router.post("/login", response_model=TokenResponse)
def login(creds: LoginRequest):
    """Authenticate using stored credentials and issue a JWT."""
    user = _users.get(creds.username)
    if user is None or user["password"] != creds.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = create_access_token(
        data={"sub": creds.username, "role": user["role"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=MessageResponse)
def register(creds: RegisterRequest, token: dict = Depends(validate_token)):
    """Create a new user (admin or user role). Requires existing admin JWT."""
    _validate_credentials(creds.username, creds.password)
    role = creds.role if creds.role in ("admin", "user") else "user"
    _users[creds.username] = {"password": creds.password, "role": role}
    return {"message": f"{role.capitalize()} '{creds.username}' created successfully"}


@router.post("/signup", response_model=MessageResponse)
def signup(creds: SignupRequest):
    """Public self-registration for regular users (no JWT required)."""
    _validate_credentials(creds.username, creds.password)
    _users[creds.username] = {"password": creds.password, "role": "user"}
    return {"message": f"Account '{creds.username}' created successfully! You can now log in."}

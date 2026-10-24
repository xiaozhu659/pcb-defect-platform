"""JWT 身份认证:账号密码换取 token,受保护接口校验 token。

账号密码来自环境变量(默认 admin / admin123,部署时务必修改)。
生产环境应把密码改成哈希存储,这里为了教学演示保持简单。
"""

import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

APP_USER = os.environ.get("APP_USER", "admin")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "admin123")
JWT_SECRET = os.environ.get("JWT_SECRET", "please-change-this-secret")
JWT_ALGORITHM = "HS256"
TOKEN_HOURS = int(os.environ.get("TOKEN_HOURS", "12"))

bearer = HTTPBearer(auto_error=False)


def authenticate(username: str, password: str) -> bool:
    """校验账号密码(用 compare_digest 避免逐字符比较的时序泄漏)。"""
    user_ok = hmac.compare_digest(username, APP_USER)
    password_ok = hmac.compare_digest(password, APP_PASSWORD)
    return user_ok and password_ok


def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> str:
    """FastAPI 依赖:从 Authorization: Bearer <token> 中解析用户。"""
    if credentials is None:
        raise HTTPException(401, "未提供 token",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET,
                             algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "token 已过期") from None
    except jwt.InvalidTokenError:
        raise HTTPException(401, "token 无效") from None
    return str(payload.get("sub", ""))

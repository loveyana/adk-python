# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
import urllib.parse
from typing import Any
from typing import Optional

from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import RedirectResponse
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OAuth2Config(BaseModel):
  """OAuth2 configuration for ADK Web Server authentication."""
  
  # OAuth2 Provider URLs
  authorize_url: str
  token_url: str
  
  # Client credentials
  client_id: str
  client_secret: Optional[str] = None
  
  # OAuth2 parameters
  scope: str = "openid profile"
  response_type: str = "code"
  
  # Local server configuration
  redirect_uri: str
  
  # Session configuration
  session_cookie_name: str = "adk_session"
  session_timeout_seconds: int = 3600  # 1 hour

  # User info endpoint (optional)
  userinfo_url: Optional[str] = None


class OAuth2Session(BaseModel):
  """OAuth2 session data stored in cookies."""
  
  access_token: str
  token_type: str = "Bearer"
  expires_at: float
  refresh_token: Optional[str] = None
  user_info: Optional[dict[str, Any]] = None
  
  def is_expired(self) -> bool:
    """Check if the access token is expired."""
    return time.time() >= self.expires_at
  
  def to_authorization_header(self) -> str:
    """Convert to Authorization header value."""
    return f"{self.token_type} {self.access_token}"


class OAuth2StateStore:
  """In-memory store for OAuth2 state parameters."""
  
  def __init__(self):
    self._states: dict[str, dict[str, Any]] = {}
  
  def create_state(self, redirect_after_auth: str = "/") -> str:
    """Create a new OAuth2 state parameter."""
    state = secrets.token_urlsafe(32)
    self._states[state] = {
        "created_at": time.time(),
        "redirect_after_auth": redirect_after_auth,
    }
    return state
  
  def validate_and_consume_state(self, state: str) -> Optional[dict[str, Any]]:
    """Validate and consume an OAuth2 state parameter."""
    if state not in self._states:
      return None
    
    state_data = self._states.pop(state)
    
    # Check if state is expired (5 minutes)
    if time.time() - state_data["created_at"] > 300:
      return None
    
    return state_data


class OAuth2Handler:
  """Handles OAuth2 authentication flow for ADK Web Server."""
  
  def __init__(self, config: OAuth2Config):
    self.config = config
    self.state_store = OAuth2StateStore()
    self._http_client = httpx.AsyncClient()
  
  async def close(self):
    """Close the HTTP client."""
    await self._http_client.aclose()
  
  def get_authorization_url(self, state: str) -> str:
    """Generate the OAuth2 authorization URL."""
    params = {
        "response_type": self.config.response_type,
        "client_id": self.config.client_id,
        "scope": self.config.scope,
        "redirect_uri": self.config.redirect_uri,
        "state": state,
    }
    
    return f"{self.config.authorize_url}?{urllib.parse.urlencode(params)}"
  
  async def exchange_code_for_token(self, code: str) -> OAuth2Session:
    """Exchange authorization code for access token."""
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": self.config.redirect_uri,
    }
    
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    # Use Basic Auth for client credentials
    if self.config.client_secret:
      import base64
      credentials = f"{self.config.client_id}:{self.config.client_secret}"
      encoded_credentials = base64.b64encode(credentials.encode()).decode()
      headers["Authorization"] = f"Basic {encoded_credentials}"
    else:
      # Fallback to client_id in form data if no client_secret
      token_data["client_id"] = self.config.client_id
    
    try:
      response = await self._http_client.post(
          self.config.token_url,
          data=token_data,
          headers=headers,
      )
      response.raise_for_status()
      
      token_response = response.json()
      
      # Calculate expiration time
      expires_in = token_response.get("expires_in", 3600)
      expires_at = time.time() + expires_in

      # Create session with token information
      session = OAuth2Session(
          access_token=token_response["access_token"],
          token_type=token_response.get("token_type", "Bearer"),
          expires_at=expires_at,
          refresh_token=token_response.get("refresh_token"),
      )

      # Fetch user info if userinfo_url is configured
      if self.config.userinfo_url:
        try:
          user_info = await self._fetch_user_info(session.access_token)
          session.user_info = user_info
          logger.info("Successfully fetched user info for user: %s",
                     user_info.get("sub") or user_info.get("email") or "unknown")
        except Exception as e:
          logger.warning("Failed to fetch user info: %s", e)
          # Continue without user info - don't fail the authentication

      return session
      
    except httpx.HTTPStatusError as e:
      logger.error("Token exchange failed: %s", e.response.text)
      raise HTTPException(
          status_code=400, 
          detail=f"Token exchange failed: {e.response.text}"
      )
    except Exception as e:
      logger.error("Token exchange error: %s", e)
      raise HTTPException(status_code=500, detail="Authentication failed")

  async def _fetch_user_info(self, access_token: str) -> dict[str, Any]:
    """Fetch user information from the userinfo endpoint."""
    if not self.config.userinfo_url:
      raise ValueError("userinfo_url not configured")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    try:
      response = await self._http_client.get(
          self.config.userinfo_url,
          headers=headers,
      )
      response.raise_for_status()

      user_info = response.json()
      logger.debug("Fetched user info: %s", user_info)
      return user_info

    except httpx.HTTPStatusError as e:
      logger.error("User info fetch failed: %s", e.response.text)
      raise Exception(f"User info fetch failed: {e.response.text}")
    except Exception as e:
      logger.error("User info fetch error: %s", e)
      raise Exception(f"User info fetch error: {e}")

  def encode_session(self, session: OAuth2Session) -> str:
    """Encode OAuth2 session data for cookie storage."""
    session_json = session.model_dump_json()
    session_bytes = session_json.encode('utf-8')
    return base64.urlsafe_b64encode(session_bytes).decode('ascii')

  def decode_session(self, encoded_session: str) -> Optional[OAuth2Session]:
    """Decode OAuth2 session data from cookie."""
    try:
      session_bytes = base64.urlsafe_b64decode(encoded_session.encode('ascii'))
      session_json = session_bytes.decode('utf-8')
      session_data = json.loads(session_json)
      return OAuth2Session.model_validate(session_data)
    except Exception as e:
      logger.warning("Failed to decode session: %s", e)
      return None

  def get_session_from_request(self, request: Request) -> Optional[OAuth2Session]:
    """Extract OAuth2 session from request cookies."""
    session_cookie = request.cookies.get(self.config.session_cookie_name)
    if not session_cookie:
      return None

    session = self.decode_session(session_cookie)
    if not session or session.is_expired():
      return None

    return session

  def create_session_cookie(self, session: OAuth2Session) -> dict[str, Any]:
    """Create session cookie parameters."""
    encoded_session = self.encode_session(session)

    return {
        "key": self.config.session_cookie_name,
        "value": encoded_session,
        "max_age": self.config.session_timeout_seconds,
        "httponly": True,
        "secure": True,  # Enable in production with HTTPS
        "samesite": "lax",
    }

  def create_user_id_cookie(self, session: OAuth2Session) -> Optional[dict[str, Any]]:
    """Create user ID cookie for frontend access (non-HTTP-only)."""
    if not session.user_info or "sub" not in session.user_info:
      return None

    return {
        "key": "adk_user_id",
        "value": session.user_info["sub"],
        "max_age": self.config.session_timeout_seconds,
        "httponly": False,  # Allow frontend JavaScript access
        "secure": True,  # Enable in production with HTTPS
        "samesite": "lax",
    }


def create_oauth2_middleware(oauth2_handler: OAuth2Handler):
  """Create OAuth2 authentication middleware for FastAPI."""

  async def oauth2_middleware(request: Request, call_next):
    """OAuth2 authentication middleware."""

    # Skip authentication for OAuth2 callback and static assets
    if request.url.path.startswith("/oauth2/"):
      return await call_next(request)

    # Check for existing session
    session = oauth2_handler.get_session_from_request(request)

    if session and not session.is_expired():
      # Add Authorization header to the request
      # Method 1: Modify the scope directly (more compatible)
      auth_header_value = session.to_authorization_header().encode()

      # Get existing headers and add the authorization header
      headers = list(request.scope.get("headers", []))

      # Remove any existing authorization header first
      headers = [(name, value) for name, value in headers
                 if name.lower() != b"authorization"]

      # Add the new authorization header
      headers.append((b"authorization", auth_header_value))

      # Update the scope in place
      request.scope["headers"] = headers

      logger.debug("Added Authorization header to request: %s",
                   session.to_authorization_header()[:20] + "...")

      return await call_next(request)

    # No valid session, redirect to OAuth2 authorization
    state = oauth2_handler.state_store.create_state(
        redirect_after_auth=str(request.url)
    )
    auth_url = oauth2_handler.get_authorization_url(state)

    return RedirectResponse(url=auth_url, status_code=302)

  return oauth2_middleware

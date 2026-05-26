# RentRight — Backend Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI backend with LLM abstraction, jurisdiction-aware RAG pipeline, semantic cache, and the Know Your Rights API endpoint — the foundation every other feature builds on.

**Architecture:** Python 3.12 + FastAPI backend. All LLM calls route through a single `LLMProvider` class using LiteLLM — swapping providers requires changing one string in config. Vector storage and semantic cache both live in Supabase pgvector. Jurisdiction (state + city) is resolved from GPS coordinates on every request and scopes all retrieval and caching.

**Tech Stack:** Python 3.12, FastAPI, LiteLLM, Supabase (Postgres + pgvector), Pydantic v2, reverse_geocoder, pytest, pytest-asyncio, ruff, mypy

**This is Plan 1 of 6.**
- Plan 2: Data Ingestion Pipeline (HUD, US Code, state laws, court opinions, cache warming)
- Plan 3: Lease Analyzer
- Plan 4: Dispute Helper
- Plan 5: Mobile App (React Native + Expo)
- Plan 6: Path B Recurring Features

---

## File Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app, middleware, exception handlers
│   ├── config.py                    # Settings, LLM_CONFIG, EMBEDDING_CONFIG
│   ├── dependencies.py              # FastAPI dependency injection (auth, tier, db)
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── health.py            # GET /api/v1/health
│   │       └── qa.py                # POST /api/v1/qa/ask
│   ├── core/
│   │   ├── __init__.py
│   │   ├── llm_provider.py          # LiteLLM abstraction (LLMProvider)
│   │   ├── embeddings.py            # EmbeddingService
│   │   ├── jurisdiction.py          # JurisdictionResolver, Jurisdiction dataclass
│   │   └── cache.py                 # SemanticCacheService (pgvector-backed)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py             # RAGRetriever (public law + user lease)
│   │   ├── pipeline.py              # RAGPipeline (orchestrates full Q&A flow)
│   │   └── prompts.py               # System prompt templates
│   ├── models/
│   │   ├── __init__.py
│   │   ├── domain.py                # Jurisdiction, Chunk, AnswerResponse dataclasses
│   │   ├── requests.py              # Pydantic request schemas
│   │   └── responses.py             # Pydantic response schemas
│   └── db/
│       ├── __init__.py
│       └── client.py                # Supabase client singleton
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── test_llm_provider.py
│   ├── test_embeddings.py
│   ├── test_jurisdiction.py
│   ├── test_cache.py
│   ├── test_retriever.py
│   ├── test_pipeline.py
│   └── test_api_qa.py
├── sql/
│   └── schema.sql                   # Supabase schema (run once in SQL editor)
├── .env.example
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

## Task 1: Project Setup

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/pyproject.toml`
- Create: `backend/.env.example`
- Create: `backend/Makefile`
- Create: `backend/app/__init__.py` (empty)
- Create: `backend/app/api/__init__.py` (empty)
- Create: `backend/app/api/v1/__init__.py` (empty)
- Create: `backend/app/core/__init__.py` (empty)
- Create: `backend/app/rag/__init__.py` (empty)
- Create: `backend/app/models/__init__.py` (empty)
- Create: `backend/app/db/__init__.py` (empty)
- Create: `backend/tests/__init__.py` (empty)
- Create: `backend/sql/` (empty dir)

- [ ] **Step 1: Create directory structure**

```bash
cd ~/repos/RentRight
mkdir -p backend/app/{api/v1,core,rag,models,db}
mkdir -p backend/{tests,sql}
touch backend/app/__init__.py
touch backend/app/api/__init__.py
touch backend/app/api/v1/__init__.py
touch backend/app/core/__init__.py
touch backend/app/rag/__init__.py
touch backend/app/models/__init__.py
touch backend/app/db/__init__.py
touch backend/tests/__init__.py
```

- [ ] **Step 2: Create `backend/requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
litellm==1.49.0
supabase==2.9.1
pydantic==2.9.2
pydantic-settings==2.5.2
python-dotenv==1.0.1
reverse-geocoder==1.5.1
httpx==0.27.2
numpy==2.1.1

# Dev/test
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-mock==3.14.0
ruff==0.6.8
mypy==1.11.2
```

- [ ] **Step 3: Create `backend/pyproject.toml`**

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 4: Create `backend/.env.example`**

```
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key

# LLM Providers (add keys for providers you use)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

# App
APP_ENV=development
```

- [ ] **Step 5: Create `backend/Makefile`**

```makefile
.PHONY: install dev test lint typecheck

install:
	pip install -r requirements.txt

dev:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -v

lint:
	ruff check app/ tests/

typecheck:
	mypy app/

check: lint typecheck test
```

- [ ] **Step 6: Install dependencies**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages install without errors.

- [ ] **Step 7: Commit**

```bash
cd ~/repos/RentRight
git add backend/
git commit -m "feat: backend project scaffold — structure, deps, tooling

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Supabase Schema

**Files:**
- Create: `backend/sql/schema.sql`

- [ ] **Step 1: Create `backend/sql/schema.sql`**

```sql
-- Enable pgvector extension (run once per project)
create extension if not exists vector;

-- User profiles (tier, jurisdiction preference)
create table if not exists user_profiles (
  id           uuid primary key references auth.users(id) on delete cascade,
  tier         text not null default 'free' check (tier in ('free', 'pro')),
  state        text,
  city         text,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

-- Public law chunks (HUD, US Code, state laws, court opinions)
create table if not exists public_law_chunks (
  id           bigserial primary key,
  source       text not null,        -- e.g. 'hud', 'us_code', 'ca_tenant_law'
  state        text,                 -- null = federal / applies to all states
  city         text,                 -- null = no city-specific rule
  content      text not null,
  metadata     jsonb not null default '{}',
  embedding    vector(1536),
  created_at   timestamptz not null default now()
);

-- User lease chunks (per-user, private)
create table if not exists user_lease_chunks (
  id           bigserial primary key,
  user_id      uuid not null references auth.users(id) on delete cascade,
  lease_id     uuid not null,
  content      text not null,
  metadata     jsonb not null default '{}',
  embedding    vector(1536),
  created_at   timestamptz not null default now()
);

-- Semantic Q&A cache (jurisdiction-scoped)
create table if not exists qa_cache (
  id           bigserial primary key,
  state        text not null,
  city         text not null default '',
  question     text not null,
  answer       text not null,
  embedding    vector(1536),
  hit_count    integer not null default 0,
  created_at   timestamptz not null default now()
);

-- Leases metadata
create table if not exists leases (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null references auth.users(id) on delete cascade,
  filename     text not null,
  end_date     date,
  address      text,
  created_at   timestamptz not null default now()
);

-- pgvector similarity search functions

-- Search public law chunks
create or replace function match_public_law(
  query_embedding vector(1536),
  filter_state    text,
  match_threshold float,
  match_count     int
)
returns table (
  id       bigint,
  source   text,
  state    text,
  city     text,
  content  text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    id, source, state, city, content, metadata,
    1 - (embedding <=> query_embedding) as similarity
  from public_law_chunks
  where
    (state is null or state = filter_state)
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- Search user lease chunks
create or replace function match_user_lease(
  query_embedding vector(1536),
  filter_user_id  uuid,
  match_threshold float,
  match_count     int
)
returns table (
  id         bigint,
  lease_id   uuid,
  content    text,
  metadata   jsonb,
  similarity float
)
language sql stable
as $$
  select
    id, lease_id, content, metadata,
    1 - (embedding <=> query_embedding) as similarity
  from user_lease_chunks
  where
    user_id = filter_user_id
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- Search semantic cache
create or replace function match_qa_cache(
  query_embedding vector(1536),
  filter_state    text,
  filter_city     text,
  match_threshold float
)
returns table (
  id         bigint,
  answer     text,
  similarity float
)
language sql stable
as $$
  select
    id, answer,
    1 - (embedding <=> query_embedding) as similarity
  from qa_cache
  where
    state = filter_state
    and city = filter_city
    and 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit 1;
$$;

-- Indexes for fast vector search
create index if not exists public_law_chunks_embedding_idx
  on public_law_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

create index if not exists user_lease_chunks_embedding_idx
  on user_lease_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 50);

create index if not exists qa_cache_embedding_idx
  on qa_cache using ivfflat (embedding vector_cosine_ops)
  with (lists = 50);
```

- [ ] **Step 2: Run schema in Supabase**

1. Open your Supabase project → SQL Editor
2. Paste the contents of `sql/schema.sql`
3. Click Run
4. Verify in Table Editor: `user_profiles`, `public_law_chunks`, `user_lease_chunks`, `qa_cache`, `leases` all exist

- [ ] **Step 3: Commit**

```bash
cd ~/repos/RentRight
git add backend/sql/schema.sql
git commit -m "feat: supabase schema — pgvector tables + similarity search functions

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Config + Supabase Client

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/app/db/client.py`
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: Create `backend/app/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str
    supabase_service_key: str
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    app_env: str = "development"


settings = Settings()

LLM_CONFIG: dict[str, dict[str, object]] = {
    "free": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 600,
        "temperature": 0.1,
    },
    "pro": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1500,
        "temperature": 0.1,
    },
    "fallback": {
        "model": "groq/llama-3.3-70b-versatile",
        "max_tokens": 600,
        "temperature": 0.1,
    },
}

EMBEDDING_CONFIG = {
    "model": "text-embedding-3-small",
    "dimensions": 1536,
}

CACHE_SIMILARITY_THRESHOLD = 0.90
RETRIEVAL_SIMILARITY_THRESHOLD = 0.75
RETRIEVAL_TOP_K_PUBLIC = 5
RETRIEVAL_TOP_K_LEASE = 3
```

- [ ] **Step 2: Create `backend/app/db/client.py`**

```python
from functools import lru_cache

from supabase import Client, create_client

from app.config import settings


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)
```

- [ ] **Step 3: Create `backend/tests/conftest.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_supabase():
    return MagicMock()


@pytest.fixture
def mock_llm_response():
    response = MagicMock()
    response.choices[0].message.content = "Test answer."
    return response


@pytest.fixture
def mock_embedding():
    """Returns a 1536-dim zero vector for testing."""
    return [0.0] * 1536
```

- [ ] **Step 4: Verify config loads (manual smoke test)**

```bash
cd backend
cp .env.example .env
# Fill in real SUPABASE_URL and SUPABASE_SERVICE_KEY
python -c "from app.config import settings; print(settings.app_env)"
```

Expected output: `development`

- [ ] **Step 5: Commit**

```bash
cd ~/repos/RentRight
git add backend/app/config.py backend/app/db/client.py backend/tests/conftest.py
git commit -m "feat: config, supabase client, test fixtures

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: LLM Provider

**Files:**
- Create: `backend/app/core/llm_provider.py`
- Create: `backend/tests/test_llm_provider.py`

- [ ] **Step 1: Write failing tests — `backend/tests/test_llm_provider.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from app.core.llm_provider import LLMProvider


def make_mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


@pytest.mark.asyncio
async def test_complete_returns_llm_content():
    provider = LLMProvider(tier="free")
    mock_response = make_mock_response("You have the right to 24 hours notice.")

    with patch("app.core.llm_provider.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response
        result = await provider.complete(
            system="You are a tenant rights assistant.",
            user="Can my landlord enter without notice?",
        )

    assert result == "You have the right to 24 hours notice."


@pytest.mark.asyncio
async def test_complete_falls_back_on_primary_failure():
    provider = LLMProvider(tier="free")
    fallback_response = make_mock_response("Fallback: landlords must give notice.")

    with patch("app.core.llm_provider.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [Exception("Provider down"), fallback_response]
        result = await provider.complete(system="sys", user="question")

    assert result == "Fallback: landlords must give notice."
    assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_complete_raises_if_fallback_also_fails():
    provider = LLMProvider(tier="free")

    with patch("app.core.llm_provider.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("All providers down")
        with pytest.raises(Exception, match="All providers down"):
            await provider.complete(system="sys", user="question")


def test_free_tier_uses_haiku_model():
    provider = LLMProvider(tier="free")
    assert "haiku" in provider.config["model"]


def test_pro_tier_uses_sonnet_model():
    provider = LLMProvider(tier="pro")
    assert "sonnet" in provider.config["model"]


def test_invalid_tier_raises():
    with pytest.raises(KeyError):
        LLMProvider(tier="invalid")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_llm_provider.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.llm_provider'`

- [ ] **Step 3: Create `backend/app/core/llm_provider.py`**

```python
import litellm

from app.config import LLM_CONFIG


class LLMProvider:
    def __init__(self, tier: str = "free") -> None:
        self.config = LLM_CONFIG[tier]  # raises KeyError on invalid tier
        self.fallback_config = LLM_CONFIG["fallback"]

    async def complete(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = await litellm.acompletion(
                model=self.config["model"],
                messages=messages,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
            )
            return str(response.choices[0].message.content)
        except Exception:
            response = await litellm.acompletion(
                model=self.fallback_config["model"],
                messages=messages,
                max_tokens=self.fallback_config["max_tokens"],
                temperature=self.fallback_config["temperature"],
            )
            return str(response.choices[0].message.content)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_llm_provider.py -v
```

Expected: All 6 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
cd ~/repos/RentRight
git add backend/app/core/llm_provider.py backend/tests/test_llm_provider.py
git commit -m "feat: LLM provider — LiteLLM abstraction with fallback

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Embedding Service

**Files:**
- Create: `backend/app/core/embeddings.py`
- Create: `backend/tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests — `backend/tests/test_embeddings.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.embeddings import EmbeddingService


def make_mock_embedding(dims: int = 1536) -> list[float]:
    return [0.1] * dims


def make_embed_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    return response


@pytest.mark.asyncio
async def test_embed_returns_float_list():
    service = EmbeddingService()
    vector = make_mock_embedding()

    with patch("app.core.embeddings.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = make_embed_response([vector])
        result = await service.embed("Can my landlord enter without notice?")

    assert isinstance(result, list)
    assert len(result) == 1536
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_embed_batch_returns_list_of_vectors():
    service = EmbeddingService()
    vectors = [make_mock_embedding(), make_mock_embedding()]

    with patch("app.core.embeddings.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = make_embed_response(vectors)
        result = await service.embed_batch(["question one", "question two"])

    assert len(result) == 2
    assert len(result[0]) == 1536


@pytest.mark.asyncio
async def test_embed_batch_empty_returns_empty():
    service = EmbeddingService()
    with patch("app.core.embeddings.litellm.aembedding", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = make_embed_response([])
        result = await service.embed_batch([])
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_embeddings.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.embeddings'`

- [ ] **Step 3: Create `backend/app/core/embeddings.py`**

```python
import litellm

from app.config import EMBEDDING_CONFIG


class EmbeddingService:
    def __init__(self) -> None:
        self.model = EMBEDDING_CONFIG["model"]

    async def embed(self, text: str) -> list[float]:
        response = await litellm.aembedding(model=self.model, input=[text])
        return list(response.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await litellm.aembedding(model=self.model, input=texts)
        return [list(item.embedding) for item in response.data]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_embeddings.py -v
```

Expected: All 3 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
cd ~/repos/RentRight
git add backend/app/core/embeddings.py backend/tests/test_embeddings.py
git commit -m "feat: embedding service — LiteLLM-backed, swappable model

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Domain Models

**Files:**
- Create: `backend/app/models/domain.py`
- Create: `backend/app/models/requests.py`
- Create: `backend/app/models/responses.py`

- [ ] **Step 1: Create `backend/app/models/domain.py`**

```python
from dataclasses import dataclass, field


@dataclass
class Jurisdiction:
    state: str          # e.g. "California"
    state_abbrev: str   # e.g. "CA"
    city: str = ""      # e.g. "Los Angeles" — empty if no local ordinance applies


@dataclass
class Chunk:
    content: str
    source: str
    state: str | None = None
    city: str | None = None
    similarity: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AnswerResponse:
    answer: str
    sources: list[str]
    jurisdiction: Jurisdiction
    from_cache: bool = False
    disclaimer: str = (
        "This is legal information, not legal advice. "
        "For serious disputes, consult a licensed tenant attorney."
    )
```

- [ ] **Step 2: Create `backend/app/models/requests.py`**

```python
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    state_override: str | None = Field(None, description="Manual state override, e.g. 'California'")
```

- [ ] **Step 3: Create `backend/app/models/responses.py`**

```python
from pydantic import BaseModel


class JurisdictionOut(BaseModel):
    state: str
    state_abbrev: str
    city: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    jurisdiction: JurisdictionOut
    from_cache: bool
    disclaimer: str
```

- [ ] **Step 4: Commit**

```bash
cd ~/repos/RentRight
git add backend/app/models/
git commit -m "feat: domain models — Jurisdiction, Chunk, AnswerResponse, request/response schemas

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Jurisdiction Resolver

**Files:**
- Create: `backend/app/core/jurisdiction.py`
- Create: `backend/tests/test_jurisdiction.py`

- [ ] **Step 1: Write failing tests — `backend/tests/test_jurisdiction.py`**

```python
import pytest
from app.core.jurisdiction import JurisdictionResolver


@pytest.fixture
def resolver():
    return JurisdictionResolver()


def test_resolve_from_coords_returns_state(resolver):
    # San Jose, CA
    result = resolver.resolve_from_coords(37.3382, -121.8863)
    assert result.state == "California"
    assert result.state_abbrev == "CA"


def test_resolve_from_coords_city_with_local_ordinance(resolver):
    # Los Angeles, CA — has rent stabilization ordinance
    result = resolver.resolve_from_coords(34.0522, -118.2437)
    assert result.state == "California"
    assert result.city == "Los Angeles"


def test_resolve_from_coords_city_without_ordinance_returns_empty_city(resolver):
    # Fresno, CA — no local rent control
    result = resolver.resolve_from_coords(36.7378, -119.7871)
    assert result.state == "California"
    assert result.city == ""


def test_resolve_from_state_name(resolver):
    result = resolver.resolve_from_state("California")
    assert result.state == "California"
    assert result.state_abbrev == "CA"
    assert result.city == ""


def test_resolve_from_state_abbrev(resolver):
    result = resolver.resolve_from_state("TX")
    assert result.state == "Texas"
    assert result.state_abbrev == "TX"


def test_resolve_from_unknown_state_raises(resolver):
    with pytest.raises(ValueError, match="Unknown state"):
        resolver.resolve_from_state("Narnia")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_jurisdiction.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.jurisdiction'`

- [ ] **Step 3: Create `backend/app/core/jurisdiction.py`**

```python
import reverse_geocoder as rg  # type: ignore[import-untyped]

from app.models.domain import Jurisdiction

# Cities that have local tenant protection ordinances beyond state law
CITIES_WITH_LOCAL_ORDINANCES: set[str] = {
    "Los Angeles", "San Francisco", "Oakland", "San Jose",
    "New York City", "Seattle", "Chicago", "Portland",
    "Denver", "Washington", "Boston", "Minneapolis",
}

STATE_NAME_TO_ABBREV: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}

ABBREV_TO_STATE_NAME: dict[str, str] = {v: k for k, v in STATE_NAME_TO_ABBREV.items()}


class JurisdictionResolver:
    def resolve_from_coords(self, lat: float, lon: float) -> Jurisdiction:
        results = rg.search((lat, lon), verbose=False)
        if not results:
            raise ValueError(f"Could not resolve coordinates ({lat}, {lon})")

        place = results[0]
        state_name = place.get("admin1", "")
        city_name = place.get("name", "")
        state_abbrev = STATE_NAME_TO_ABBREV.get(state_name, "")

        city = city_name if city_name in CITIES_WITH_LOCAL_ORDINANCES else ""

        return Jurisdiction(
            state=state_name,
            state_abbrev=state_abbrev,
            city=city,
        )

    def resolve_from_state(self, state: str) -> Jurisdiction:
        # Try as full name first
        if state in STATE_NAME_TO_ABBREV:
            return Jurisdiction(state=state, state_abbrev=STATE_NAME_TO_ABBREV[state])
        # Try as abbreviation
        if state.upper() in ABBREV_TO_STATE_NAME:
            abbrev = state.upper()
            return Jurisdiction(state=ABBREV_TO_STATE_NAME[abbrev], state_abbrev=abbrev)
        raise ValueError(f"Unknown state: {state!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_jurisdiction.py -v
```

Expected: All 6 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
cd ~/repos/RentRight
git add backend/app/core/jurisdiction.py backend/tests/test_jurisdiction.py
git commit -m "feat: jurisdiction resolver — GPS coords + state name to Jurisdiction

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Semantic Cache

**Files:**
- Create: `backend/app/core/cache.py`
- Create: `backend/tests/test_cache.py`

- [ ] **Step 1: Write failing tests — `backend/tests/test_cache.py`**

```python
import pytest
from unittest.mock import MagicMock, patch

from app.core.cache import SemanticCacheService
from app.models.domain import Jurisdiction


@pytest.fixture
def jurisdiction():
    return Jurisdiction(state="California", state_abbrev="CA", city="Los Angeles")


@pytest.fixture
def cache(mock_supabase):
    return SemanticCacheService(db=mock_supabase)


@pytest.fixture
def sample_vector():
    return [0.1] * 1536


def test_get_returns_cached_answer_on_hit(cache, jurisdiction, sample_vector):
    mock_result = MagicMock()
    mock_result.data = [{"answer": "Landlord must give 24 hours notice.", "id": 1}]
    cache.db.rpc.return_value.execute.return_value = mock_result

    result = cache.get(question_vector=sample_vector, jurisdiction=jurisdiction)

    assert result == "Landlord must give 24 hours notice."
    cache.db.rpc.assert_called_once_with(
        "match_qa_cache",
        {
            "query_embedding": sample_vector,
            "filter_state": "California",
            "filter_city": "Los Angeles",
            "match_threshold": pytest.approx(0.90),
        },
    )


def test_get_returns_none_on_cache_miss(cache, jurisdiction, sample_vector):
    mock_result = MagicMock()
    mock_result.data = []
    cache.db.rpc.return_value.execute.return_value = mock_result

    result = cache.get(question_vector=sample_vector, jurisdiction=jurisdiction)

    assert result is None


def test_set_inserts_to_supabase(cache, jurisdiction, sample_vector):
    cache.db.table.return_value.insert.return_value.execute.return_value = MagicMock()

    cache.set(
        question_vector=sample_vector,
        question="Can my landlord enter without notice?",
        answer="No. California requires 24 hours notice.",
        jurisdiction=jurisdiction,
    )

    cache.db.table.assert_called_with("qa_cache")
    insert_call = cache.db.table.return_value.insert.call_args[0][0]
    assert insert_call["state"] == "California"
    assert insert_call["city"] == "Los Angeles"
    assert insert_call["answer"] == "No. California requires 24 hours notice."
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_cache.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.cache'`

- [ ] **Step 3: Create `backend/app/core/cache.py`**

```python
from supabase import Client

from app.config import CACHE_SIMILARITY_THRESHOLD
from app.models.domain import Jurisdiction


class SemanticCacheService:
    def __init__(self, db: Client) -> None:
        self.db = db

    def get(self, question_vector: list[float], jurisdiction: Jurisdiction) -> str | None:
        result = self.db.rpc(
            "match_qa_cache",
            {
                "query_embedding": question_vector,
                "filter_state": jurisdiction.state,
                "filter_city": jurisdiction.city,
                "match_threshold": CACHE_SIMILARITY_THRESHOLD,
            },
        ).execute()

        if result.data:
            # Increment hit count asynchronously (best-effort, don't block)
            try:
                self.db.table("qa_cache").update({"hit_count": result.data[0]["id"]}).eq(
                    "id", result.data[0]["id"]
                ).execute()
            except Exception:
                pass
            return str(result.data[0]["answer"])

        return None

    def set(
        self,
        question_vector: list[float],
        question: str,
        answer: str,
        jurisdiction: Jurisdiction,
    ) -> None:
        self.db.table("qa_cache").insert(
            {
                "state": jurisdiction.state,
                "city": jurisdiction.city,
                "question": question,
                "answer": answer,
                "embedding": question_vector,
            }
        ).execute()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_cache.py -v
```

Expected: All 3 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
cd ~/repos/RentRight
git add backend/app/core/cache.py backend/tests/test_cache.py
git commit -m "feat: semantic cache — pgvector-backed, jurisdiction-scoped Q&A cache

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 9: RAG Retriever

**Files:**
- Create: `backend/app/rag/retriever.py`
- Create: `backend/tests/test_retriever.py`

- [ ] **Step 1: Write failing tests — `backend/tests/test_retriever.py`**

```python
import pytest
from unittest.mock import MagicMock

from app.rag.retriever import RAGRetriever
from app.models.domain import Jurisdiction


@pytest.fixture
def jurisdiction():
    return Jurisdiction(state="California", state_abbrev="CA", city="")


@pytest.fixture
def retriever(mock_supabase):
    return RAGRetriever(db=mock_supabase)


@pytest.fixture
def sample_vector():
    return [0.1] * 1536


def test_retrieve_public_law_returns_chunks(retriever, jurisdiction, sample_vector):
    mock_result = MagicMock()
    mock_result.data = [
        {
            "content": "Landlord must give 24 hours notice before entry.",
            "source": "ca_tenant_law",
            "state": "California",
            "city": None,
            "similarity": 0.93,
            "metadata": {"section": "CA Civil Code §1954"},
        }
    ]
    retriever.db.rpc.return_value.execute.return_value = mock_result

    chunks = retriever.retrieve_public_law(
        question_vector=sample_vector, jurisdiction=jurisdiction
    )

    assert len(chunks) == 1
    assert chunks[0].content == "Landlord must give 24 hours notice before entry."
    assert chunks[0].source == "ca_tenant_law"
    assert chunks[0].similarity == 0.93


def test_retrieve_public_law_returns_empty_on_no_results(retriever, jurisdiction, sample_vector):
    mock_result = MagicMock()
    mock_result.data = []
    retriever.db.rpc.return_value.execute.return_value = mock_result

    chunks = retriever.retrieve_public_law(
        question_vector=sample_vector, jurisdiction=jurisdiction
    )
    assert chunks == []


def test_retrieve_user_lease_returns_chunks(retriever, sample_vector):
    user_id = "user-123"
    mock_result = MagicMock()
    mock_result.data = [
        {
            "content": "Tenant shall provide 30 days written notice.",
            "lease_id": "lease-abc",
            "similarity": 0.88,
            "metadata": {"clause": "Section 12"},
        }
    ]
    retriever.db.rpc.return_value.execute.return_value = mock_result

    chunks = retriever.retrieve_user_lease(
        question_vector=sample_vector, user_id=user_id
    )

    assert len(chunks) == 1
    assert chunks[0].source == "your_lease"


def test_retrieve_user_lease_returns_empty_when_no_lease(retriever, sample_vector):
    mock_result = MagicMock()
    mock_result.data = []
    retriever.db.rpc.return_value.execute.return_value = mock_result

    chunks = retriever.retrieve_user_lease(question_vector=sample_vector, user_id="user-no-lease")
    assert chunks == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_retriever.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.rag.retriever'`

- [ ] **Step 3: Create `backend/app/rag/retriever.py`**

```python
from supabase import Client

from app.config import (
    RETRIEVAL_SIMILARITY_THRESHOLD,
    RETRIEVAL_TOP_K_LEASE,
    RETRIEVAL_TOP_K_PUBLIC,
)
from app.models.domain import Chunk, Jurisdiction


class RAGRetriever:
    def __init__(self, db: Client) -> None:
        self.db = db

    def retrieve_public_law(
        self,
        question_vector: list[float],
        jurisdiction: Jurisdiction,
        k: int = RETRIEVAL_TOP_K_PUBLIC,
    ) -> list[Chunk]:
        result = self.db.rpc(
            "match_public_law",
            {
                "query_embedding": question_vector,
                "filter_state": jurisdiction.state,
                "match_threshold": RETRIEVAL_SIMILARITY_THRESHOLD,
                "match_count": k,
            },
        ).execute()

        return [
            Chunk(
                content=row["content"],
                source=row["source"],
                state=row.get("state"),
                city=row.get("city"),
                similarity=row.get("similarity", 0.0),
                metadata=row.get("metadata", {}),
            )
            for row in (result.data or [])
        ]

    def retrieve_user_lease(
        self,
        question_vector: list[float],
        user_id: str,
        k: int = RETRIEVAL_TOP_K_LEASE,
    ) -> list[Chunk]:
        result = self.db.rpc(
            "match_user_lease",
            {
                "query_embedding": question_vector,
                "filter_user_id": user_id,
                "match_threshold": RETRIEVAL_SIMILARITY_THRESHOLD,
                "match_count": k,
            },
        ).execute()

        return [
            Chunk(
                content=row["content"],
                source="your_lease",
                similarity=row.get("similarity", 0.0),
                metadata=row.get("metadata", {}),
            )
            for row in (result.data or [])
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_retriever.py -v
```

Expected: All 4 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
cd ~/repos/RentRight
git add backend/app/rag/retriever.py backend/tests/test_retriever.py
git commit -m "feat: RAG retriever — public law + user lease vector search

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 10: RAG Pipeline + Prompts

**Files:**
- Create: `backend/app/rag/prompts.py`
- Create: `backend/app/rag/pipeline.py`
- Create: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Create `backend/app/rag/prompts.py`**

```python
SYSTEM_PROMPT = """You are RentRight, a tenant rights assistant. You help renters understand their legal rights in plain, accessible English.

RULES:
1. Answer based ONLY on the context provided below. Do not use outside knowledge.
2. Always cite your source (e.g. "According to CA Civil Code §1954...").
3. If the user has uploaded a lease, reference their specific lease clauses where relevant.
4. Be direct and clear. Avoid legalese.
5. Structure every answer as:
   - Direct answer (1-2 sentences)
   - What the law says (with citation)
   - What their lease says (if lease context is provided)
   - Recommended next step
6. Never give a definitive legal conclusion — always note that a tenant attorney can give advice specific to their full situation.

JURISDICTION: {state}{city_clause}

CONTEXT FROM LAW:
{law_context}

{lease_section}"""

LEASE_SECTION_TEMPLATE = """CONTEXT FROM YOUR LEASE:
{lease_context}"""

NO_CONTEXT_ANSWER = (
    "I wasn't able to find specific legal information about this in my current database for your state. "
    "I recommend contacting your local tenant rights organization or a tenant attorney for guidance on this question."
)
```

- [ ] **Step 2: Write failing tests — `backend/tests/test_pipeline.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.rag.pipeline import RAGPipeline
from app.models.domain import Chunk, Jurisdiction


@pytest.fixture
def jurisdiction():
    return Jurisdiction(state="California", state_abbrev="CA", city="Los Angeles")


@pytest.fixture
def sample_vector():
    return [0.1] * 1536


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            content="Landlord must give 24 hours notice before entry.",
            source="ca_tenant_law",
            state="California",
            similarity=0.93,
            metadata={"section": "CA Civil Code §1954"},
        )
    ]


@pytest.fixture
def pipeline(mock_supabase):
    mock_embeddings = MagicMock()
    mock_embeddings.embed = AsyncMock(return_value=[0.1] * 1536)
    mock_retriever = MagicMock()
    mock_cache = MagicMock()
    mock_llm_free = MagicMock()
    mock_llm_free.complete = AsyncMock(return_value="Your landlord must give 24 hours notice.")
    mock_llm_pro = MagicMock()
    mock_llm_pro.complete = AsyncMock(return_value="Pro answer.")

    return RAGPipeline(
        embeddings=mock_embeddings,
        retriever=mock_retriever,
        cache=mock_cache,
        llm_free=mock_llm_free,
        llm_pro=mock_llm_pro,
    )


@pytest.mark.asyncio
async def test_answer_returns_cached_response(pipeline, jurisdiction, sample_vector):
    pipeline.cache.get.return_value = "Cached: landlord must give 24 hours notice."
    pipeline.embeddings.embed.return_value = sample_vector

    result = await pipeline.answer(
        question="Can my landlord enter without notice?",
        jurisdiction=jurisdiction,
        user_id="user-123",
        tier="free",
    )

    assert result.from_cache is True
    assert "24 hours" in result.answer
    pipeline.retriever.retrieve_public_law.assert_not_called()


@pytest.mark.asyncio
async def test_answer_calls_llm_on_cache_miss(pipeline, jurisdiction, sample_chunks):
    pipeline.cache.get.return_value = None
    pipeline.retriever.retrieve_public_law.return_value = sample_chunks
    pipeline.retriever.retrieve_user_lease.return_value = []

    result = await pipeline.answer(
        question="Can my landlord enter without notice?",
        jurisdiction=jurisdiction,
        user_id="user-123",
        tier="free",
    )

    assert result.from_cache is False
    assert result.answer == "Your landlord must give 24 hours notice."
    pipeline.llm_free.complete.assert_called_once()
    pipeline.cache.set.assert_called_once()


@pytest.mark.asyncio
async def test_answer_uses_pro_llm_for_pro_tier(pipeline, jurisdiction, sample_chunks):
    pipeline.cache.get.return_value = None
    pipeline.retriever.retrieve_public_law.return_value = sample_chunks
    pipeline.retriever.retrieve_user_lease.return_value = []

    result = await pipeline.answer(
        question="Can my landlord enter without notice?",
        jurisdiction=jurisdiction,
        user_id="user-pro",
        tier="pro",
    )

    pipeline.llm_pro.complete.assert_called_once()
    pipeline.llm_free.complete.assert_not_called()


@pytest.mark.asyncio
async def test_answer_includes_disclaimer(pipeline, jurisdiction, sample_chunks):
    pipeline.cache.get.return_value = None
    pipeline.retriever.retrieve_public_law.return_value = sample_chunks
    pipeline.retriever.retrieve_user_lease.return_value = []

    result = await pipeline.answer(
        question="Can my landlord enter without notice?",
        jurisdiction=jurisdiction,
        user_id="user-123",
        tier="free",
    )

    assert "not legal advice" in result.disclaimer
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.rag.pipeline'`

- [ ] **Step 4: Create `backend/app/rag/pipeline.py`**

```python
from app.core.cache import SemanticCacheService
from app.core.embeddings import EmbeddingService
from app.core.llm_provider import LLMProvider
from app.models.domain import AnswerResponse, Jurisdiction
from app.rag.prompts import (
    LEASE_SECTION_TEMPLATE,
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
)
from app.rag.retriever import RAGRetriever


class RAGPipeline:
    def __init__(
        self,
        embeddings: EmbeddingService,
        retriever: RAGRetriever,
        cache: SemanticCacheService,
        llm_free: LLMProvider,
        llm_pro: LLMProvider,
    ) -> None:
        self.embeddings = embeddings
        self.retriever = retriever
        self.cache = cache
        self.llm_free = llm_free
        self.llm_pro = llm_pro

    async def answer(
        self,
        question: str,
        jurisdiction: Jurisdiction,
        user_id: str,
        tier: str,
    ) -> AnswerResponse:
        question_vector = await self.embeddings.embed(question)

        # 1. Check semantic cache
        cached = self.cache.get(question_vector=question_vector, jurisdiction=jurisdiction)
        if cached:
            return AnswerResponse(
                answer=cached,
                sources=["cache"],
                jurisdiction=jurisdiction,
                from_cache=True,
            )

        # 2. Retrieve context
        law_chunks = self.retriever.retrieve_public_law(
            question_vector=question_vector, jurisdiction=jurisdiction
        )
        lease_chunks = self.retriever.retrieve_user_lease(
            question_vector=question_vector, user_id=user_id
        )

        # 3. Return early if no context found
        if not law_chunks and not lease_chunks:
            return AnswerResponse(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                jurisdiction=jurisdiction,
                from_cache=False,
            )

        # 4. Build prompt
        law_context = "\n\n".join(
            f"[{c.source}] {c.content}" for c in law_chunks
        )
        lease_section = (
            LEASE_SECTION_TEMPLATE.format(
                lease_context="\n\n".join(c.content for c in lease_chunks)
            )
            if lease_chunks
            else ""
        )
        city_clause = f" / {jurisdiction.city}" if jurisdiction.city else ""
        system = SYSTEM_PROMPT.format(
            state=jurisdiction.state,
            city_clause=city_clause,
            law_context=law_context,
            lease_section=lease_section,
        )

        # 5. Call LLM
        llm = self.llm_pro if tier == "pro" else self.llm_free
        answer_text = await llm.complete(system=system, user=question)

        # 6. Store in cache
        self.cache.set(
            question_vector=question_vector,
            question=question,
            answer=answer_text,
            jurisdiction=jurisdiction,
        )

        sources = list({c.source for c in law_chunks})
        if lease_chunks:
            sources.append("your_lease")

        return AnswerResponse(
            answer=answer_text,
            sources=sources,
            jurisdiction=jurisdiction,
            from_cache=False,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_pipeline.py -v
```

Expected: All 4 tests `PASSED`

- [ ] **Step 6: Commit**

```bash
cd ~/repos/RentRight
git add backend/app/rag/ backend/tests/test_pipeline.py
git commit -m "feat: RAG pipeline — cache → retrieve → prompt → LLM → answer flow

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 11: FastAPI App + Health Endpoint

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/api/v1/health.py`
- Create: `backend/app/dependencies.py`

- [ ] **Step 1: Create `backend/app/api/v1/health.py`**

```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 2: Create `backend/app/dependencies.py`**

```python
from functools import lru_cache

from supabase import Client

from app.core.cache import SemanticCacheService
from app.core.embeddings import EmbeddingService
from app.core.jurisdiction import JurisdictionResolver
from app.core.llm_provider import LLMProvider
from app.db.client import get_supabase
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RAGRetriever


def get_db() -> Client:
    return get_supabase()


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    db = get_supabase()
    return RAGPipeline(
        embeddings=EmbeddingService(),
        retriever=RAGRetriever(db=db),
        cache=SemanticCacheService(db=db),
        llm_free=LLMProvider(tier="free"),
        llm_pro=LLMProvider(tier="pro"),
    )


@lru_cache(maxsize=1)
def get_jurisdiction_resolver() -> JurisdictionResolver:
    return JurisdictionResolver()
```

- [ ] **Step 3: Create `backend/app/main.py`**

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.health import router as health_router

app = FastAPI(
    title="RentRight API",
    version="0.1.0",
    description="Tenant rights RAG API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten before production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health_router, prefix="/api/v1")
```

- [ ] **Step 4: Write and run health endpoint test**

```bash
cat > backend/tests/test_api_health.py << 'EOF'
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
EOF
```

```bash
cd backend
pytest tests/test_api_health.py -v
```

Expected: `PASSED`

- [ ] **Step 5: Smoke test server starts**

```bash
cd backend
uvicorn app.main:app --port 8000 &
sleep 2
curl http://localhost:8000/api/v1/health
kill %1
```

Expected: `{"status":"ok"}`

- [ ] **Step 6: Commit**

```bash
cd ~/repos/RentRight
git add backend/app/main.py backend/app/dependencies.py backend/app/api/v1/health.py backend/tests/test_api_health.py
git commit -m "feat: FastAPI app — CORS, error handlers, health endpoint

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 12: Know Your Rights Endpoint

**Files:**
- Create: `backend/app/api/v1/qa.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_api_qa.py`

- [ ] **Step 1: Write failing tests — `backend/tests/test_api_qa.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.domain import AnswerResponse, Jurisdiction


client = TestClient(app)

MOCK_ANSWER = AnswerResponse(
    answer="Your landlord must give 24 hours notice before entering.",
    sources=["ca_tenant_law"],
    jurisdiction=Jurisdiction(state="California", state_abbrev="CA", city=""),
    from_cache=False,
)

MOCK_JURISDICTION = Jurisdiction(state="California", state_abbrev="CA", city="")


def test_ask_with_state_override_returns_answer():
    with (
        patch("app.api.v1.qa.get_pipeline") as mock_get_pipeline,
        patch("app.api.v1.qa.get_jurisdiction_resolver") as mock_get_resolver,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.answer = AsyncMock(return_value=MOCK_ANSWER)
        mock_get_pipeline.return_value = mock_pipeline

        mock_resolver = MagicMock()
        mock_resolver.resolve_from_state.return_value = MOCK_JURISDICTION
        mock_get_resolver.return_value = mock_resolver

        response = client.post(
            "/api/v1/qa/ask",
            json={
                "question": "Can my landlord enter without notice?",
                "state_override": "California",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Your landlord must give 24 hours notice before entering."
    assert data["jurisdiction"]["state"] == "California"
    assert "disclaimer" in data


def test_ask_with_coords_resolves_jurisdiction():
    with (
        patch("app.api.v1.qa.get_pipeline") as mock_get_pipeline,
        patch("app.api.v1.qa.get_jurisdiction_resolver") as mock_get_resolver,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.answer = AsyncMock(return_value=MOCK_ANSWER)
        mock_get_pipeline.return_value = mock_pipeline

        mock_resolver = MagicMock()
        mock_resolver.resolve_from_coords.return_value = MOCK_JURISDICTION
        mock_get_resolver.return_value = mock_resolver

        response = client.post(
            "/api/v1/qa/ask",
            json={
                "question": "Can my landlord enter without notice?",
                "latitude": 37.33,
                "longitude": -121.88,
            },
        )

    assert response.status_code == 200
    mock_resolver.resolve_from_coords.assert_called_once_with(37.33, -121.88)


def test_ask_without_location_returns_400():
    response = client.post(
        "/api/v1/qa/ask",
        json={"question": "Can my landlord enter without notice?"},
    )
    assert response.status_code == 400
    assert "location" in response.json()["detail"].lower()


def test_ask_question_too_short_returns_422():
    response = client.post(
        "/api/v1/qa/ask",
        json={"question": "Hi", "state_override": "California"},
    )
    assert response.status_code == 422


def test_ask_returns_from_cache_flag():
    cached_answer = AnswerResponse(
        answer="Cached answer.",
        sources=["cache"],
        jurisdiction=MOCK_JURISDICTION,
        from_cache=True,
    )
    with (
        patch("app.api.v1.qa.get_pipeline") as mock_get_pipeline,
        patch("app.api.v1.qa.get_jurisdiction_resolver") as mock_get_resolver,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.answer = AsyncMock(return_value=cached_answer)
        mock_get_pipeline.return_value = mock_pipeline

        mock_resolver = MagicMock()
        mock_resolver.resolve_from_state.return_value = MOCK_JURISDICTION
        mock_get_resolver.return_value = mock_resolver

        response = client.post(
            "/api/v1/qa/ask",
            json={"question": "Can my landlord enter?", "state_override": "California"},
        )

    assert response.json()["from_cache"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_api_qa.py -v
```

Expected: All 5 tests fail with `404` (route not registered yet)

- [ ] **Step 3: Create `backend/app/api/v1/qa.py`**

```python
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_jurisdiction_resolver, get_pipeline
from app.models.requests import AskRequest
from app.models.responses import AskResponse, JurisdictionOut
from app.rag.pipeline import RAGPipeline
from app.core.jurisdiction import JurisdictionResolver

router = APIRouter()

# Hardcoded anonymous user ID for unauthenticated requests (Plan 1 scope)
# Auth middleware (user tiers) added in Plan 3
ANON_USER_ID = "anonymous"


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    resolver: JurisdictionResolver = Depends(get_jurisdiction_resolver),
) -> AskResponse:
    # Resolve jurisdiction
    if request.state_override:
        try:
            jurisdiction = resolver.resolve_from_state(request.state_override)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif request.latitude is not None and request.longitude is not None:
        try:
            jurisdiction = resolver.resolve_from_coords(request.latitude, request.longitude)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail="Location is required. Provide latitude/longitude or state_override.",
        )

    result = await pipeline.answer(
        question=request.question,
        jurisdiction=jurisdiction,
        user_id=ANON_USER_ID,
        tier="free",
    )

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        jurisdiction=JurisdictionOut(
            state=result.jurisdiction.state,
            state_abbrev=result.jurisdiction.state_abbrev,
            city=result.jurisdiction.city,
        ),
        from_cache=result.from_cache,
        disclaimer=result.disclaimer,
    )
```

- [ ] **Step 4: Register the router in `backend/app/main.py`**

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.health import router as health_router
from app.api.v1.qa import router as qa_router          # add this

app = FastAPI(
    title="RentRight API",
    version="0.1.0",
    description="Tenant rights RAG API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health_router, prefix="/api/v1")
app.include_router(qa_router, prefix="/api/v1/qa")    # add this
```

- [ ] **Step 5: Run all tests**

```bash
cd backend
pytest -v
```

Expected: All tests `PASSED`. You should see green for all 20+ tests across all modules.

- [ ] **Step 6: Run lint and type check**

```bash
cd backend
make lint
make typecheck
```

Expected: No errors.

- [ ] **Step 7: Final commit**

```bash
cd ~/repos/RentRight
git add backend/
git commit -m "feat: Know Your Rights endpoint — full RAG pipeline wired to POST /api/v1/qa/ask

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin main
```

---

## Verification

After completing all tasks, run the full test suite and verify the server responds correctly:

```bash
cd backend

# Full test suite
pytest -v --tb=short

# Start server
uvicorn app.main:app --port 8000

# In another terminal — test the endpoint
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can my landlord enter without notice?", "state_override": "California"}'
```

Expected response shape:
```json
{
  "answer": "...",
  "sources": ["ca_tenant_law"],
  "jurisdiction": { "state": "California", "state_abbrev": "CA", "city": "" },
  "from_cache": false,
  "disclaimer": "This is legal information, not legal advice..."
}
```

---

## What's Next

**Plan 2 — Data Ingestion Pipeline:**
- Ingest HUD guidelines, US Code, state tenant laws (all 50 states), and court opinions into `public_law_chunks`
- Cache warming script: top 100 questions × 50 states → pre-populate `qa_cache`

**Plan 3 — Lease Analyzer:**
- PDF upload endpoint
- Text extraction + chunking
- Automated red flag analysis
- Auth middleware + user tier enforcement

**Plan 4 — Dispute Helper**

**Plan 5 — Mobile App (React Native + Expo)**

**Plan 6 — Path B Recurring Features**

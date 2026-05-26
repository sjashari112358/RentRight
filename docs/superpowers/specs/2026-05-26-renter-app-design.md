# RentRight — Design Spec
**Date:** 2026-05-26  
**Status:** Approved for Implementation Planning  
**Version:** 1.0

---

## 1. Product Vision

RentRight is a mobile app (iOS + Android) that acts as a **Renter OS** — protecting tenants before they sign a lease, while they live in a property, and if anything goes wrong. It uses RAG (Retrieval-Augmented Generation) over public legal datasets to give renters plain-English, jurisdiction-accurate answers grounded in real law.

> *"For less than $9 a month — less than one hour of a tenant attorney — RentRight is the only app that protects you before you sign, while you live there, and if anything goes wrong."*

---

## 2. Target User

**Primary:** Renters and tenants in the United States  
**Market size:** 44 million renter households in the U.S.  
**Pain:** Most renters have zero practical knowledge of their legal rights, no record-keeping system, and no affordable access to legal guidance.

---

## 3. Core Features

### Three Pillars

#### Pillar 1 — Know Your Rights (Q&A)
- Unlimited plain-English Q&A about tenant law
- Every answer grounded in jurisdiction-specific law (state + city)
- Answers structured as:
  1. Direct answer in plain English
  2. Legal source cited (e.g. "CA Civil Code §1954")
  3. How it applies to the user's uploaded lease (if available)
  4. Recommended next step
  5. Disclaimer: *"This is legal information, not legal advice. For serious disputes, consult a licensed tenant attorney."*
- Free tier uses Haiku model + jurisdiction-aware cache (see Section 7)
- Pro tier uses Sonnet model for richer, more nuanced answers

#### Pillar 2 — Lease Analyzer
- User uploads lease PDF
- Backend extracts and chunks text (by clause/paragraph)
- Stored in user's private namespace in vector DB
- Claude performs full-pass analysis flagging:
  - 🔴 **Red Flags** — illegal clauses, waived rights, illegal entry terms, retaliation clauses
  - 🟡 **Watch Out** — above-average fees, vague language, auto-renewals, unusual restrictions
  - 🟢 **Standard** — normal terms, fair deposits, typical notice periods
- Summary screen: *"3 Red Flags · 2 Watch Out · Rest looks standard"*
- Tap any flag → see the clause + plain-English explanation + what your state's law says about it
- Free tier: shows flag count only ("3 red flags found — upgrade to see them")
- Pro tier: full clause-by-clause breakdown

#### Pillar 3 — Dispute Helper
- User selects their situation from a guided menu:
  - Landlord won't fix something
  - Deposit not returned
  - Eviction notice received
  - Illegal entry
  - Harassment / retaliation
  - Rent increase dispute
- App asks 3–5 targeted follow-up questions
- RAGs across: user's lease + state law + court opinions + HUD/CFPB data
- Delivers a **Situation Report**:
  - Where you stand legally
  - What your lease says
  - Your options (ranked by strength)
  - Clear next step (demand letter, housing authority, attorney referral)
- Pro tier only

---

### Path B — Recurring Value Features (Subscription Justification)

#### 🔔 Lease Renewal Alerts
- Reads lease end date from uploaded document
- Schedules push notifications at 90 / 60 / 30 / 14 days before expiry
- Each alert includes a contextual tip (e.g. *"60 days out: best time to negotiate rent"*)

#### 📸 Move-In / Move-Out Inspector
- Guided photo checklist on move-in (timestamped, GPS-tagged, cloud-stored)
- Identical checklist on move-out for comparison
- Creates a legal record defending against wrongful deposit deductions
- Exportable as PDF

#### 🧾 Rent Payment Tracker
- Log every payment: amount, date, method
- One-tap export of full payment history as PDF
- Proof of payment if landlord claims missed rent

#### 🔧 Maintenance Request Log
- Log every repair request with date, description, landlord response
- Auto-surfaces: *"Your landlord hasn't responded in 14 days — here's what that means legally in [state]"*
- Becomes legal evidence of habitability violations if ignored
- Exportable as PDF

#### 📊 Am I Paying Too Much?
- Monthly rent benchmarks for user's exact neighborhood
- Data: HUD Fair Market Rent data + Census ACS
- Output: *"You're paying 23% above median for a 1BR in your zip code"*
- Free tier: state-level only. Pro tier: neighborhood-level

#### 📬 Know Before You Renew
- Triggers 60 days before lease end
- AI analysis combining: local rent trends, payment history, unresolved maintenance issues, current rent vs. market rate
- Clear recommendation: renew, negotiate, or move — with supporting reasoning

---

## 4. Monetization

```
FREE TIER                              PRO TIER
                                       $8.99/month or $69.99/year
─────────────────────────────────────  ──────────────────────────────────────
✅ Unlimited Q&A (Haiku + cache)       ✅ Everything in Free
✅ Basic red flag count only           ✅ Full lease analysis (all flags)
   ("3 red flags found")               ✅ Dispute Helper (full access)
✅ Rent benchmark (state level)        ✅ Lease Renewal Alerts
                                       ✅ Move-In / Move-Out Inspector
                                       ✅ Rent Payment Tracker
                                       ✅ Maintenance Request Log
                                       ✅ Neighborhood rent benchmark
                                       ✅ Know Before You Renew
                                       ✅ Claude Sonnet (richer answers)
                                       ✅ Multi-lease storage
```

**Payment provider:** RevenueCat (handles iOS + Android billing, subscriptions, receipts)

**Revenue model:**
```
100,000 downloads → 8% Pro conversion → 8,000 users
8,000 × $8.99/mo = ~$71,920 MRR = ~$863,040 ARR
```

---

## 5. Architecture

```
┌─────────────────────────────────────────────────┐
│                  MOBILE APP                      │
│         (React Native + Expo — iOS + Android)    │
│                                                  │
│  [Chat Q&A]  [Lease Upload]  [Dispute Helper]   │
│  [Inspector] [Rent Tracker]  [Maintenance Log]  │
└──────────────────────┬──────────────────────────┘
                       │ HTTPS API calls
┌──────────────────────▼──────────────────────────┐
│                 BACKEND API                      │
│              (Python + FastAPI)                  │
│                                                  │
│  • Auth & user sessions (Supabase Auth)          │
│  • Location → jurisdiction resolver              │
│  • Lease PDF processing (extract + chunk)        │
│  • Jurisdiction-aware cache lookup               │
│  • LLM orchestration via LiteLLM abstraction     │
│  • Query routing (public law vs. user's lease)   │
└─────────┬──────────────────────────┬────────────┘
          │                          │
┌─────────▼──────────┐   ┌──────────▼─────────────┐
│   PUBLIC LAW DB    │   │   USER LEASE DB         │
│  (pgvector)        │   │  (pgvector, per user)   │
│                    │   │                         │
│ • HUD rules        │   │ • Uploaded lease text   │
│ • U.S. Code        │   │ • Chunked + embedded    │
│ • State tenant law │   │ • Namespaced by user_id │
│ • Court opinions   │   │ • Lease metadata        │
│ • CFPB housing     │   │   (end date, address)   │
└────────────────────┘   └─────────────────────────┘

┌─────────────────────────────────────────────────┐
│           JURISDICTION-AWARE CACHE              │
│  (Redis or Supabase table)                      │
│                                                  │
│  Key: question_embedding + state + city         │
│  Pre-warmed: top 100 questions × 50 states      │
│  Auto-populates on cache misses                 │
└─────────────────────────────────────────────────┘
```

---

## 6. Data Sources

| Source | Content | License | Ingestion |
|--------|---------|---------|-----------|
| HUD Guidelines | Fair housing rules, habitability standards | Public Domain | Bulk download + chunk |
| U.S. Code (Title 42) | Fair Housing Act, federal tenant protections | Public Domain | api.law.cornell.edu |
| State Tenant Laws | All 50 states — notice, deposits, entry rules | Public Domain | CourtListener + state gov sites |
| Court Opinions | Landlord-tenant rulings | Public Domain | Free Law Project bulk data |
| CFPB Complaints | Housing-related complaints | Public Domain | CFPB public API |
| HUD Fair Market Rents | Neighborhood rent benchmarks | Public Domain | HUD API |
| Census ACS | Neighborhood demographics + housing data | Public Domain | Census API |
| User's Lease | Their specific uploaded document | User-owned | PDF upload, private namespace |

---

## 7. LLM Abstraction Layer

All LLM calls route through a single abstraction using **LiteLLM**. No feature calls any provider SDK directly. Swapping providers requires changing one config value.

### Config

```python
LLM_CONFIG = {
    "free_tier": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 500,
        "temperature": 0.1
    },
    "pro_tier": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1500,
        "temperature": 0.1
    },
    "fallback": {
        "model": "groq/llama-3.3-70b",
        "max_tokens": 500,
        "temperature": 0.1
    }
}

EMBEDDING_CONFIG = {
    "model": "text-embedding-3-small"
}
```

### Provider Class

```python
class LLMProvider:
    def __init__(self, tier: str = "free"):
        self.config = LLM_CONFIG[tier]

    async def complete(self, system: str, user: str) -> str:
        try:
            response = await litellm.acompletion(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"]
            )
            return response.choices[0].message.content
        except Exception:
            # Auto-fallback to backup provider
            fallback_config = LLM_CONFIG["fallback"]
            response = await litellm.acompletion(
                model=fallback_config["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                max_tokens=fallback_config["max_tokens"],
                temperature=fallback_config["temperature"]
            )
            return response.choices[0].message.content
```

**Supported swap targets:** Claude (any version), GPT-4o, Gemini Flash/Pro, Groq (Llama), Ollama (local/offline), and 90+ others via LiteLLM.

---

## 8. Jurisdiction Detection

```
App Launch
    ↓
Request location permission
    ↓
Granted?
  YES → Auto-detect state + city → resolve applicable laws
  NO  → "What state do you rent in?" (one-tap list)
    ↓
Always: manual override available in Settings
    ↓
Jurisdiction stored in user profile
    ↓
Every query tagged with: { state, city (if applicable) }
```

**City-level law** applied only for cities with local ordinances beyond state law:
NYC, Los Angeles, San Francisco, Seattle, Chicago, Portland, Denver, and others as laws evolve.

---

## 9. Jurisdiction-Aware Caching

### Cache Key Structure
```python
cache_key = {
    "question_vector": embed(user_question),  # semantic match
    "state":           "CA",
    "city":            "Los Angeles"          # only where city law differs
}
```

### Cache Lookup Flow
```
Query received
    ↓
Embed question → search cache (question_vector + state + city)
    ↓
Similarity > 90%?
  YES → Return cached answer          Cost: $0.00
  NO  →
    Free tier: call Haiku → cache result    Cost: ~$0.0003
    Pro tier:  call Sonnet → cache result   Cost: ~$0.006
```

### Pre-warming Strategy
- Top 100 most common tenant questions × 50 states = 5,000 entries
- Run once before launch: ~$30 total one-time cost
- Covers ~70% of all free tier queries from day one

### Cost Model (at 50,000 free users, 20 queries/month each)
```
Cache hits (~70%):      700,000 queries × $0.00    = $0
Haiku calls (~20%):     200,000 queries × $0.0003  = $60
Sonnet/Pro calls (~10%):100,000 queries × $0.006   = $600
─────────────────────────────────────────────────────────
Total free tier cost:                                ~$660/mo
Break-even Pro users needed: ~74 users
```

---

## 10. Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Mobile | React Native + Expo | Cross-platform iOS + Android, good PDF support |
| Backend | Python + FastAPI | Best AI/ML ecosystem, async, fast to build |
| Database | Supabase (Postgres) | Auth + DB + Storage + pgvector in one |
| Vector Search | pgvector (in Supabase) | No extra service needed at MVP scale |
| LLM Routing | LiteLLM | Swap any provider with one config change |
| LLM (Free) | Claude Haiku | 20x cheaper than Sonnet, sufficient for Q&A |
| LLM (Pro) | Claude Sonnet | Best legal reasoning, large context window |
| LLM (Fallback) | Groq / Llama 3.3 | Fast, cheap, automatic failover |
| Embeddings | text-embedding-3-small | Cheap, accurate, LiteLLM-compatible |
| PDF Processing | PyMuPDF + pdfplumber | Handles real-world messy lease PDFs |
| Cache | Redis (Upstash) | Fast semantic cache lookups, free tier |
| Payments | RevenueCat | Industry standard for mobile subscriptions |
| Hosting | Railway or Render | Simple deploys, scales easily |
| Push Notifications | Expo Notifications | Built into Expo, handles iOS + Android |
| File Storage | Supabase Storage | Secure per-user lease PDF storage |

---

## 11. Running Costs (MVP Scale)

```
Claude Haiku (free tier queries)     ~$10-30/mo
Claude Sonnet (pro tier queries)     ~$20-50/mo
OpenAI Embeddings                    ~$2-5/mo
Supabase                             Free tier (500MB DB, 1GB storage)
Redis / Upstash                      Free tier
Railway / Render hosting             Free → $10/mo at scale
RevenueCat                           Free up to $2,500 MRR

TOTAL MVP COST:                      ~$35-100/mo until real traction
```

---

## 12. Legal Protection (Non-Negotiable)

1. **Disclaimer on every answer:** *"This is legal information, not legal advice. For serious disputes, consult a licensed tenant attorney."*
2. **Terms of Service:** App is informational only, not a law firm, no attorney-client relationship
3. **Answer framing:** Always *"Based on California law..."* never *"You should sue your landlord"*
4. **Data privacy:** User lease documents stored encrypted, never used for training, deletable on request

---

## 13. Engineering Standards

### Reusability
- All UI elements built as isolated, documented components in a shared `components/` library — never one-off inline components
- Business logic lives in custom hooks (`useLeaseAnalyzer`, `useJurisdiction`, `useCache`) — never inside UI components
- LLM calls, RAG retrieval, and cache logic are each their own service class — composable, not entangled
- Backend follows a service/repository pattern: routes → services → repositories → DB. No business logic in route handlers.

### Testability
- **Backend:** Every service class is unit-testable in isolation via dependency injection. LLMProvider, CacheService, and RAGRetriever are injected, not imported directly — swap real implementations for mocks in tests.
- **Frontend:** Components tested with React Native Testing Library. Business logic tested via hook tests (no UI rendering required).
- **Integration tests:** Key RAG pipelines tested end-to-end with fixture leases and fixture legal data.
- **LLM outputs:** Non-deterministic by nature — test prompt structure, context assembly, and citation formatting rather than exact output strings.
- **CI:** All tests run on every PR via GitHub Actions before merge is allowed.

### Accessibility
- All interactive elements meet **WCAG 2.1 AA** standards minimum
- Every touchable has an `accessibilityLabel` and `accessibilityHint`
- Font sizes respect system accessibility settings (no hardcoded px sizes — use `sp` units)
- Color contrast ratios: minimum 4.5:1 for normal text, 3:1 for large text
- Screen reader tested on VoiceOver (iOS) and TalkBack (Android) before each release
- Legal answers never conveyed by color alone — always paired with icons and text
- Tap targets minimum 44×44pt (Apple HIG standard)
- No time-limited interactions — renters in stressful situations need unhurried UX

### Code Quality
- TypeScript strict mode on frontend — no `any` types
- Python type hints throughout backend — mypy clean
- Linting: ESLint + Prettier (frontend), Ruff (backend)
- Pre-commit hooks enforce lint + format before every commit
- All functions documented with clear purpose, inputs, and outputs
- No function longer than 50 lines — split if it grows beyond that

---

## 14. V2 Roadmap (Post-Validation)

- **Dispute letter drafting** — generate demand letters based on situation + law
- **Attorney referral network** — connect to tenant attorneys for serious cases (revenue share)
- **Landlord reputation lookup** — cross-reference landlord name/address against CFPB + court records
- **Multi-city ordinance expansion** — deeper city-level law coverage
- **Web app** — browser version for lease uploads on desktop
- **Document scanner** — camera-based lease capture (no PDF needed)

---

## 14. Success Metrics (MVP)

| Metric | Target (Month 3) |
|--------|-----------------|
| Downloads | 10,000 |
| Free → Pro conversion | 6–8% |
| Pro MRR | $5,000+ |
| D30 retention (free) | >25% |
| D30 retention (pro) | >60% |
| Avg questions per session | >3 |
| Cache hit rate | >65% |

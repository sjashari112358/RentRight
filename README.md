# RentRight 🏠

> The Renter OS — protect tenants before they sign, while they live there, and if anything goes wrong.

RentRight is a mobile app (iOS + Android) that gives renters plain-English, jurisdiction-accurate answers grounded in real law — powered by RAG over public legal datasets.

## Features

- **Know Your Rights** — Unlimited Q&A backed by state + federal tenant law
- **Lease Analyzer** — Upload your lease, get plain-English red flag analysis
- **Dispute Helper** — Active conflict? Get a situation report and next steps
- **Renewal Alerts** — Never miss a lease deadline
- **Move-In/Out Inspector** — Timestamped photo evidence for deposit disputes
- **Rent Payment Tracker** — Full payment history, exportable as PDF
- **Maintenance Log** — Timestamped repair request trail
- **Rent Benchmarks** — Know if you're overpaying for your neighborhood

## Stack

| Layer | Technology |
|-------|-----------|
| Mobile | React Native + Expo |
| Backend | Python + FastAPI |
| Database | Supabase (Postgres + pgvector) |
| LLM Routing | LiteLLM (swap any provider) |
| Cache | Redis / Upstash |
| Payments | RevenueCat |

## Project Structure

\`\`\`
RentRight/
├── mobile/          # React Native + Expo app
├── backend/         # Python FastAPI backend
├── docs/
│   └── superpowers/
│       └── specs/   # Design specs
└── scripts/         # Data ingestion + cache warming
\`\`\`

## Design Spec

See [docs/superpowers/specs/2026-05-26-renter-app-design.md](docs/superpowers/specs/2026-05-26-renter-app-design.md) for the full product design.

## Data Sources

All public domain, free to use commercially:
- HUD Guidelines
- U.S. Code (Fair Housing Act)
- State Tenant Laws (all 50 states)
- Court Opinions (Free Law Project)
- CFPB Consumer Complaints
- HUD Fair Market Rents + Census ACS

## License

MIT

# DHS APFS Forecast Scraper

A .NET 8 console application that scrapes the [DHS APFS Forecast](https://apfs-cloud.dhs.gov/forecast/) for opportunities related to **ICAM, CIAM, Cybersecurity, Cloud, Software Development, and DevSecOps**, persists them to a SQL Server database, and on every subsequent run detects and reports only **new or changed records** (delta mode).

---

## Features

| Feature | Detail |
|---|---|
| Keyword filtering | Configurable list — ICAM, CIAM, Cybersecurity, Cloud, Software Development, DevSecOps, Zero Trust |
| Detail enrichment | Fetches `https://apfs-cloud.dhs.gov/record/{APFS#}/public-print/` for every match |
| Delta detection | SHA-256 hash of all fields; skips records that haven't changed |
| Field-level diff | Stores before/after values per changed field in `OpportunityChanges` table |
| Sync audit log | Every run is recorded in `SyncRuns` (started, finished, counts, errors) |
| Retry logic | Polly — 3 retries with exponential back-off for transient HTTP errors |
| Configurable | `appsettings.json` or `DHS_SCRAPER_*` environment variables |

---

## Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8)
- SQL Server (LocalDB, Developer Edition, or Azure SQL)

---

## Quick Start

### 1. Configure the connection string

Edit `DhsForecastScraper/appsettings.json`:

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=(localdb)\\mssqllocaldb;Database=DhsForecastDb;Trusted_Connection=True;TrustServerCertificate=True;"
  }
}
```

For a full SQL Server instance:
```
Server=myserver;Database=DhsForecastDb;User Id=myuser;Password=mypassword;TrustServerCertificate=True;
```

### 2. Build and run

```bash
cd DhsForecastScraper
dotnet restore
dotnet build -c Release
dotnet run --project DhsForecastScraper -c Release
```

The app creates the database schema automatically on first run (`EnsureCreated`).

---

## Configuration (`appsettings.json`)

```json
{
  "Scraping": {
    "ForecastBaseUrl": "https://apfs-cloud.dhs.gov/forecast/",
    "RecordDetailBaseUrl": "https://apfs-cloud.dhs.gov/record/{0}/public-print/",
    "Keywords": ["ICAM", "CIAM", "Cybersecurity", "Cloud", "Software Development", "DevSecOps"],
    "RequestDelayMs": 1500,
    "MaxRetries": 3,
    "TimeoutSeconds": 45
  }
}
```

You can also override any setting via environment variables prefixed with `DHS_SCRAPER_`:

```bash
DHS_SCRAPER_ConnectionStrings__DefaultConnection="..." dotnet run ...
```

---

## Database Schema

```
Opportunities
  Id                  INT PK
  ApfsNumber          NVARCHAR(20) UNIQUE   ← natural key
  Title               NVARCHAR(500)
  Component           NVARCHAR(200)
  Office              NVARCHAR(200)
  NaicsCode           NVARCHAR(10)
  NaicsDescription    NVARCHAR(300)
  FiscalYear          NVARCHAR(50)
  EstimatedValue      NVARCHAR(50)
  AnticipatedAwardDate NVARCHAR(100)
  AnticipatedRfpDate  NVARCHAR(100)
  PeriodOfPerformance NVARCHAR(200)
  ContractType        NVARCHAR(200)
  SetAside            NVARCHAR(200)
  PlaceOfPerformance  NVARCHAR(200)
  SecurityClearance   NVARCHAR(200)
  IncumbentContractor NVARCHAR(200)
  PointOfContact      NVARCHAR(200)
  ContactEmail        NVARCHAR(200)
  ContactPhone        NVARCHAR(50)
  Status              NVARCHAR(50)
  Description         NVARCHAR(MAX)
  AdditionalNotes     NVARCHAR(MAX)
  MatchedKeywords     NVARCHAR(500)
  ContentHash         NVARCHAR(64)          ← SHA-256 for delta detection
  FirstSeenAt         DATETIME2
  LastUpdatedAt       DATETIME2
  LastScrapedAt       DATETIME2

OpportunityChanges
  Id              INT PK
  OpportunityId   INT FK → Opportunities
  SyncRunId       INT FK → SyncRuns
  DetectedAt      DATETIME2
  FieldName       NVARCHAR(100)
  OldValue        NVARCHAR(MAX)
  NewValue        NVARCHAR(MAX)

SyncRuns
  Id              INT PK
  StartedAt       DATETIME2
  CompletedAt     DATETIME2
  TotalScraped    INT
  NewRecords      INT
  UpdatedRecords  INT
  UnchangedRecords INT
  ErrorCount      INT
  Status          NVARCHAR(20)
  ErrorSummary    NVARCHAR(MAX)
```

---

## How Delta Detection Works

1. **First run** — every keyword-matching opportunity is inserted fresh.
2. **Subsequent runs** — for each scraped opportunity:
   - Compute `SHA-256(all fields concatenated)`
   - Compare to stored `ContentHash`
   - **Same hash** → mark `LastScrapedAt`, skip
   - **Different hash** → diff every field, store before/after in `OpportunityChanges`, update the record
   - **Not in DB** → insert as new
3. The console prints a **Delta Report** at the end of every run showing exactly what changed.

---

## Note on HTML Parsing

The scraper uses **AngleSharp** and employs multiple fallback strategies to handle different HTML layouts (tables, definition lists, labelled divs, card elements). If the site changes its markup, the selectors in `ForecastListScraper.cs` and `DetailScraper.cs` can be updated without touching any other code.

---

## Adjusting for Site Structure

If the listing page uses a non-standard layout:
1. Enable `Debug` logging in `appsettings.json` → `"Default": "Debug"` to see what the scraper finds.
2. Inspect the live HTML with browser DevTools.
3. Add a custom CSS selector in `ForecastListScraper.ParseListPageAsync`.

---

## Project Structure

```
DhsForecastScraper/
├── DhsForecastScraper.sln
└── DhsForecastScraper/
    ├── Program.cs                  ← DI wiring, entry point
    ├── appsettings.json
    ├── DhsForecastScraper.csproj
    ├── Models/
    │   ├── Opportunity.cs          ← main entity
    │   ├── OpportunityChange.cs    ← field-level delta record
    │   └── SyncRun.cs              ← audit log per execution
    ├── Data/
    │   └── AppDbContext.cs         ← EF Core DbContext
    └── Services/
        ├── ScrapingOptions.cs      ← strongly-typed config
        ├── ForecastListScraper.cs  ← scrapes the listing page
        ├── DetailScraper.cs        ← scrapes individual record pages
        └── SyncService.cs          ← orchestrates everything + reporting
```

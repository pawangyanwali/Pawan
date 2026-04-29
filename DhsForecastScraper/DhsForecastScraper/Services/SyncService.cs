using System.Security.Cryptography;
using System.Text;
using DhsForecastScraper.Data;
using DhsForecastScraper.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DhsForecastScraper.Services;

/// <summary>
/// Orchestrates a full sync cycle:
/// 1. Scrape the list page for keyword-matching opportunities
/// 2. Fetch detail pages
/// 3. Compare against DB (hash-based delta detection)
/// 4. Persist new / updated records and log field-level changes
/// 5. Print a delta report to the console
/// </summary>
public class SyncService
{
    private readonly ForecastListScraper _listScraper;
    private readonly DetailScraper _detailScraper;
    private readonly AppDbContext _db;
    private readonly ScrapingOptions _opts;
    private readonly ILogger<SyncService> _log;

    public SyncService(
        ForecastListScraper listScraper,
        DetailScraper detailScraper,
        AppDbContext db,
        IOptions<ScrapingOptions> opts,
        ILogger<SyncService> log)
    {
        _listScraper = listScraper;
        _detailScraper = detailScraper;
        _db = db;
        _opts = opts.Value;
        _log = log;
    }

    public async Task RunAsync(CancellationToken ct = default)
    {
        // Ensure the DB schema exists (idempotent)
        await _db.Database.EnsureCreatedAsync(ct);

        var run = new SyncRun { StartedAt = DateTime.UtcNow };
        _db.SyncRuns.Add(run);
        await _db.SaveChangesAsync(ct);

        try
        {
            // ---- Step 1: scrape list ----------------------------------------
            Console.WriteLine();
            Console.WriteLine("=== DHS APFS Forecast Scraper ===");
            Console.WriteLine($"Started : {run.StartedAt:yyyy-MM-dd HH:mm:ss} UTC");
            Console.WriteLine($"Keywords: {string.Join(", ", _opts.Keywords)}");
            Console.WriteLine();

            var stubs = await _listScraper.ScrapeListAsync(ct);
            run.TotalScraped = stubs.Count;

            if (stubs.Count == 0)
            {
                Console.WriteLine("[WARN] No matching opportunities found on the listing page.");
                Console.WriteLine("       Check ForecastBaseUrl and Keywords in appsettings.json.");
                FinishRun(run, "CompletedEmpty");
                await _db.SaveChangesAsync(ct);
                return;
            }

            Console.WriteLine($"Found {stubs.Count} keyword-matching opportunities on listing page.");
            Console.WriteLine();

            // ---- Step 2: enrich each stub with detail page data ---------------
            // Load all existing records in one query for efficiency
            var existingByApfs = await _db.Opportunities
                .AsNoTracking()
                .ToDictionaryAsync(o => o.ApfsNumber, StringComparer.OrdinalIgnoreCase, ct);

            var newRecords = new List<Opportunity>();
            var updatedRecords = new List<(Opportunity Fresh, Opportunity Existing, List<(string Field, string? Old, string? New)> Deltas)>();
            var unchangedCount = 0;

            for (int i = 0; i < stubs.Count; i++)
            {
                var stub = stubs[i];
                Console.Write($"  [{i + 1}/{stubs.Count}] APFS#{stub.ApfsNumber} ... ");

                // Delay between detail requests to be polite
                if (i > 0)
                    await Task.Delay(_opts.RequestDelayMs, ct);

                var success = await _detailScraper.EnrichOpportunityAsync(stub, ct);
                if (!success)
                {
                    _log.LogWarning("Skipping APFS#{Num} — could not fetch detail page", stub.ApfsNumber);
                    Console.WriteLine("SKIP (detail fetch failed)");
                    run.ErrorCount++;
                    continue;
                }

                // Record matched keywords
                stub.MatchedKeywords = string.Join(", ",
                    _opts.Keywords.Where(kw =>
                        $"{stub.Title} {stub.Component} {stub.NaicsDescription} {stub.Description}"
                            .Contains(kw, StringComparison.OrdinalIgnoreCase)));

                var freshHash = ComputeHash(stub);
                stub.ContentHash = freshHash;

                if (!existingByApfs.TryGetValue(stub.ApfsNumber, out var existing))
                {
                    // Brand new record
                    stub.FirstSeenAt = DateTime.UtcNow;
                    stub.LastUpdatedAt = DateTime.UtcNow;
                    stub.LastScrapedAt = DateTime.UtcNow;
                    stub.IsNew = true;
                    newRecords.Add(stub);
                    Console.WriteLine("NEW");
                }
                else if (freshHash != existing.ContentHash)
                {
                    // Changed record — compute field-level delta
                    var deltas = ComputeDeltas(existing, stub);
                    stub.LastUpdatedAt = DateTime.UtcNow;
                    stub.LastScrapedAt = DateTime.UtcNow;
                    stub.FirstSeenAt = existing.FirstSeenAt;
                    stub.IsChanged = true;
                    updatedRecords.Add((stub, existing, deltas));
                    Console.WriteLine($"CHANGED ({deltas.Count} field(s))");
                }
                else
                {
                    // No change — just update the scrape timestamp
                    unchangedCount++;
                    Console.WriteLine("unchanged");
                }
            }

            // ---- Step 3: persist --------------------------------------------
            Console.WriteLine();
            Console.WriteLine("--- Persisting to database ---");

            // Insert new records
            if (newRecords.Count > 0)
            {
                _db.Opportunities.AddRange(newRecords);
                await _db.SaveChangesAsync(ct);
                run.NewRecords = newRecords.Count;
                Console.WriteLine($"  Inserted {newRecords.Count} new record(s).");
            }

            // Update changed records
            foreach (var (fresh, existing2, deltas) in updatedRecords)
            {
                var tracked = await _db.Opportunities.FirstAsync(o => o.ApfsNumber == fresh.ApfsNumber, ct);
                CopyFields(fresh, tracked);
                tracked.LastScrapedAt = DateTime.UtcNow;

                foreach (var (field, oldVal, newVal) in deltas)
                {
                    _db.OpportunityChanges.Add(new OpportunityChange
                    {
                        OpportunityId = tracked.Id,
                        SyncRunId = run.Id,
                        DetectedAt = DateTime.UtcNow,
                        FieldName = field,
                        OldValue = oldVal,
                        NewValue = newVal
                    });
                }
            }

            if (updatedRecords.Count > 0)
            {
                await _db.SaveChangesAsync(ct);
                run.UpdatedRecords = updatedRecords.Count;
                Console.WriteLine($"  Updated {updatedRecords.Count} existing record(s).");
            }

            // Update scrape timestamp for unchanged records
            if (unchangedCount > 0)
            {
                var unchangedApfsNums = stubs
                    .Where(s => !s.IsNew && !s.IsChanged)
                    .Select(s => s.ApfsNumber)
                    .ToList();

                await _db.Opportunities
                    .Where(o => unchangedApfsNums.Contains(o.ApfsNumber))
                    .ExecuteUpdateAsync(s => s.SetProperty(o => o.LastScrapedAt, DateTime.UtcNow), ct);

                run.UnchangedRecords = unchangedCount;
                Console.WriteLine($"  {unchangedCount} record(s) unchanged.");
            }

            // ---- Step 4: delta report ---------------------------------------
            PrintDeltaReport(newRecords, updatedRecords);

            FinishRun(run, "Completed");
            await _db.SaveChangesAsync(ct);

            Console.WriteLine();
            Console.WriteLine($"=== Sync complete at {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC ===");
            Console.WriteLine($"    New: {run.NewRecords}  |  Updated: {run.UpdatedRecords}  |  Unchanged: {run.UnchangedRecords}  |  Errors: {run.ErrorCount}");
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Sync run failed");
            run.ErrorSummary = ex.Message;
            FinishRun(run, "Failed");
            await _db.SaveChangesAsync(ct);
            throw;
        }
    }

    // -------------------------------------------------------------------------
    // Delta detection
    // -------------------------------------------------------------------------

    private static List<(string Field, string? Old, string? New)> ComputeDeltas(
        Opportunity existing, Opportunity fresh)
    {
        var changes = new List<(string, string?, string?)>();

        void Check(string field, string? oldVal, string? newVal)
        {
            if (!string.Equals(oldVal?.Trim(), newVal?.Trim(), StringComparison.OrdinalIgnoreCase))
                changes.Add((field, oldVal, newVal));
        }

        Check("Title", existing.Title, fresh.Title);
        Check("Component", existing.Component, fresh.Component);
        Check("Office", existing.Office, fresh.Office);
        Check("NaicsCode", existing.NaicsCode, fresh.NaicsCode);
        Check("NaicsDescription", existing.NaicsDescription, fresh.NaicsDescription);
        Check("FiscalYear", existing.FiscalYear, fresh.FiscalYear);
        Check("EstimatedValue", existing.EstimatedValue, fresh.EstimatedValue);
        Check("AnticipatedAwardDate", existing.AnticipatedAwardDate, fresh.AnticipatedAwardDate);
        Check("AnticipatedRfpDate", existing.AnticipatedRfpDate, fresh.AnticipatedRfpDate);
        Check("PeriodOfPerformance", existing.PeriodOfPerformance, fresh.PeriodOfPerformance);
        Check("ContractType", existing.ContractType, fresh.ContractType);
        Check("SetAside", existing.SetAside, fresh.SetAside);
        Check("PlaceOfPerformance", existing.PlaceOfPerformance, fresh.PlaceOfPerformance);
        Check("SecurityClearance", existing.SecurityClearance, fresh.SecurityClearance);
        Check("IncumbentContractor", existing.IncumbentContractor, fresh.IncumbentContractor);
        Check("PointOfContact", existing.PointOfContact, fresh.PointOfContact);
        Check("ContactEmail", existing.ContactEmail, fresh.ContactEmail);
        Check("ContactPhone", existing.ContactPhone, fresh.ContactPhone);
        Check("Status", existing.Status, fresh.Status);
        Check("Description", existing.Description, fresh.Description);
        Check("AdditionalNotes", existing.AdditionalNotes, fresh.AdditionalNotes);

        return changes;
    }

    // -------------------------------------------------------------------------
    // Reporting
    // -------------------------------------------------------------------------

    private static void PrintDeltaReport(
        List<Opportunity> newRecords,
        List<(Opportunity Fresh, Opportunity Existing, List<(string Field, string? Old, string? New)> Deltas)> updatedRecords)
    {
        if (newRecords.Count == 0 && updatedRecords.Count == 0)
        {
            Console.WriteLine();
            Console.WriteLine("No changes detected since last run.");
            return;
        }

        Console.WriteLine();
        Console.WriteLine(new string('=', 80));
        Console.WriteLine("DELTA REPORT");
        Console.WriteLine(new string('=', 80));

        if (newRecords.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine($"NEW OPPORTUNITIES ({newRecords.Count})");
            Console.WriteLine(new string('-', 60));
            foreach (var opp in newRecords)
            {
                Console.WriteLine($"  APFS#  : {opp.ApfsNumber}");
                Console.WriteLine($"  Title  : {opp.Title ?? "(not parsed)"}");
                Console.WriteLine($"  Component: {opp.Component ?? "—"}");
                Console.WriteLine($"  NAICS  : {opp.NaicsCode} {opp.NaicsDescription}");
                Console.WriteLine($"  Value  : {opp.EstimatedValue ?? "—"}");
                Console.WriteLine($"  Award  : {opp.AnticipatedAwardDate ?? "—"}");
                Console.WriteLine($"  Keywords: {opp.MatchedKeywords}");
                Console.WriteLine();
            }
        }

        if (updatedRecords.Count > 0)
        {
            Console.WriteLine($"CHANGED OPPORTUNITIES ({updatedRecords.Count})");
            Console.WriteLine(new string('-', 60));
            foreach (var (fresh, _, deltas) in updatedRecords)
            {
                Console.WriteLine($"  APFS#  : {fresh.ApfsNumber}");
                Console.WriteLine($"  Title  : {fresh.Title ?? "(not parsed)"}");
                Console.WriteLine($"  Fields changed:");
                foreach (var (field, oldVal, newVal) in deltas)
                {
                    Console.WriteLine($"    • {field}");
                    Console.WriteLine($"        BEFORE: {Truncate(oldVal, 120)}");
                    Console.WriteLine($"        AFTER : {Truncate(newVal, 120)}");
                }
                Console.WriteLine();
            }
        }

        Console.WriteLine(new string('=', 80));
    }

    // -------------------------------------------------------------------------
    // Hashing
    // -------------------------------------------------------------------------

    private static string ComputeHash(Opportunity opp)
    {
        var sb = new StringBuilder();
        sb.Append(opp.Title?.Trim());
        sb.Append('|'); sb.Append(opp.Component?.Trim());
        sb.Append('|'); sb.Append(opp.Office?.Trim());
        sb.Append('|'); sb.Append(opp.NaicsCode?.Trim());
        sb.Append('|'); sb.Append(opp.NaicsDescription?.Trim());
        sb.Append('|'); sb.Append(opp.FiscalYear?.Trim());
        sb.Append('|'); sb.Append(opp.EstimatedValue?.Trim());
        sb.Append('|'); sb.Append(opp.AnticipatedAwardDate?.Trim());
        sb.Append('|'); sb.Append(opp.AnticipatedRfpDate?.Trim());
        sb.Append('|'); sb.Append(opp.PeriodOfPerformance?.Trim());
        sb.Append('|'); sb.Append(opp.ContractType?.Trim());
        sb.Append('|'); sb.Append(opp.SetAside?.Trim());
        sb.Append('|'); sb.Append(opp.PlaceOfPerformance?.Trim());
        sb.Append('|'); sb.Append(opp.SecurityClearance?.Trim());
        sb.Append('|'); sb.Append(opp.IncumbentContractor?.Trim());
        sb.Append('|'); sb.Append(opp.PointOfContact?.Trim());
        sb.Append('|'); sb.Append(opp.ContactEmail?.Trim());
        sb.Append('|'); sb.Append(opp.ContactPhone?.Trim());
        sb.Append('|'); sb.Append(opp.Status?.Trim());
        sb.Append('|'); sb.Append(opp.Description?.Trim());
        sb.Append('|'); sb.Append(opp.AdditionalNotes?.Trim());

        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString()));
        return Convert.ToHexString(bytes).ToLowerInvariant();
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static void CopyFields(Opportunity src, Opportunity dst)
    {
        dst.Title = src.Title;
        dst.Component = src.Component;
        dst.Office = src.Office;
        dst.NaicsCode = src.NaicsCode;
        dst.NaicsDescription = src.NaicsDescription;
        dst.FiscalYear = src.FiscalYear;
        dst.EstimatedValue = src.EstimatedValue;
        dst.AnticipatedAwardDate = src.AnticipatedAwardDate;
        dst.AnticipatedRfpDate = src.AnticipatedRfpDate;
        dst.PeriodOfPerformance = src.PeriodOfPerformance;
        dst.ContractType = src.ContractType;
        dst.SetAside = src.SetAside;
        dst.PlaceOfPerformance = src.PlaceOfPerformance;
        dst.SecurityClearance = src.SecurityClearance;
        dst.IncumbentContractor = src.IncumbentContractor;
        dst.PointOfContact = src.PointOfContact;
        dst.ContactEmail = src.ContactEmail;
        dst.ContactPhone = src.ContactPhone;
        dst.Status = src.Status;
        dst.Description = src.Description;
        dst.AdditionalNotes = src.AdditionalNotes;
        dst.MatchedKeywords = src.MatchedKeywords;
        dst.ContentHash = src.ContentHash;
        dst.LastUpdatedAt = src.LastUpdatedAt;
    }

    private static void FinishRun(SyncRun run, string status)
    {
        run.CompletedAt = DateTime.UtcNow;
        run.Status = status;
    }

    private static string? Truncate(string? s, int max) =>
        s is null ? null : s.Length <= max ? s : s[..max] + "…";
}

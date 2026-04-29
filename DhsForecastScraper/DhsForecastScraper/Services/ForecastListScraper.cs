using AngleSharp;
using AngleSharp.Dom;
using DhsForecastScraper.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DhsForecastScraper.Services;

/// <summary>
/// Scrapes the DHS APFS forecast listing page(s) and returns opportunities
/// that match the configured keywords.
/// </summary>
public class ForecastListScraper
{
    private readonly HttpClient _http;
    private readonly ScrapingOptions _opts;
    private readonly ILogger<ForecastListScraper> _log;

    public ForecastListScraper(
        HttpClient http,
        IOptions<ScrapingOptions> opts,
        ILogger<ForecastListScraper> log)
    {
        _http = http;
        _opts = opts.Value;
        _log = log;
    }

    /// <summary>
    /// Returns a flat list of stub opportunities (ApfsNumber + basic fields) that
    /// match at least one keyword. Detail scraping is done separately.
    /// </summary>
    public async Task<List<Opportunity>> ScrapeListAsync(CancellationToken ct = default)
    {
        var results = new List<Opportunity>();
        var visited = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // The APFS forecast may support pagination; we try common patterns.
        string? nextUrl = _opts.ForecastBaseUrl;

        while (nextUrl is not null)
        {
            _log.LogInformation("Fetching forecast list page: {Url}", nextUrl);
            var html = await FetchHtmlAsync(nextUrl, ct);
            if (string.IsNullOrWhiteSpace(html))
            {
                _log.LogWarning("Empty response from {Url}", nextUrl);
                break;
            }

            var (pageResults, next) = await ParseListPageAsync(html, nextUrl, ct);
            foreach (var opp in pageResults)
            {
                if (visited.Add(opp.ApfsNumber))
                    results.Add(opp);
            }

            nextUrl = next;
            if (nextUrl is not null)
                await Task.Delay(_opts.RequestDelayMs, ct);
        }

        _log.LogInformation("List scrape complete. {Count} keyword-matching opportunities found.", results.Count);
        return results;
    }

    // -------------------------------------------------------------------------

    private async Task<(List<Opportunity> items, string? nextPageUrl)> ParseListPageAsync(
        string html, string pageUrl, CancellationToken ct)
    {
        var config = Configuration.Default;
        var context = BrowsingContext.New(config);
        var doc = await context.OpenAsync(req => req.Content(html), ct);

        var items = new List<Opportunity>();

        // Try to find a <table> containing forecast rows first
        var tables = doc.QuerySelectorAll("table");
        foreach (var table in tables)
        {
            var rows = table.QuerySelectorAll("tbody tr, tr").Skip(1); // skip header row
            foreach (var row in rows)
            {
                var cells = row.QuerySelectorAll("td").ToList();
                if (cells.Count < 2) continue;

                var opp = ParseRowCells(cells, doc);
                if (opp is null) continue;

                if (MatchesKeywords(opp))
                {
                    _log.LogDebug("Matched opportunity {ApfsNumber} — {Title}", opp.ApfsNumber, opp.Title);
                    items.Add(opp);
                }
            }
        }

        // If no table rows found, try card/div-based layout
        if (items.Count == 0)
        {
            var cards = doc.QuerySelectorAll(
                ".forecast-item, .opportunity-card, .record-row, [class*='forecast'], [class*='opportunity']");

            foreach (var card in cards)
            {
                var opp = ParseCardElement(card);
                if (opp is null) continue;

                if (MatchesKeywords(opp))
                {
                    _log.LogDebug("Matched card opportunity {ApfsNumber} — {Title}", opp.ApfsNumber, opp.Title);
                    items.Add(opp);
                }
            }
        }

        // Detect inline anchor links to individual records (href contains /record/)
        if (items.Count == 0)
        {
            var links = doc.QuerySelectorAll("a[href*='/record/']");
            foreach (var a in links)
            {
                var href = a.GetAttribute("href") ?? string.Empty;
                var apfsNum = ExtractApfsFromHref(href);
                if (string.IsNullOrEmpty(apfsNum)) continue;

                var text = a.TextContent.Trim();
                var opp = new Opportunity
                {
                    ApfsNumber = apfsNum,
                    Title = text.Length > 0 ? text : null
                };

                // Walk up to parent row/div to collect adjacent text
                EnrichFromParentContext(a, opp);

                if (MatchesKeywords(opp))
                {
                    _log.LogDebug("Matched link opportunity {ApfsNumber}", opp.ApfsNumber);
                    items.Add(opp);
                }
            }
        }

        // Also scan ALL anchor links whose href contains the record URL pattern
        // even when we already found table rows (we de-dup by ApfsNumber in caller)
        var allRecordLinks = doc.QuerySelectorAll("a[href*='/record/']")
            .Select(a => a.GetAttribute("href"))
            .Where(h => h != null)
            .Distinct()
            .ToList();

        _log.LogDebug("Found {Count} record links on page", allRecordLinks.Count);

        // Detect "next page" link
        string? nextUrl = null;
        var nextLink = doc.QuerySelector("a[rel='next'], a.next, a:contains('Next'), a[aria-label='Next']");
        if (nextLink is not null)
        {
            var href = nextLink.GetAttribute("href");
            if (!string.IsNullOrEmpty(href))
                nextUrl = ResolveUrl(pageUrl, href);
        }

        // Also check for a page=N query parameter pattern
        if (nextUrl is null)
        {
            var pageLinks = doc.QuerySelectorAll("a[href*='page=']");
            // Find the link for (currentPage + 1)
            var currentPage = GetCurrentPage(pageUrl);
            var nextPageLink = pageLinks
                .FirstOrDefault(a => GetCurrentPage(a.GetAttribute("href") ?? "") == currentPage + 1);
            if (nextPageLink is not null)
                nextUrl = ResolveUrl(pageUrl, nextPageLink.GetAttribute("href")!);
        }

        return (items, nextUrl);
    }

    private static Opportunity? ParseRowCells(List<IElement> cells, IDocument doc)
    {
        // The APFS table column order can vary; we try to find the APFS number
        // by looking for a cell that contains a link to /record/ or a 5-digit number.
        string? apfsNumber = null;
        string? title = null;
        string? component = null;
        string? naics = null;
        string? value = null;
        string? awardDate = null;
        string? status = null;

        foreach (var cell in cells)
        {
            var text = cell.TextContent.Trim();
            var link = cell.QuerySelector("a[href*='/record/']");

            if (link is not null && apfsNumber is null)
            {
                var href = link.GetAttribute("href") ?? string.Empty;
                apfsNumber = ExtractApfsFromHref(href);
                if (apfsNumber is null)
                    apfsNumber = ExtractApfsFromText(text);
                // Title often lives in the same cell as the link
                if (title is null && link.TextContent.Length > 5)
                    title = link.TextContent.Trim();
            }

            // Heuristics for field detection
            if (apfsNumber is null && System.Text.RegularExpressions.Regex.IsMatch(text, @"^\d{4,6}$"))
                apfsNumber = text;

            if (System.Text.RegularExpressions.Regex.IsMatch(text, @"^\d{6}$") && naics is null)
                naics = text;

            if (text.StartsWith("$", StringComparison.Ordinal) && value is null)
                value = text;

            if (System.Text.RegularExpressions.Regex.IsMatch(text, @"\d{1,2}/\d{1,2}/\d{2,4}") && awardDate is null)
                awardDate = text;
        }

        if (string.IsNullOrEmpty(apfsNumber))
            return null;

        // If title not set yet, use second non-empty cell
        if (title is null)
        {
            title = cells.Skip(1)
                .Select(c => c.TextContent.Trim())
                .FirstOrDefault(t => t.Length > 5 && !System.Text.RegularExpressions.Regex.IsMatch(t, @"^\d+$"));
        }

        // Component is often 3rd or 4th column
        if (component is null && cells.Count >= 3)
            component = cells[2].TextContent.Trim();

        return new Opportunity
        {
            ApfsNumber = apfsNumber,
            Title = title,
            Component = component,
            NaicsCode = naics,
            EstimatedValue = value,
            AnticipatedAwardDate = awardDate,
            Status = status
        };
    }

    private static Opportunity? ParseCardElement(IElement card)
    {
        var apfsNumber = ExtractApfsFromText(card.TextContent)
            ?? ExtractApfsFromHref(card.QuerySelector("a")?.GetAttribute("href") ?? string.Empty);

        if (string.IsNullOrEmpty(apfsNumber))
            return null;

        var titleEl = card.QuerySelector("h2, h3, h4, .title, .name, strong");

        return new Opportunity
        {
            ApfsNumber = apfsNumber,
            Title = titleEl?.TextContent.Trim() ?? card.TextContent.Trim().Split('\n')[0].Trim()
        };
    }

    private static void EnrichFromParentContext(IElement anchor, Opportunity opp)
    {
        var parent = anchor.ParentElement;
        if (parent is null) return;

        var text = parent.TextContent.Trim();
        if (opp.Title is null && text.Length > 5)
            opp.Title = text.Length > 200 ? text[..200] : text;
    }

    private bool MatchesKeywords(Opportunity opp)
    {
        var searchable = string.Join(" ",
            opp.Title, opp.Component, opp.NaicsDescription,
            opp.Description, opp.Office).ToUpperInvariant();

        return _opts.Keywords.Any(kw =>
            searchable.Contains(kw.ToUpperInvariant()));
    }

    private async Task<string> FetchHtmlAsync(string url, CancellationToken ct)
    {
        try
        {
            var response = await _http.GetAsync(url, ct);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync(ct);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Failed to fetch {Url}", url);
            return string.Empty;
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    public static string? ExtractApfsFromHref(string href)
    {
        // Pattern: /record/54660/... → "54660"
        var m = System.Text.RegularExpressions.Regex.Match(href, @"/record/(\d+)");
        return m.Success ? m.Groups[1].Value : null;
    }

    public static string? ExtractApfsFromText(string text)
    {
        // Match standalone 4-6 digit numbers that look like record IDs
        var m = System.Text.RegularExpressions.Regex.Match(text, @"\b(\d{4,6})\b");
        return m.Success ? m.Groups[1].Value : null;
    }

    private static int GetCurrentPage(string url)
    {
        var m = System.Text.RegularExpressions.Regex.Match(url, @"[?&]page=(\d+)");
        return m.Success ? int.Parse(m.Groups[1].Value) : 1;
    }

    private static string ResolveUrl(string baseUrl, string relative)
    {
        if (relative.StartsWith("http", StringComparison.OrdinalIgnoreCase))
            return relative;

        var baseUri = new Uri(baseUrl);
        return new Uri(baseUri, relative).ToString();
    }
}

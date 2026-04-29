using AngleSharp;
using AngleSharp.Dom;
using DhsForecastScraper.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DhsForecastScraper.Services;

/// <summary>
/// Fetches the public-print detail page for a single APFS record and
/// populates the Opportunity object with all available fields.
/// URL pattern: https://apfs-cloud.dhs.gov/record/{apfsNumber}/public-print/
/// </summary>
public class DetailScraper
{
    private readonly HttpClient _http;
    private readonly ScrapingOptions _opts;
    private readonly ILogger<DetailScraper> _log;

    public DetailScraper(
        HttpClient http,
        IOptions<ScrapingOptions> opts,
        ILogger<DetailScraper> log)
    {
        _http = http;
        _opts = opts.Value;
        _log = log;
    }

    public async Task<bool> EnrichOpportunityAsync(Opportunity opp, CancellationToken ct = default)
    {
        var apfsId = ExtractNumericId(opp.ApfsNumber);
        var url = string.Format(_opts.RecordDetailBaseUrl, apfsId);

        _log.LogInformation("  Fetching detail for APFS#{ApfsNumber} → {Url}", opp.ApfsNumber, url);

        string html;
        try
        {
            var resp = await _http.GetAsync(url, ct);
            if (!resp.IsSuccessStatusCode)
            {
                _log.LogWarning("  Detail page returned {Status} for {ApfsNumber}", resp.StatusCode, opp.ApfsNumber);
                return false;
            }
            html = await resp.Content.ReadAsStringAsync(ct);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "  Failed to fetch detail for {ApfsNumber}", opp.ApfsNumber);
            return false;
        }

        await ParseDetailPageAsync(html, opp, ct);
        return true;
    }

    // -------------------------------------------------------------------------

    private static async Task ParseDetailPageAsync(string html, Opportunity opp, CancellationToken ct)
    {
        var config = Configuration.Default;
        var context = BrowsingContext.New(config);
        var doc = await context.OpenAsync(req => req.Content(html), ct);

        // Strategy 1 — definition list (<dl><dt>Label</dt><dd>Value</dd></dl>)
        ParseDefinitionLists(doc, opp);

        // Strategy 2 — two-column table (label | value)
        ParseLabelValueTable(doc, opp);

        // Strategy 3 — look for labelled sections / divs with data-label attributes
        ParseLabelledDivs(doc, opp);

        // Strategy 4 — extract all visible text pairs using a generic heuristic
        if (AllFieldsEmpty(opp))
            ParseFallbackTextPairs(doc, opp);

        // Always try to capture the description / scope of work
        if (string.IsNullOrEmpty(opp.Description))
            opp.Description = ExtractDescription(doc);

        // Trim all string fields
        TrimFields(opp);
    }

    // -------------------------

    private static void ParseDefinitionLists(IDocument doc, Opportunity opp)
    {
        var dls = doc.QuerySelectorAll("dl");
        foreach (var dl in dls)
        {
            var terms = dl.QuerySelectorAll("dt").ToList();
            var defs = dl.QuerySelectorAll("dd").ToList();

            for (int i = 0; i < Math.Min(terms.Count, defs.Count); i++)
            {
                var label = terms[i].TextContent.Trim();
                var value = defs[i].TextContent.Trim();
                ApplyField(opp, label, value);
            }
        }
    }

    private static void ParseLabelValueTable(IDocument doc, Opportunity opp)
    {
        var tables = doc.QuerySelectorAll("table");
        foreach (var table in tables)
        {
            foreach (var row in table.QuerySelectorAll("tr"))
            {
                var cells = row.QuerySelectorAll("th, td").ToList();
                if (cells.Count == 2)
                {
                    ApplyField(opp, cells[0].TextContent.Trim(), cells[1].TextContent.Trim());
                }
                else if (cells.Count == 4)
                {
                    // Some pages use a 4-col layout: label | value | label | value
                    ApplyField(opp, cells[0].TextContent.Trim(), cells[1].TextContent.Trim());
                    ApplyField(opp, cells[2].TextContent.Trim(), cells[3].TextContent.Trim());
                }
            }
        }
    }

    private static void ParseLabelledDivs(IDocument doc, Opportunity opp)
    {
        // <div class="field"><label>...</label><span>...</span></div>
        var fields = doc.QuerySelectorAll(".field, .form-group, .data-field, [class*='field-']");
        foreach (var field in fields)
        {
            var label = field.QuerySelector("label, .field-label, strong")?.TextContent.Trim();
            var value = field.QuerySelector("span, p, .field-value, input, textarea")?.TextContent.Trim()
                        ?? field.QuerySelectorAll("*").LastOrDefault()?.TextContent.Trim();

            if (!string.IsNullOrEmpty(label) && !string.IsNullOrEmpty(value))
                ApplyField(opp, label, value);
        }

        // data-label attributes
        foreach (var el in doc.QuerySelectorAll("[data-label]"))
        {
            var label = el.GetAttribute("data-label")?.Trim() ?? string.Empty;
            var value = el.TextContent.Trim();
            if (!string.IsNullOrEmpty(label))
                ApplyField(opp, label, value);
        }
    }

    private static void ParseFallbackTextPairs(IDocument doc, Opportunity opp)
    {
        // Walk all elements looking for bold/strong labels followed by text
        var strongEls = doc.QuerySelectorAll("strong, b, th, label");
        foreach (var el in strongEls)
        {
            var label = el.TextContent.Trim().TrimEnd(':');
            var sibling = el.NextElementSibling;
            var value = sibling?.TextContent.Trim() ?? el.ParentElement?.TextContent.Replace(label, "").Trim();
            if (!string.IsNullOrEmpty(label) && !string.IsNullOrEmpty(value))
                ApplyField(opp, label, value);
        }
    }

    private static string? ExtractDescription(IDocument doc)
    {
        // Look for a section labelled "description", "scope", "requirements", etc.
        var candidates = doc.QuerySelectorAll(
            "p, div.description, div.scope, div.requirements, .narrative, textarea, [class*='desc'], [class*='scope']");

        foreach (var el in candidates)
        {
            var text = el.TextContent.Trim();
            // Pick the first substantial block of text (> 100 chars)
            if (text.Length > 100)
                return text;
        }
        return null;
    }

    // -------------------------------------------------------------------------
    // Field mapping — normalises label text to Opportunity properties
    // -------------------------------------------------------------------------

    private static void ApplyField(Opportunity opp, string rawLabel, string value)
    {
        if (string.IsNullOrWhiteSpace(rawLabel) || string.IsNullOrWhiteSpace(value)) return;

        var label = rawLabel.ToUpperInvariant()
                            .Replace(":", "")
                            .Replace("_", " ")
                            .Trim();

        switch (label)
        {
            case var l when l.Contains("APFS") && l.Contains("NUMBER"):
            case var _ when label is "APFS #" or "RECORD NUMBER" or "RECORD #" or "APFS":
                if (string.IsNullOrEmpty(opp.ApfsNumber)) opp.ApfsNumber = value;
                break;

            case var l when l.Contains("TITLE") || l.Contains("REQUIREMENT"):
                if (string.IsNullOrEmpty(opp.Title)) opp.Title = value;
                break;

            case var l when l.Contains("COMPONENT") && !l.Contains("CONTRACT"):
                if (string.IsNullOrEmpty(opp.Component)) opp.Component = value;
                break;

            case var l when l.Contains("OFFICE"):
                if (string.IsNullOrEmpty(opp.Office)) opp.Office = value;
                break;

            case var l when l.Contains("NAICS") && l.Contains("CODE"):
                if (string.IsNullOrEmpty(opp.NaicsCode)) opp.NaicsCode = value;
                break;

            case var l when l.Contains("NAICS") && l.Contains("DESCRIPTION"):
            case "NAICS DESC":
                if (string.IsNullOrEmpty(opp.NaicsDescription)) opp.NaicsDescription = value;
                break;

            case var l when l.Contains("FISCAL YEAR") || l == "FY":
                if (string.IsNullOrEmpty(opp.FiscalYear)) opp.FiscalYear = value;
                break;

            case var l when l.Contains("ESTIMATED") && l.Contains("VALUE"):
            case "CONTRACT VALUE":
            case "ESTIMATED CONTRACT VALUE":
                if (string.IsNullOrEmpty(opp.EstimatedValue)) opp.EstimatedValue = value;
                break;

            case var l when l.Contains("AWARD DATE") || l.Contains("ANTICIPATED AWARD"):
                if (string.IsNullOrEmpty(opp.AnticipatedAwardDate)) opp.AnticipatedAwardDate = value;
                break;

            case var l when l.Contains("RFP") || l.Contains("SOLICITATION"):
                if (string.IsNullOrEmpty(opp.AnticipatedRfpDate)) opp.AnticipatedRfpDate = value;
                break;

            case var l when l.Contains("PERIOD OF PERFORMANCE") || l == "POP":
                if (string.IsNullOrEmpty(opp.PeriodOfPerformance)) opp.PeriodOfPerformance = value;
                break;

            case var l when l.Contains("CONTRACT TYPE"):
                if (string.IsNullOrEmpty(opp.ContractType)) opp.ContractType = value;
                break;

            case var l when l.Contains("SET-ASIDE") || l.Contains("SET ASIDE"):
                if (string.IsNullOrEmpty(opp.SetAside)) opp.SetAside = value;
                break;

            case var l when l.Contains("PLACE OF PERFORMANCE") || l == "POP LOCATION":
                if (string.IsNullOrEmpty(opp.PlaceOfPerformance)) opp.PlaceOfPerformance = value;
                break;

            case var l when l.Contains("SECURITY CLEARANCE") || l.Contains("CLEARANCE"):
                if (string.IsNullOrEmpty(opp.SecurityClearance)) opp.SecurityClearance = value;
                break;

            case var l when l.Contains("INCUMBENT"):
                if (string.IsNullOrEmpty(opp.IncumbentContractor)) opp.IncumbentContractor = value;
                break;

            case var l when l.Contains("POINT OF CONTACT") || l == "POC" || l.Contains("CONTRACTING OFFICER"):
                if (string.IsNullOrEmpty(opp.PointOfContact)) opp.PointOfContact = value;
                break;

            case var l when l.Contains("EMAIL"):
                if (string.IsNullOrEmpty(opp.ContactEmail)) opp.ContactEmail = value;
                break;

            case var l when l.Contains("PHONE") || l.Contains("TELEPHONE"):
                if (string.IsNullOrEmpty(opp.ContactPhone)) opp.ContactPhone = value;
                break;

            case var l when l.Contains("STATUS"):
                if (string.IsNullOrEmpty(opp.Status)) opp.Status = value;
                break;

            case var l when l.Contains("DESCRIPTION") || l.Contains("SCOPE") || l.Contains("NARRATIVE"):
                if (string.IsNullOrEmpty(opp.Description)) opp.Description = value;
                break;

            case var l when l.Contains("NOTE") || l.Contains("COMMENT") || l.Contains("ADDITIONAL"):
                if (string.IsNullOrEmpty(opp.AdditionalNotes)) opp.AdditionalNotes = value;
                break;
        }
    }

    // -------------------------------------------------------------------------

    private static bool AllFieldsEmpty(Opportunity opp) =>
        string.IsNullOrEmpty(opp.Title) &&
        string.IsNullOrEmpty(opp.Component) &&
        string.IsNullOrEmpty(opp.NaicsCode) &&
        string.IsNullOrEmpty(opp.EstimatedValue);

    private static void TrimFields(Opportunity opp)
    {
        opp.Title = opp.Title?.Trim();
        opp.Component = opp.Component?.Trim();
        opp.Office = opp.Office?.Trim();
        opp.NaicsCode = opp.NaicsCode?.Trim();
        opp.NaicsDescription = opp.NaicsDescription?.Trim();
        opp.FiscalYear = opp.FiscalYear?.Trim();
        opp.EstimatedValue = opp.EstimatedValue?.Trim();
        opp.AnticipatedAwardDate = opp.AnticipatedAwardDate?.Trim();
        opp.AnticipatedRfpDate = opp.AnticipatedRfpDate?.Trim();
        opp.PeriodOfPerformance = opp.PeriodOfPerformance?.Trim();
        opp.ContractType = opp.ContractType?.Trim();
        opp.SetAside = opp.SetAside?.Trim();
        opp.PlaceOfPerformance = opp.PlaceOfPerformance?.Trim();
        opp.SecurityClearance = opp.SecurityClearance?.Trim();
        opp.IncumbentContractor = opp.IncumbentContractor?.Trim();
        opp.PointOfContact = opp.PointOfContact?.Trim();
        opp.ContactEmail = opp.ContactEmail?.Trim();
        opp.ContactPhone = opp.ContactPhone?.Trim();
        opp.Status = opp.Status?.Trim();
        opp.Description = opp.Description?.Trim();
        opp.AdditionalNotes = opp.AdditionalNotes?.Trim();
    }

    /// <summary>
    /// The detail URL uses only the numeric portion of the APFS number.
    /// e.g. "DHS-24-54660" → "54660", "54660" → "54660"
    /// </summary>
    public static string ExtractNumericId(string apfsNumber)
    {
        // Take last contiguous group of digits (up to 6)
        var matches = System.Text.RegularExpressions.Regex.Matches(apfsNumber, @"\d+");
        return matches.Count > 0 ? matches[^1].Value : apfsNumber;
    }
}

using System.ComponentModel.DataAnnotations;

namespace DhsForecastScraper.Models;

/// <summary>
/// Audit record for each execution of the scraper.
/// </summary>
public class SyncRun
{
    [Key]
    public int Id { get; set; }

    public DateTime StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }

    public int TotalScraped { get; set; }
    public int NewRecords { get; set; }
    public int UpdatedRecords { get; set; }
    public int UnchangedRecords { get; set; }
    public int ErrorCount { get; set; }

    [MaxLength(20)]
    public string Status { get; set; } = "Running";

    public string? ErrorSummary { get; set; }

    public ICollection<OpportunityChange> Changes { get; set; } = new List<OpportunityChange>();
}

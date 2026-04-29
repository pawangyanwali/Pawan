using System.ComponentModel.DataAnnotations;

namespace DhsForecastScraper.Models;

/// <summary>
/// Records a single field-level change detected on a re-scrape.
/// </summary>
public class OpportunityChange
{
    [Key]
    public int Id { get; set; }

    public int OpportunityId { get; set; }
    public Opportunity Opportunity { get; set; } = null!;

    public int SyncRunId { get; set; }
    public SyncRun SyncRun { get; set; } = null!;

    public DateTime DetectedAt { get; set; }

    [MaxLength(100)]
    public string FieldName { get; set; } = string.Empty;

    public string? OldValue { get; set; }
    public string? NewValue { get; set; }
}

using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DhsForecastScraper.Models;

/// <summary>
/// Represents a single DHS APFS forecast opportunity.
/// </summary>
public class Opportunity
{
    [Key]
    public int Id { get; set; }

    /// <summary>
    /// APFS record number (e.g. "54660"). Used as the natural unique key.
    /// </summary>
    [Required, MaxLength(20)]
    public string ApfsNumber { get; set; } = string.Empty;

    [MaxLength(500)]
    public string? Title { get; set; }

    [MaxLength(200)]
    public string? Component { get; set; }

    [MaxLength(200)]
    public string? Office { get; set; }

    [MaxLength(10)]
    public string? NaicsCode { get; set; }

    [MaxLength(300)]
    public string? NaicsDescription { get; set; }

    [MaxLength(50)]
    public string? FiscalYear { get; set; }

    [MaxLength(50)]
    public string? EstimatedValue { get; set; }

    [MaxLength(100)]
    public string? AnticipatedAwardDate { get; set; }

    [MaxLength(100)]
    public string? AnticipatedRfpDate { get; set; }

    [MaxLength(200)]
    public string? PeriodOfPerformance { get; set; }

    [MaxLength(200)]
    public string? ContractType { get; set; }

    [MaxLength(200)]
    public string? SetAside { get; set; }

    [MaxLength(200)]
    public string? PlaceOfPerformance { get; set; }

    [MaxLength(200)]
    public string? SecurityClearance { get; set; }

    [MaxLength(200)]
    public string? IncumbentContractor { get; set; }

    [MaxLength(200)]
    public string? PointOfContact { get; set; }

    [MaxLength(200)]
    public string? ContactEmail { get; set; }

    [MaxLength(50)]
    public string? ContactPhone { get; set; }

    [MaxLength(50)]
    public string? Status { get; set; }

    /// <summary>
    /// Full description / scope of work from the detail page.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Additional notes or comments from the detail page.
    /// </summary>
    public string? AdditionalNotes { get; set; }

    /// <summary>
    /// Keywords that caused this record to match the filter.
    /// </summary>
    [MaxLength(500)]
    public string? MatchedKeywords { get; set; }

    /// <summary>
    /// SHA-256 hash of all scraped fields — used for delta detection.
    /// </summary>
    [MaxLength(64)]
    public string ContentHash { get; set; } = string.Empty;

    public DateTime FirstSeenAt { get; set; }
    public DateTime LastUpdatedAt { get; set; }
    public DateTime LastScrapedAt { get; set; }

    [NotMapped]
    public bool IsNew { get; set; }

    [NotMapped]
    public bool IsChanged { get; set; }

    public ICollection<OpportunityChange> Changes { get; set; } = new List<OpportunityChange>();
}

namespace DhsForecastScraper.Services;

public class ScrapingOptions
{
    public const string Section = "Scraping";

    public string ForecastBaseUrl { get; set; } = "https://apfs-cloud.dhs.gov/forecast/";
    public string RecordDetailBaseUrl { get; set; } = "https://apfs-cloud.dhs.gov/record/{0}/public-print/";
    public List<string> Keywords { get; set; } = new();
    public int RequestDelayMs { get; set; } = 1500;
    public int MaxRetries { get; set; } = 3;
    public int TimeoutSeconds { get; set; } = 45;
    public string UserAgent { get; set; } = "Mozilla/5.0 (compatible; DhsForecastBot/1.0)";
}

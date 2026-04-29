using DhsForecastScraper.Data;
using DhsForecastScraper.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Polly;
using Polly.Extensions.Http;

var host = Host.CreateDefaultBuilder(args)
    .ConfigureAppConfiguration((ctx, cfg) =>
    {
        cfg.SetBasePath(AppContext.BaseDirectory)
           .AddJsonFile("appsettings.json", optional: false, reloadOnChange: false)
           .AddEnvironmentVariables("DHS_SCRAPER_");
    })
    .ConfigureServices((ctx, services) =>
    {
        var cfg = ctx.Configuration;

        // Options
        services.Configure<ScrapingOptions>(cfg.GetSection(ScrapingOptions.Section));

        // EF Core — SQL Server
        services.AddDbContext<AppDbContext>(options =>
            options.UseSqlServer(cfg.GetConnectionString("DefaultConnection")));

        // HTTP client with retry policy (3 retries, exponential back-off)
        var retryPolicy = HttpPolicyExtensions
            .HandleTransientHttpError()
            .WaitAndRetryAsync(3, attempt => TimeSpan.FromSeconds(Math.Pow(2, attempt)));

        var scrapingOpts = cfg.GetSection(ScrapingOptions.Section).Get<ScrapingOptions>()!;

        services.AddHttpClient<ForecastListScraper>(client =>
        {
            client.Timeout = TimeSpan.FromSeconds(scrapingOpts.TimeoutSeconds);
            client.DefaultRequestHeaders.Add("User-Agent", scrapingOpts.UserAgent);
            client.DefaultRequestHeaders.Add("Accept", "text/html,application/xhtml+xml");
        }).AddPolicyHandler(retryPolicy);

        services.AddHttpClient<DetailScraper>(client =>
        {
            client.Timeout = TimeSpan.FromSeconds(scrapingOpts.TimeoutSeconds);
            client.DefaultRequestHeaders.Add("User-Agent", scrapingOpts.UserAgent);
            client.DefaultRequestHeaders.Add("Accept", "text/html,application/xhtml+xml");
        }).AddPolicyHandler(retryPolicy);

        // Application services
        services.AddScoped<ForecastListScraper>();
        services.AddScoped<DetailScraper>();
        services.AddScoped<SyncService>();
    })
    .ConfigureLogging(logging =>
    {
        logging.ClearProviders();
        logging.AddConsole(opts => opts.FormatterName = "simple");
        logging.AddFilter("Microsoft.EntityFrameworkCore", LogLevel.Warning);
        logging.AddFilter("Polly", LogLevel.Warning);
    })
    .Build();

// Run the sync
using var scope = host.Services.CreateScope();
var sync = scope.ServiceProvider.GetRequiredService<SyncService>();

try
{
    await sync.RunAsync();
}
catch (Exception ex)
{
    Console.Error.WriteLine($"[FATAL] {ex.Message}");
    Environment.Exit(1);
}

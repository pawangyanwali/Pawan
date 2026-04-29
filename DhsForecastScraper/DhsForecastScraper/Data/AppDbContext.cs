using DhsForecastScraper.Models;
using Microsoft.EntityFrameworkCore;

namespace DhsForecastScraper.Data;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    public DbSet<Opportunity> Opportunities => Set<Opportunity>();
    public DbSet<OpportunityChange> OpportunityChanges => Set<OpportunityChange>();
    public DbSet<SyncRun> SyncRuns => Set<SyncRun>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Opportunity>(e =>
        {
            e.HasIndex(o => o.ApfsNumber).IsUnique();
            e.HasMany(o => o.Changes)
             .WithOne(c => c.Opportunity)
             .HasForeignKey(c => c.OpportunityId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        modelBuilder.Entity<OpportunityChange>(e =>
        {
            e.HasOne(c => c.SyncRun)
             .WithMany(r => r.Changes)
             .HasForeignKey(c => c.SyncRunId)
             .OnDelete(DeleteBehavior.Restrict);
        });
    }
}

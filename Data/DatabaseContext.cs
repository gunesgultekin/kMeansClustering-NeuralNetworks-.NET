using Microsoft.EntityFrameworkCore;
using static Tensorflow.Keras.Engine.InputSpec;

namespace NeuralNetworks.Data
{
    public class DatabaseContext : DbContext
    {
        private readonly IConfiguration config;

        public DatabaseContext(DbContextOptions<DatabaseContext> dbContextOptions)
        {

            this.ChangeTracker.LazyLoadingEnabled = false;
            config = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .Build();
        }

        public DbSet<part1_train> part1_train { get; set; }

        public DbSet<part1_test> part1_test { get; set; }



        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlServer(connectionConfiguration.connectionString);


        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<part1_train>(entity =>
            {
                entity.HasNoKey();

            });

            modelBuilder.Entity<part1_test>(entity =>
            {
                entity.HasNoKey();
            });
            

        }
    }
}

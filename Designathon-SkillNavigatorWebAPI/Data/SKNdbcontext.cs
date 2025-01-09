using Designathon_SkillNavigatorWebAPI.Entities;
using Microsoft.EntityFrameworkCore;

namespace Designathon_SkillNavigatorWebAPI.Data
{
    public class SKNdbcontext : DbContext
    {
        public SKNdbcontext(DbContextOptions options) : base(options)
        {

        }
        public DbSet<User> Users { get; set; }
        public DbSet<Batch> Batches { get; set; }
        public DbSet<BatchAllocation> BatchAllocations { get; set; }
        public DbSet<Course> Courses { get; set; }
        public DbSet<Enrollment> Enrollments { get; set; }
        public DbSet<Feedback> Feedbacks { get; set; }
        public DbSet<Trainer> Trainers { get; set; }
        public DbSet<TrainerAssignment> TrainerAssignments { get; set; }
        public DbSet<Report> Reports { get; set; }
        public DbSet<Certificate> Certificates { get; set; }
    }
}

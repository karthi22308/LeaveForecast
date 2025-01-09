using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace Designathon_SkillNavigatorWebAPI.Entities
{
    public class Report
    {
        [Key]
        public int ReportId { get; set; }

        [ForeignKey("User")]
        public int UserId { get; set; }

        [ForeignKey("Batch")]
        public int BatchId { get; set; }

        [Required]
        public string ReportType { get; set; } // e.g., Individual, Batch, College-Wise

        public DateTime GeneratedDate { get; set; } = DateTime.UtcNow;

        [Column(TypeName = "nvarchar(max)")]
        public string Content { get; set; }

        // Navigation Properties
        public virtual User User { get; set; }
        public virtual Batch Batch { get; set; }
        public class CandidateReport
        {
            public string Name { get; set; }
            public string Email { get; set; }
            public string Degree { get; set; }
            public string Specialization { get; set; }
            public List<string> Certifications { get; set; }
            public string Internships { get; set; }
            public int Completion { get; set; }
        }

        public class TrainersFeedbackReport
        {
            public double AverageScore { get; set; }
            public List<string> Comments { get; set; }
        }

        public class BatchScoreReport
        {
            public double AverageMCQ { get; set; }
            public double ProjectScore { get; set; }
        }

        public class CollegeScoreReport
        {
            public string TopCollege { get; set; }
        }

        public class TopperReport
        {

            public List<Topper> Toppers { get; set; }

            public class Topper
            {
                public string Name { get; set; }
                public double Score { get; set; }

                public Topper(string name, double score)
                {
                    Name = name;
                    Score = score;
                }
            }



        }

        public class BatchComparisonReport
        {
            public string BestBatch { get; set; }
        }
    }
}
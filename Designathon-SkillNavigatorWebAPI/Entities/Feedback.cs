using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace Designathon_SkillNavigatorWebAPI.Entities
{
    public class Feedback
    {
        [Key]
        public int FeedbackId { get; set; }

        [ForeignKey("User")]
        public int UserId { get; set; }

        [ForeignKey("Batch")]
        public int BatchId { get; set; }

        [ForeignKey("Course")]
        public int CourseId { get; set; }

        public string FeedbackText { get; set; }

        [Range(1, 5)]
        public int Rating { get; set; }

        public DateTime FeedbackDate { get; set; } = DateTime.UtcNow;

        // Navigation Properties
        public virtual User User { get; set; }
        public virtual Batch Batch { get; set; }
        public virtual Course Course { get; set; }
    }
}


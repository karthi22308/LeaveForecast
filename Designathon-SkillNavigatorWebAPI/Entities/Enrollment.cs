using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace Designathon_SkillNavigatorWebAPI.Entities
{
    public class Enrollment
    {
        [Key]
        public int EnrollmentId { get; set; }

        [ForeignKey("Course")]
        public int CourseId { get; set; }

        [ForeignKey("User")]
        public int UserId { get; set; }

        public DateTime EnrollmentDate { get; set; }

        public DateTime? CompletionDate { get; set; }

        [MaxLength(50)]
        public string CompletionStatus { get; set; } // e.g., Completed, In Progress

        public int? Score { get; set; }

        // Navigation Properties
        public virtual Course Course { get; set; }
        public virtual User User { get; set; }

    }
}

using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace Designathon_SkillNavigatorWebAPI.Entities
{
    public class Certificate
    {
        [Key]
        public int CertificateId { get; set; }

        [ForeignKey("User")]
        public int UserId { get; set; }

        [ForeignKey("Course")]
        public int CourseId { get; set; }

        public DateTime IssueDate { get; set; } = DateTime.UtcNow;

        [Required]
        [Url]
        public string CertificateLink { get; set; }

        // Navigation Properties
        public virtual User User { get; set; }
        public virtual Course Course { get; set; }
    }
}

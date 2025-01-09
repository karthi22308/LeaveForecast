using Microsoft.VisualBasic;
using System.ComponentModel.DataAnnotations;

namespace Designathon_SkillNavigatorWebAPI.Entities
{
    public class Batch
    {
        [Key]
        public int BatchId { get; set; }

        [Required]
        [MaxLength(100)]
        public string BatchName { get; set; }

        [Required]
        [MaxLength(50)]
        public string BatchType { get; set; } // e.g., Java, .NET, Data Engineering

        [Required]
        public DateTime StartDate { get; set; }

        [Required]
        public DateTime EndDate { get; set; }

        [Required]
        public int Capacity { get; set; }

        // Navigation Properties
        public virtual ICollection<BatchAllocation> BatchAllocations { get; set; }
        public virtual ICollection<TrainerAssignment> TrainerAssignments { get; set; }
        public virtual ICollection<Feedback> Feedbacks { get; set; }
        public virtual ICollection<Report> Reports { get; set; }
    }
}

using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace Designathon_SkillNavigatorWebAPI.Entities
{
    public class TrainerAssignment
    {
        [Key]
        public int AssignmentId { get; set; }

        [ForeignKey("Trainer")]
        public int TrainerId { get; set; }

        [ForeignKey("Batch")]
        public int BatchId { get; set; }

        public DateTime AssignedDate { get; set; } = DateTime.UtcNow;

        // Navigation Properties
        public virtual Trainer Trainer { get; set; }
        public virtual Batch Batch { get; set; }
    }
}

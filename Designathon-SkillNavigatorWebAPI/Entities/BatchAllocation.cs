using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Designathon_SkillNavigatorWebAPI.Entities
{
    public class BatchAllocation
    {
        [Key]
        public int AllocationId { get; set; }

        [ForeignKey("Batch")]
        public int BatchId { get; set; }

        [ForeignKey("User")]
        public int UserId { get; set; }

        [Required]
        public DateTime AllocationDate { get; set; }

        // Navigation Properties
        public virtual Batch Batch { get; set; }
        public virtual User User { get; set; }
    }
}

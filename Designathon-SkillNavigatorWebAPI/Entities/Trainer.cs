using System.ComponentModel.DataAnnotations;

namespace Designathon_SkillNavigatorWebAPI.Entities
{
    public class Trainer
    {
        [Key]
        public int TrainerId { get; set; }

        [Required]
        [MaxLength(100)]
        public string Name { get; set; }

        [Required]
        [EmailAddress]
        public string Email { get; set; }

        [Required]
        [Phone]
        public string PhoneNumber { get; set; }

        public string Specialization { get; set; }

        // Navigation Property
        public virtual ICollection<TrainerAssignment> TrainerAssignments { get; set; }
    }
}

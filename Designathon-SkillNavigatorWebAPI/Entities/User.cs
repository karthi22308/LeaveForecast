using System.Numerics;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace Designathon_SkillNavigatorWebAPI.Entities
{
    public class User
    {
        public int UserId { get; set; }
        public string Username { get; set; }
        public string Password { get; set; }
        public string Email { get; set; }
        public string Role { get; set; }
        public string Degree { get; set; }
        public string Specialization { get; set; }
        public string PhoneNumber { get; set; }
        public string Certifications { get; set; }

        public string InternshipDetails { get; set; }
        public string College { get; set; }
        public string LinkedInProfileLink { get; set; }
        public string ProgrammingLanguagesKnown { get; set; }

    }
}

using Designathon_SkillNavigatorWebAPI.Entities;

namespace Designathon_SkillNavigatorWebAPI.Data
{
    public class DbInitializer
    {
        public static void Initialize(SKNdbcontext context)
        {

            if (context.Users.Any()) return;
            var User = new List<User>
            {
                new User
                {
                    Username="student",
                    Password="student",
                    Email ="student@hexaware.com",
                    Role = "Student",
                                        
                },
                  new User
                {
                    Username="student2",
                    Password="student2",
                    Email ="student2@hexaware.com",
                    Role = "Student",

                },
                    new User
                {
                    Username="admin",
                    Password="admin",
                    Email ="admin@hexaware.com",
                    Role = "Admin",

                },
                      new User
                {
                    Username="trainer",
                    Password="trainer",
                    Email ="trainer@hexaware.com",
                    Role = "Trainer",

                }
            };
            context.Users.AddRange(User);
            context.SaveChanges();
        }
    }
}

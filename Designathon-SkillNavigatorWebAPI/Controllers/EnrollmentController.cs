using Designathon_SkillNavigatorWebAPI.Data;
using Designathon_SkillNavigatorWebAPI.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

[Route("api/[controller]")]
[ApiController]
public class EnrollmentController : ControllerBase
{
    private readonly SKNdbcontext _context;

    public EnrollmentController(SKNdbcontext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<IActionResult> GetAllEnrollments()
    {
        var enrollments = await _context.Enrollments.Include(e => e.Course).Include(e => e.User).ToListAsync();
        return Ok(enrollments);
    }

    [HttpPost]
    public async Task<IActionResult> CreateEnrollment([FromBody] Enrollment enrollment)
    {
        _context.Enrollments.Add(enrollment);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetAllEnrollments), enrollment);
    }
}

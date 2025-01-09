using Designathon_SkillNavigatorWebAPI.Data;
using Designathon_SkillNavigatorWebAPI.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

[Route("api/[controller]")]
[ApiController]
public class FeedbackController : ControllerBase
{
    private readonly SKNdbcontext _context;

    public FeedbackController(SKNdbcontext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<IActionResult> GetAllFeedbacks()
    {
        var feedbacks = await _context.Feedbacks.Include(f => f.User).Include(f => f.Batch).Include(f => f.Course).ToListAsync();
        return Ok(feedbacks);
    }

    [HttpPost]
    public async Task<IActionResult> CreateFeedback([FromBody] Feedback feedback)
    {
        _context.Feedbacks.Add(feedback);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetAllFeedbacks), feedback);
    }
}

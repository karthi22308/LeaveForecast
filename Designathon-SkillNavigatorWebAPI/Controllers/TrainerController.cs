using Designathon_SkillNavigatorWebAPI.Data;
using Designathon_SkillNavigatorWebAPI.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

[Route("api/[controller]")]
[ApiController]
public class TrainerController : ControllerBase
{
    private readonly SKNdbcontext _context;

    public TrainerController(SKNdbcontext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<IActionResult> GetAllTrainers()
    {
        var trainers = await _context.Trainers.ToListAsync();
        return Ok(trainers);
    }

    [HttpPost]
    public async Task<IActionResult> CreateTrainer([FromBody] Trainer trainer)
    {
        _context.Trainers.Add(trainer);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetAllTrainers), trainer);
    }
}

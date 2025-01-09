using Designathon_SkillNavigatorWebAPI.Data;
using Designathon_SkillNavigatorWebAPI.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

[Route("api/[controller]")]
[ApiController]
public class BatchAllocationController : ControllerBase
{
    private readonly SKNdbcontext _context;

    public BatchAllocationController(SKNdbcontext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<IActionResult> GetAllAllocations()
    {
        var allocations = await _context.BatchAllocations.Include(b => b.Batch).Include(u => u.User).ToListAsync();
        return Ok(allocations);
    }

    [HttpPost]
    public async Task<IActionResult> CreateAllocation([FromBody] BatchAllocation allocation)
    {
        _context.BatchAllocations.Add(allocation);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetAllAllocations), allocation);
    }
}

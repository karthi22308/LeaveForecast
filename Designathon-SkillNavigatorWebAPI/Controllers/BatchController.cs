using Designathon_SkillNavigatorWebAPI.Data;
using Designathon_SkillNavigatorWebAPI.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

[Route("api/[controller]")]
[ApiController]
public class BatchController : ControllerBase
{
    private readonly SKNdbcontext _context;

    public BatchController(SKNdbcontext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<IActionResult> GetAllBatches()
    {
        var batches = await _context.Batches.ToListAsync();
        return Ok(batches);
    }

    [HttpGet("{id}")]
    public async Task<IActionResult> GetBatch(int id)
    {
        var batch = await _context.Batches.FindAsync(id);
        if (batch == null) return NotFound();
        return Ok(batch);
    }

    [HttpPost]
    public async Task<IActionResult> CreateBatch([FromBody] Batch batch)
    {
        _context.Batches.Add(batch);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetBatch), new { id = batch.BatchId }, batch);
    }

    [HttpPut("{id}")]
    public async Task<IActionResult> UpdateBatch(int id, [FromBody] Batch batch)
    {
        if (id != batch.BatchId) return BadRequest();

        _context.Entry(batch).State = EntityState.Modified;
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id}")]
    public async Task<IActionResult> DeleteBatch(int id)
    {
        var batch = await _context.Batches.FindAsync(id);
        if (batch == null) return NotFound();

        _context.Batches.Remove(batch);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}

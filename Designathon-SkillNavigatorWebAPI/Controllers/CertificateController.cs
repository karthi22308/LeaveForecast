using Designathon_SkillNavigatorWebAPI.Data;
using Designathon_SkillNavigatorWebAPI.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

[Route("api/[controller]")]
[ApiController]
public class CertificateController : ControllerBase
{
    private readonly SKNdbcontext _context;

    public CertificateController(SKNdbcontext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<IActionResult> GetAllCertificates()
    {
        var certificates = await _context.Certificates.Include(c => c.User).Include(c => c.Course).ToListAsync();
        return Ok(certificates);
    }

    [HttpPost]
    public async Task<IActionResult> CreateCertificate([FromBody] Certificate certificate)
    {
        _context.Certificates.Add(certificate);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetAllCertificates), certificate);
    }
}

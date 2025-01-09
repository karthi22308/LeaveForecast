using Designathon_SkillNavigatorWebAPI.Data;
using Designathon_SkillNavigatorWebAPI.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using static Designathon_SkillNavigatorWebAPI.Entities.Report;
using static Designathon_SkillNavigatorWebAPI.Entities.Report.TopperReport;

[Route("api/[controller]")]
[ApiController]
public class ReportController : ControllerBase
{
    //private readonly SKNdbcontext _context;

    //public ReportController(SKNdbcontext context)
    //{
    //    _context = context;
    //}

    //[HttpGet]
    //public async Task<IActionResult> GetAllReports()
    //{
    //    var reports = await _context.Reports.Include(r => r.User).Include(r => r.Batch).ToListAsync();
    //    return Ok(reports);
    //}

    //[HttpPost]
    //public async Task<IActionResult> CreateReport([FromBody] Report report)
    //{
    //    _context.Reports.Add(report);
    //    await _context.SaveChangesAsync();
    //    return CreatedAtAction(nameof(GetAllReports), report);
    //}
    [HttpGet("individual-candidate")]
    public IActionResult GetIndividualCandidateReport()
    {
        var report = new CandidateReport
        {
            Name = "John Doe",
            Email = "john@example.com",
            Degree = "B.Tech",
            Specialization = "Computer Science",
            Certifications = new List<string> { "AI Specialist", "ML Engineer" },
            Internships = "Google Internship",
            Completion = 85
        };
        return Ok(report);
    }

    [HttpGet("trainers-feedback")]
    public IActionResult GetTrainersFeedbackReport()
    {
        var report = new TrainersFeedbackReport
        {
            AverageScore = 4.5,
            Comments = new List<string> { "Excellent sessions", "Needs improvement in AI topic" }
        };
        return Ok(report);
    }

    [HttpGet("batch-score")]
    public IActionResult GetBatchScoreReport()
    {
        var report = new BatchScoreReport
        {
            AverageMCQ = 78.5,
            ProjectScore = 82.3
        };
        return Ok(report);
    }

    [HttpGet("college-score")]
    public IActionResult GetCollegeScoreReport()
    {
        var report = new CollegeScoreReport
        {
            TopCollege = "ABC Engineering College"
        };
        return Ok(report);
    }

    [HttpGet("topper-list")]
    public IActionResult GetTopperListReport()
    {
        var report = new TopperReport
        {
            Toppers = new List<Topper>
        {
            new Topper("Alice", 95.0),
            new Topper("Bob", 93.0)
        }
        };

        return Ok(report);
    }


    [HttpGet("batch-comparison")]
    public IActionResult GetBatchComparisonReport()
    {
        var report = new BatchComparisonReport
        {
            BestBatch = "Batch A"
        };
        return Ok(report);
    }
}
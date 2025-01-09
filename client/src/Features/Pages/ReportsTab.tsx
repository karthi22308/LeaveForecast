import React, { useState } from 'react';
import { Button, Typography, Box, Card, CardContent, CircularProgress } from '@mui/material';
import * as XLSX from 'xlsx'; // Import XLSX library

// Define types for different report data
interface CandidateReport {
    name: string;
    email: string;
    degree: string;
    specialization: string;
    certifications: string[];
    internships: string;
    completion: number;
}

interface FeedbackReport {
    averageScore: number;
    comments: string[];
}

interface BatchScoreReport {
    averageMCQ: number;
    projectScore: number;
}

interface CollegeScoreReport {
    topCollege: string;
}

interface TopperReport {
    toppers: { name: string; score: number }[];
}

interface BatchComparisonReport {
    bestBatch: string;
}

// Union type for all possible report data
type ReportData = CandidateReport | FeedbackReport | BatchScoreReport | CollegeScoreReport | TopperReport | BatchComparisonReport;

const ReportsTab: React.FC = () => {
    const [reportData, setReportData] = useState<ReportData | null>(null);
    const [selectedReport, setSelectedReport] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const fetchReportData = async (type: string) => {
        setLoading(true);
        setError(null); // Reset error state before fetching
        try {
            const response = await fetch(`https://localhost:7181/api/Report/${type}`);
            if (!response.ok) {
                throw new Error('Failed to fetch report data');
            }
            const data = await response.json();
            setReportData(data);
            setSelectedReport(type);
        } catch (error: unknown) {
            if (error instanceof Error) {
                setError(`Error fetching ${type} report: ${error.message}`);
            } else {
                setError('An unknown error occurred.');
            }
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    // Type narrowing for reports
    const isCandidateReport = (data: ReportData): data is CandidateReport => {
        return (data as CandidateReport).name !== undefined;
    };

    const isFeedbackReport = (data: ReportData): data is FeedbackReport => {
        return (data as FeedbackReport).averageScore !== undefined;
    };

    const isBatchScoreReport = (data: ReportData): data is BatchScoreReport => {
        return (data as BatchScoreReport).averageMCQ !== undefined;
    };

    const isCollegeScoreReport = (data: ReportData): data is CollegeScoreReport => {
        return (data as CollegeScoreReport).topCollege !== undefined;
    };

    const isTopperReport = (data: ReportData): data is TopperReport => {
        return (data as TopperReport).toppers !== undefined;
    };

    const isBatchComparisonReport = (data: ReportData): data is BatchComparisonReport => {
        return (data as BatchComparisonReport).bestBatch !== undefined;
    };

    const renderReportContent = () => {
        if (loading) {
            return <CircularProgress />;
        }

        if (error) {
            return <Typography color="error">{error}</Typography>;
        }

        if (!reportData) {
            return null; // Do not render if no data is present
        }

        switch (selectedReport) {
            case 'individual-candidate':
                if (isCandidateReport(reportData)) {
                    return (
                        <Card>
                            <CardContent>
                                <Typography variant="h6">Individual Candidate Report</Typography>
                                <Typography>Name: {reportData.name}</Typography>
                                <Typography>Email: {reportData.email}</Typography>
                                <Typography>Degree: {reportData.degree}</Typography>
                                <Typography>Specialization: {reportData.specialization}</Typography>
                                <Typography>Certifications: {reportData.certifications.join(', ')}</Typography>
                                <Typography>Internship Details: {reportData.internships}</Typography>
                                <Typography>Self-reported Course Completion: {reportData.completion}%</Typography>
                            </CardContent>
                        </Card>
                    );
                }
                break;

            case 'trainers-feedback':
                if (isFeedbackReport(reportData)) {
                    return (
                        <Card>
                            <CardContent>
                                <Typography variant="h6">Trainers' Feedback Analysis Report</Typography>
                                <Typography>Average Feedback Score: {reportData.averageScore}</Typography>
                                <Typography>Top Comments: {reportData.comments.join('; ')}</Typography>
                            </CardContent>
                        </Card>
                    );
                }
                break;

            case 'batch-score':
                if (isBatchScoreReport(reportData)) {
                    return (
                        <Card>
                            <CardContent>
                                <Typography variant="h6">Batch-wise Score Card</Typography>
                                <Typography>Average MCQ Score: {reportData.averageMCQ}</Typography>
                                <Typography>Project Evaluation Score: {reportData.projectScore}</Typography>
                            </CardContent>
                        </Card>
                    );
                }
                break;

            case 'college-score':
                if (isCollegeScoreReport(reportData)) {
                    return (
                        <Card>
                            <CardContent>
                                <Typography variant="h6">College-wise Score Card</Typography>
                                <Typography>Top Performing College: {reportData.topCollege}</Typography>
                            </CardContent>
                        </Card>
                    );
                }
                break;

            case 'topper-list':
                if (isTopperReport(reportData)) {
                    return (
                        <Card>
                            <CardContent>
                                <Typography variant="h6">Topper List</Typography>
                                {reportData.toppers.map((topper, index) => (
                                    <Typography key={index}>{topper.name}: {topper.score}</Typography>
                                ))}
                            </CardContent>
                        </Card>
                    );
                }
                break;

            case 'batch-comparison':
                if (isBatchComparisonReport(reportData)) {
                    return (
                        <Card>
                            <CardContent>
                                <Typography variant="h6">Batch Comparison Report</Typography>
                                <Typography>Best Performing Batch: {reportData.bestBatch}</Typography>
                            </CardContent>
                        </Card>
                    );
                }
                break;

            default:
                return null; // No fallback message
        }
    };

    // Function to export the report as an Excel file
    const exportToExcel = (data: ReportData, reportType: string) => {
        let ws: XLSX.WorkSheet;
        let wb: XLSX.WorkBook = XLSX.utils.book_new();

        switch (reportType) {
            case 'individual-candidate':
                if (isCandidateReport(data)) {
                    const candidateData = [
                        ['Name', 'Email', 'Degree', 'Specialization', 'Certifications', 'Internships', 'Completion'],
                        [
                            data.name,
                            data.email,
                            data.degree,
                            data.specialization,
                            data.certifications.join(', ') || 'No certifications available',
                            data.internships || 'No internship data',
                            data.completion || 0
                        ]
                    ];
                    ws = XLSX.utils.aoa_to_sheet(candidateData);
                    XLSX.utils.book_append_sheet(wb, ws, 'Individual Candidate Report');
                }
                break;

            case 'trainers-feedback':
                if (isFeedbackReport(data)) {
                    const feedbackData = [
                        ['Average Score', 'Comments'],
                        [data.averageScore, data.comments.join('; ') || 'No comments available']
                    ];
                    ws = XLSX.utils.aoa_to_sheet(feedbackData);
                    XLSX.utils.book_append_sheet(wb, ws, 'Trainers Feedback');
                }
                break;

            case 'batch-score':
                if (isBatchScoreReport(data)) {
                    const batchScoreData = [
                        ['Average MCQ Score', 'Project Score'],
                        [data.averageMCQ, data.projectScore]
                    ];
                    ws = XLSX.utils.aoa_to_sheet(batchScoreData);
                    XLSX.utils.book_append_sheet(wb, ws, 'Batch Score Report');
                }
                break;

            case 'college-score':
                if (isCollegeScoreReport(data)) {
                    const collegeScoreData = [
                        ['Top Performing College'],
                        [data.topCollege]
                    ];
                    ws = XLSX.utils.aoa_to_sheet(collegeScoreData);
                    XLSX.utils.book_append_sheet(wb, ws, 'College Score Report');
                }
                break;

            case 'topper-list':
                if (isTopperReport(data)) {
                    const topperData = [
                        ['Name', 'Score'],
                        ...data.toppers.map((topper) => [topper.name, topper.score])
                    ];
                    ws = XLSX.utils.aoa_to_sheet(topperData);
                    XLSX.utils.book_append_sheet(wb, ws, 'Topper List');
                }
                break;

            case 'batch-comparison':
                if (isBatchComparisonReport(data)) {
                    const batchComparisonData = [
                        ['Best Performing Batch'],
                        [data.bestBatch]
                    ];
                    ws = XLSX.utils.aoa_to_sheet(batchComparisonData);
                    XLSX.utils.book_append_sheet(wb, ws, 'Batch Comparison');
                }
                break;

            default:
                return;
        }

        // Generate and download the Excel file
        XLSX.writeFile(wb, `${reportType}_Report.xlsx`);
    };

    return (
        <Box p={2}>
            <Typography variant="h5" gutterBottom>Reports Dashboard</Typography>
            <Box display="flex" flexWrap="wrap" gap={2} mb={4}>
                <Button onClick={() => fetchReportData('individual-candidate')} variant="outlined">Individual Candidate Report</Button>
                <Button onClick={() => fetchReportData('trainers-feedback')} variant="outlined">Trainers Feedback Analysis</Button>
                <Button onClick={() => fetchReportData('batch-score')} variant="outlined">Batch-wise Score Card</Button>
                <Button onClick={() => fetchReportData('college-score')} variant="outlined">College-wise Score Card</Button>
                <Button onClick={() => fetchReportData('topper-list')} variant="outlined">Topper List</Button>
                <Button onClick={() => fetchReportData('batch-comparison')} variant="outlined">Batch Comparison</Button>
            </Box>

            {/* Report Content Section */}
            {renderReportContent()}

            {/* Add Export Buttons */}
            {reportData && (
                <Box mt={2}>
                    <Button
                        variant="contained"
                        onClick={() => exportToExcel(reportData, selectedReport)}
                    >
                        Download {selectedReport} as Excel
                    </Button>
                </Box>
            )}
        </Box>
    );
};

export default ReportsTab;
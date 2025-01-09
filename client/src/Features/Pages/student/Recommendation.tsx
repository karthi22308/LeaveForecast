import  { useState, useEffect } from 'react';
import { Box, Typography, Button, CircularProgress, List, ListItem, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import jsPDF from 'jspdf';

const AIRecommendations = () => {
    const [scores, setScores] = useState({ mcq_score: 0, project_score: 0 });
    const [strengthAreas, setStrengthAreas] = useState<string[]>([]);
    const [weakAreas, setWeakAreas] = useState<string[]>([]);
    const [recommendedCourses, setRecommendedCourses] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);

    // Dummy data for now
    const dummyData = {
        mcq_score: 85,
        project_score: 92,
        strength_areas: ['React', 'Git', 'Problem Solving'],
        weak_areas: ['Python', 'Communication'],
        recommended_courses: [
            'Java for Beginners',
            'Advanced Git',
            'Git for Beginners',
            'GenAi Suggestions:',
            `Based on the candidate's strengths and weaknesses, it seems they have a good foundation in Git but need to improve their skills in SQL and Python. Here are some Udemy course recommendations along with an improvement plan:`,
            'Udemy Course Recommendations\nFor SQL:\n"Complete SQL Bootcamp 2023: Go from Zero to Hero" - A comprehensive course that covers the basics of SQL, database design, and advanced querying techniques.\n"SQL for Data Science" - Focuses on using SQL for data analysis and is suitable for those looking to apply SQL in real-world scenarios.\n"SQL Fundamentals: Learn SQL Basics from Scratch" - A beginner-friendly course that starts with the basics of SQL syntax and gradually moves into more complex queries.',
            'For Python:\n"Complete Python Developer in 2023: Zero to Mastery" - Covers everything from basic syntax to advanced topics like web development and data science applications.\n"Automate the Boring Stuff with Python Programming" - Focuses on practical applications of Python, which can help reinforce concepts through real-world examples.\n"Python for Data Science and Machine Learning Bootcamp" - Aimed at those interested in data science, this course covers essential libraries like Pandas, NumPy, Matplotlib, etc.',
            `Improvement Plan\nStructured Learning Path:\nDedicate specific hours each week (e.g., 5-10 hours) to complete one course at a time.\nStart with foundational courses before moving on to intermediate or advanced topics.\nPractice Projects:\nImplement small projects using Git for version control.\nExample: Create a personal portfolio website or a simple application that interacts with a database using both SQL and Python.\nRegular Assessments:\nAfter completing each module or section of a course, take quizzes or practice exercises to reinforce learning.\nJoin Online Communities:\nEngage with platforms like Stack Overflow or Reddit communities related to programming where they can ask questions and share knowledge.\nPair Programming / Study Groups:\nFind peers who are also learning these technologies; pair programming can enhance understanding through collaboration.\nBuild Real-World Applications:\nOnce comfortable with the basics, try building small applications that require both backend (using Python) and database interactions (using SQL).\nFeedback Loop:\nSeek feedback from mentors or peers on projects developed during this learning phase; constructive criticism will help identify areas needing further improvement.\nSet Milestones:\nEstablish short-term goals (e.g., completing one course per month) as well as long-term goals (e.g., building a full-stack application by year-end).`
        ]
    };

    // Simulate API call on component mount
    useEffect(() => {
        setLoading(true);
        setTimeout(() => {
            // Simulating data fetching with dummy data
            setScores({
                mcq_score: dummyData.mcq_score,
                project_score: dummyData.project_score,
            });
            setStrengthAreas(dummyData.strength_areas);
            setWeakAreas(dummyData.weak_areas);
            setRecommendedCourses(dummyData.recommended_courses);
            setLoading(false);
        }, 1000); // Simulate API delay
    }, []);

    const generatePDF = () => {
        const doc = new jsPDF();
        doc.text('AI Recommendations Report', 20, 20);
        doc.text(`MCQ Score: ${scores.mcq_score}`, 20, 40);
        doc.text(`Project Score: ${scores.project_score}`, 20, 50);
        doc.text('Strength Areas:', 20, 70);
        strengthAreas.forEach((area, index) => {
            doc.text(`- ${area}`, 30, 80 + index * 10);
        });
        doc.text('Weak Areas:', 20, 110);
        weakAreas.forEach((area, index) => {
            doc.text(`- ${area}`, 30, 120 + index * 10);
        });
        doc.text('Recommended Courses:', 20, 150);
        recommendedCourses.forEach((course, index) => {
            doc.text(`- ${course}`, 30, 160 + index * 10);
        });
        doc.save('AI_Recommendations.pdf');
    };

    return (
        <Box sx={{ padding: '20px', maxWidth: '900px', margin: 'auto' }}>
            <Typography variant="h4" gutterBottom align="center" sx={{ marginBottom: '30px' }}>
                AI Recommendations
            </Typography>
            {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <CircularProgress />
                </Box>
            ) : (
                <Box>
                    {/* MCQ and Project Scores */}
                    <Accordion sx={{ marginBottom: '10px' }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography variant="h6">Scores</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                            <Typography>MCQ Score: {scores.mcq_score}</Typography>
                            <Typography>Project Score: {scores.project_score}</Typography>
                        </AccordionDetails>
                    </Accordion>

                    {/* Strength Areas */}
                    <Accordion sx={{ marginBottom: '10px' }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography variant="h6">Strength Areas</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                            <List>
                                {strengthAreas.map((area, index) => (
                                    <ListItem key={index}>{area}</ListItem>
                                ))}
                            </List>
                        </AccordionDetails>
                    </Accordion>

                    {/* Weak Areas */}
                    <Accordion sx={{ marginBottom: '10px' }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography variant="h6">Weak Areas</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                            <List>
                                {weakAreas.map((area, index) => (
                                    <ListItem key={index}>{area}</ListItem>
                                ))}
                            </List>
                        </AccordionDetails>
                    </Accordion>

                    {/* Recommended Courses */}
                    <Accordion sx={{ marginBottom: '10px' }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography variant="h6">Recommended suggestions</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                            <List>
                                {recommendedCourses.map((course, index) => (
                                    <ListItem key={index}>{course}</ListItem>
                                ))}
                            </List>
                        </AccordionDetails>
                    </Accordion>

                    {/* Download PDF Button */}
                    <Box sx={{ textAlign: 'center' }}>
                        <Button variant="contained" color="secondary" onClick={generatePDF} sx={{ marginTop: '20px' }}>
                            Download PDF
                        </Button>
                    </Box>
                </Box>
            )}
        </Box>
    );
};

export default AIRecommendations;

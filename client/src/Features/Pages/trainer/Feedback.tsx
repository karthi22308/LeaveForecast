import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Button,
    List,
    ListItem,
    CircularProgress,
    Paper,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Grid,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { jsPDF } from 'jspdf';
import { Bar } from 'react-chartjs-2';
import 'chart.js/auto';

const AIFeedback: React.FC = () => {
    const [feedbackTexts, setFeedbackTexts] = useState<string[]>([]);
    const [apiResponse, setApiResponse] = useState<any>(null);
    const [loading, setLoading] = useState<boolean>(false);

    useEffect(() => {
        fetchFeedbackData();
    }, []);

    const fetchFeedbackData = async () => {
        setLoading(true);
        try {
            // Simulate an API call to fetch feedback
            const dummyResponse = {
                sentiments: ['Positive', 'Negative'],
                topics: [
                    'Topic 1: application, process, seamless',
                    'Topic 2: system, slow, frustration',
                ],
                insights: {
                    'Positive Feedback': 1,
                    'Negative Feedback': 1,
                    'Neutral Feedback': 0,
                    'Top Suggestions': [
                        'Improvement Plan for Trainer\n\n' +
                        '1. Website Navigation\n' +
                        'Action: Conduct a usability audit of the website.\n' +
                        'Implementation: Collaborate with web developers to identify and address navigation issues.\n' +
                        'Outcome: Simplified navigation that allows users to find information quickly and easily.\n\n' +
                        '2. File Upload System\n' +
                        'Action: Review and upgrade the file upload system.\n' +
                        'Implementation:\n' +
                        ' - Implement drag-and-drop functionality for easier uploads.\n' +
                        ' - Ensure compatibility with various file formats and sizes.\n' +
                        ' - Provide clear instructions on acceptable file types and size limits.\n' +
                        'Outcome: A more user-friendly upload process that minimizes errors and frustrations.\n\n' +
                        '3. System Performance\n' +
                        'Action: Optimize website performance to reduce loading times.\n' +
                        'Implementation:\n' +
                        ' - Analyze current server performance and consider upgrading hosting solutions if necessary.\n' +
                        ' - Optimize images, scripts, and other resources to improve load times.\n' +
                        ' - Regularly monitor system performance metrics to identify bottlenecks.\n' +
                        'Outcome: A faster, more responsive website that enhances user experience.\n\n' +
                        '4. Feedback Mechanism\n' +
                        'Action: Establish a continuous feedback loop with participants post-training sessions.\n' +
                        'Implementation:\n' +
                        ' - Create a brief survey focusing on specific areas such as navigation, document upload, and overall satisfaction after each session.\n' +
                        ' - Encourage open-ended feedback for additional suggestions or concerns not covered in surveys.\n' +
                        'Outcome: Ongoing insights into participant experiences that can guide future improvements.\n\n' +
                        '5. Training & Support Resources\n' +
                        'Action: Develop comprehensive training materials or FAQs addressing common issues faced by users (e.g., uploading documents).\n' +
                        'Implementation:\n' +
                        ' - Create video tutorials or step-by-step guides on how to navigate the website effectively and use its features (like document uploads).\n' +
                        ' - Offer live chat support during peak usage times for immediate assistance with technical issues.\n' +
                        'Outcome: Increased confidence among participants in using the platform effectively.\n\n' +
                        'Conclusion\n' +
                        'By implementing these improvements, the trainer can significantly enhance user experience, reduce frustration related to technical issues, and ultimately foster a more positive learning environment. Regularly revisiting these strategies based on ongoing feedback will ensure continuous improvement in training delivery.'
                    ]
                },
            };
            
            setFeedbackTexts(dummyResponse.topics);
            setApiResponse(dummyResponse);
        } catch (error) {
            console.error('Error fetching feedback:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleDownloadPDF = () => {
        if (!apiResponse) return;

        const doc = new jsPDF();
        doc.setFontSize(12);
        doc.text('AI Feedback Analysis', 10, 10);

        doc.text('Sentiments:', 10, 20);
        apiResponse.sentiments.forEach((sentiment: string, index: number) => {
            doc.text(`${index + 1}. ${sentiment}`, 10, 30 + index * 10);
        });

        doc.text('Topics:', 10, 40 + apiResponse.sentiments.length * 10);
        apiResponse.topics.forEach((topic: string, index: number) => {
            doc.text(`${index + 1}. ${topic}`, 10, 50 + apiResponse.sentiments.length * 10 + index * 10);
        });

        doc.text('Insights:', 10, 60 + apiResponse.sentiments.length * 20);
        Object.entries(apiResponse.insights).forEach(([key, value], index) => {
            const yOffset = 70 + apiResponse.sentiments.length * 20 + index * 10;
            if (Array.isArray(value)) {
                doc.text(`${key}:`, 10, yOffset);
                value.forEach((item: string, i: number) => {
                    doc.text(`- ${item}`, 20, yOffset + (i + 1) * 10);
                });
            } else {
                doc.text(`${key}: ${value}`, 10, yOffset);
            }
        });

        doc.save('AI_Feedback_Analysis.pdf');
    };

    const chartData = {
        labels: ['Positive', 'Negative', 'Neutral'],
        datasets: [
            {
                label: 'Feedback Sentiments',
                data: apiResponse
                    ? [
                          apiResponse.insights['Positive Feedback'],
                          apiResponse.insights['Negative Feedback'],
                          apiResponse.insights['Neutral Feedback'],
                      ]
                    : [],
                backgroundColor: ['#4caf50', '#f44336', '#ff9800'],
            },
        ],
    };

    return (
        <Box sx={{ padding: '30px', backgroundColor: '#f5f5f5' }}>
            <Typography variant="h4" gutterBottom align="center" sx={{ fontWeight: 'bold', marginBottom: '30px' }}>
                AI Feedback Analysis
            </Typography>

            {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                    <CircularProgress />
                </Box>
            ) : (
                <Paper elevation={3} sx={{ padding: '20px', borderRadius: '8px' }}>
                    <Accordion sx={{ marginBottom: '10px' }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography variant="h6">Feedback Topics</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                            <List sx={{ paddingLeft: '20px' }}>
                                {feedbackTexts.map((text, index) => (
                                    <ListItem key={index} sx={{ padding: '8px 0' }}>
                                        {text}
                                    </ListItem>
                                ))}
                            </List>
                        </AccordionDetails>
                    </Accordion>

                    {apiResponse && (
                        <>
                            {/* Sentiment Analysis */}
                            <Accordion sx={{ marginBottom: '10px' }}>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography variant="h6">Sentiments</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Typography variant="body1">
                                        Sentiments: {apiResponse.sentiments.join(', ')}
                                    </Typography>
                                </AccordionDetails>
                            </Accordion>

                            {/* Topics */}
                            <Accordion sx={{ marginBottom: '10px' }}>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography variant="h6">Topics</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <List sx={{ paddingLeft: '20px' }}>
                                        {apiResponse.topics.map((topic: string, index: number) => (
                                            <ListItem key={index} sx={{ padding: '8px 0' }}>
                                                {topic}
                                            </ListItem>
                                        ))}
                                    </List>
                                </AccordionDetails>
                            </Accordion>

                            {/* Graph Section */}
                            <Box sx={{ marginTop: '20px', maxWidth: '600px', margin: '0 auto' }}>
                                <Bar data={chartData} options={{ responsive: true }} />
                            </Box>

                            {/* Insights */}
                            <Accordion sx={{ marginBottom: '10px' }}>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography variant="h6">Insights</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Box>
                                        <Typography variant="body1" sx={{ marginBottom: '10px' }}>
                                            Positive Feedback: {apiResponse.insights['Positive Feedback']}
                                        </Typography>
                                        <Typography variant="body1" sx={{ marginBottom: '10px' }}>
                                            Negative Feedback: {apiResponse.insights['Negative Feedback']}
                                        </Typography>
                                        <Typography variant="body1" sx={{ marginBottom: '10px' }}>
                                            Neutral Feedback: {apiResponse.insights['Neutral Feedback']}
                                        </Typography>
                                        <Typography variant="body1" sx={{ marginBottom: '10px' }}>
                                            Top Suggestions:
                                        </Typography>
                                        {apiResponse.insights['Top Suggestions'].map((suggestion: string, index: number) => (
                                            <Typography key={index} variant="body2" sx={{ paddingLeft: '20px' }}>
                                                - {suggestion}
                                            </Typography>
                                        ))}
                                    </Box>
                                </AccordionDetails>
                            </Accordion>

                            {/* Download PDF Button */}
                            <Grid container spacing={2} sx={{ marginTop: '20px' }}>
                                <Grid item xs={12} sm={6} sx={{ display: 'flex', justifyContent: 'center' }}>
                                    <Button
                                        variant="contained"
                                        color="success"
                                        onClick={handleDownloadPDF}
                                        sx={{
                                            width: '100%',
                                            padding: '10px',
                                            fontWeight: 'bold',
                                            borderRadius: '5px',
                                        }}
                                    >
                                        Download PDF
                                    </Button>
                                </Grid>
                            </Grid>
                        </>
                    )}
                </Paper>
            )}
        </Box>
    );
};

export default AIFeedback;

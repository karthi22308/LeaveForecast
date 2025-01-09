import React, { useState } from 'react';
import {
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Typography,
    Button,
    Box,
    LinearProgress,
    TextField,
    List,
    ListItem,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

interface Module {
    id: number;
    name: string;
    completed: boolean;
    score?: number; // Added score for the module
}

interface Course {
    id: number;
    name: string;
    trainer: string; // Trainer name for the course
    modules: Module[];
    progress: number;
    completed: boolean;
    review?: string; // Optional review for the course
}

const StudentAssignments: React.FC = () => {
    const [courses, setCourses] = useState<Course[]>([
        {
            id: 1,
            name: 'React Development',
            trainer: 'John Doe',
            modules: [
                { id: 101, name: 'Introduction to React', completed: false },
                { id: 102, name: 'React State Management', completed: false },
                { id: 103, name: 'Advanced React Patterns', completed: false },
            ],
            progress: 0,
            completed: false,
        },
        {
            id: 2,
            name: 'Python for Data Science',
            trainer: 'Jane Smith',
            modules: [
                { id: 201, name: 'Python Basics', completed: false },
                { id: 202, name: 'Data Analysis with Pandas', completed: false },
                { id: 203, name: 'Machine Learning Basics', completed: false },
            ],
            progress: 0,
            completed: false,
        },
    ]);

    const [score, setScore] = useState<number | string>(''); // Manage score as a string initially for validation
    const [selectedModule, setSelectedModule] = useState<number | null>(null); // Track selected module for score input
    const [error, setError] = useState<string>(''); // Error message for validation
    const [review, setReview] = useState<string>(''); // Review input

    const toggleModuleCompletion = (_: number, moduleId: number) => {
        // When a module is marked as complete, show score input
        setSelectedModule(moduleId);
    };

    const handleScoreChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setScore(e.target.value);
    };

    const submitScore = (courseId: number, moduleId: number) => {
        if (!score) {
            setError('Please enter a score to complete the module.');
            return;
        }

        setError(''); // Clear error message if score is provided

        setCourses((prevCourses) =>
            prevCourses.map((course) =>
                course.id === courseId
                    ? {
                          ...course,
                          modules: course.modules.map((module) =>
                              module.id === moduleId
                                  ? {
                                        ...module,
                                        completed: true,
                                        score: parseInt(score as string, 10), // Assign score to module
                                  }
                                  : module
                          ),
                          progress: calculateProgress(course.modules),
                          completed: course.modules.every((module) => module.completed), // Check if all modules are complete
                      }
                    : course
            )
        );
        setSelectedModule(null); // Reset selected module after submitting score
        setScore(''); // Clear score field
    };

    const calculateProgress = (modules: Module[]): number => {
        const completedModules = modules.filter((module) => module.completed).length;
        return Math.round((completedModules / modules.length) * 100);
    };

    const handleReviewChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setReview(e.target.value);
    };

    const submitReview = (courseId: number) => {
        // Submit review for the course (you can handle API call here)
        console.log(`Course ID: ${courseId}, Review: ${review}`);
        alert('Review submitted successfully!');
        setReview(''); // Reset review field
    };

    const handleCourseCompletion = (courseId: number) => {
        const course = courses.find((course) => course.id === courseId);
        if (course && course.modules.every((module) => module.completed)) {
            // If all modules are completed, mark the course as completed
            setCourses((prevCourses) =>
                prevCourses.map((course) =>
                    course.id === courseId
                        ? {
                              ...course,
                              completed: true,
                          }
                        : course
                )
            );
        } else {
            alert('Please complete all modules before marking the course as completed.');
        }
    };

    return (
        <Box sx={{ padding: '20px' }}>
            <Typography variant="h4" gutterBottom>
                Student Assignments
            </Typography>
            {courses.map((course) => (
                <Accordion key={course.id} sx={{ marginBottom: '10px' }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="h6">
                            {course.name} - Trainer: {course.trainer}
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <List>
                            {course.modules.map((module) => (
                                <ListItem
                                    key={module.id}
                                    sx={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                    }}
                                >
                                    <Typography>{module.name}</Typography>
                                    {module.completed ? (
                                        <Typography variant="body2" color="green">
                                            Completed - Score: {module.score}
                                        </Typography>
                                    ) : (
                                        <>
                                            <Button
                                                variant="contained"
                                                color="primary"
                                                onClick={() => toggleModuleCompletion(course.id, module.id)}
                                            >
                                                Mark as Complete
                                            </Button>

                                            {selectedModule === module.id && (
                                                <Box sx={{ marginTop: '10px' }}>
                                                    <TextField
                                                        label="Enter Score"
                                                        type="number"
                                                        value={score}
                                                        onChange={handleScoreChange}
                                                        sx={{ marginBottom: '10px' }}
                                                    />
                                                    {error && (
                                                        <Typography variant="body2" color="error">
                                                            {error}
                                                        </Typography>
                                                    )}
                                                    <Button
                                                        variant="contained"
                                                        color="secondary"
                                                        onClick={() => submitScore(course.id, module.id)}
                                                        disabled={!score}
                                                    >
                                                        Submit Score
                                                    </Button>
                                                </Box>
                                            )}
                                        </>
                                    )}
                                </ListItem>
                            ))}
                        </List>
                        <Box sx={{ marginTop: '20px' }}>
                            <Typography variant="body1">Progress: {calculateProgress(course.modules)}%</Typography>
                            <LinearProgress
                                variant="determinate"
                                value={calculateProgress(course.modules)}
                                sx={{ marginBottom: '10px' }}
                            />
                            {!course.completed && (
                                <Button
                                    variant="contained"
                                    color="primary"
                                    onClick={() => handleCourseCompletion(course.id)}
                                    sx={{ marginTop: '20px' }}
                                >
                                    Mark Course as Completed
                                </Button>
                            )}
                            {course.completed && (
                                <Box>
                                    <Typography variant="body2">Course Completed! Please provide your review:</Typography>
                                    <TextField
                                        fullWidth
                                        multiline
                                        rows={4}
                                        placeholder="Provide your review here..."
                                        value={review}
                                        onChange={handleReviewChange}
                                        sx={{ marginBottom: '10px' }}
                                    />
                                    <Button
                                        variant="contained"
                                        color="secondary"
                                        onClick={() => submitReview(course.id)}
                                        disabled={!review}
                                    >
                                        Submit Review
                                    </Button>
                                </Box>
                            )}
                        </Box>
                    </AccordionDetails>
                </Accordion>
            ))}
        </Box>
    );
};

export default StudentAssignments;

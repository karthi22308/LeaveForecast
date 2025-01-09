import React, { useState, useEffect } from 'react';
import { Accordion, AccordionSummary, AccordionDetails, Typography, List, ListItem, Box } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

interface Student {
    id: string;
    name: string;
}

interface Batch {
    id: string;
    name: string;
    students: Student[];
}

const TrainerBatches: React.FC = () => {
    const trainerId = 'T001'; // Example hardcoded trainer ID
    const [batches, setBatches] = useState<Batch[]>([]);

    useEffect(() => {
        const fetchBatches = async (_: string): Promise<void> => {
            try {
                // Simulate API call with hardcoded data
                const response = [
                    {
                        id: 'B001',
                        name: 'Batch A',
                        students: [
                            { id: 'S001', name: 'John Doe' },
                            { id: 'S002', name: 'Jane Smith' },
                        ],
                    },
                    {
                        id: 'B002',
                        name: 'Batch B',
                        students: [
                            { id: 'S003', name: 'Alice Brown' },
                            { id: 'S004', name: 'Bob White' },
                        ],
                    },
                ];
                setBatches(response);
            } catch (error) {
                console.error('Error fetching batches:', error);
            }
        };

        fetchBatches(trainerId);
    }, [trainerId]);

    return (
        <Box sx={{ padding: '20px' }}>
            <Typography variant="h4" gutterBottom>Trainer Batches</Typography>
            {batches.map(batch => (
                <Accordion key={batch.id} sx={{ marginBottom: '10px' }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="h6">{batch.name}</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <List>
                            {batch.students.map(student => (
                                <ListItem key={student.id}>
                                    <Typography>{student.name}</Typography>
                                </ListItem>
                            ))}
                        </List>
                    </AccordionDetails>
                </Accordion>
            ))}
        </Box>
    );
};

export default TrainerBatches;

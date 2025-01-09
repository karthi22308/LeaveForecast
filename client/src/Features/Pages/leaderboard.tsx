import React, { useState, useEffect } from 'react';
import { Typography, Box, Button, List, ListItem, Paper } from '@mui/material';

interface Leader {
    id: string;
    name: string;
    score: number;
}

interface Batch {
    id: string;
    name: string;
    leaders: Leader[];
}

const LeaderBoard: React.FC = () => {
    const [batches, setBatches] = useState<Batch[]>([]);
    const [selectedBatchId, setSelectedBatchId] = useState<string>('');

    useEffect(() => {
        const fetchBatches = async (): Promise<void> => {
            try {
                // Simulate API call with hardcoded data
                const dummyData = [
                    {
                        id: 'B001',
                        name: 'Batch A',
                        leaders: [
                            { id: 'L001', name: 'Alice', score: 95 },
                            { id: 'L002', name: 'Bob', score: 90 },
                            { id: 'L003', name: 'Charlie', score: 85 },
                        ],
                    },
                    {
                        id: 'B002',
                        name: 'Batch B',
                        leaders: [
                            { id: 'L004', name: 'Diana', score: 92 },
                            { id: 'L005', name: 'Eve', score: 88 },
                            { id: 'L006', name: 'Frank', score: 80 },
                        ],
                    },
                ];
                setBatches(dummyData);
                setSelectedBatchId(dummyData[0].id); // Set the first batch as the default selection
            } catch (error) {
                console.error('Error fetching batches:', error);
            }
        };

        fetchBatches();
    }, []);

    const selectedBatch = batches.find((batch) => batch.id === selectedBatchId);

    return (
        <Box sx={{ padding: '20px' }}>
            <Typography variant="h4" gutterBottom>
                Leaderboard
            </Typography>

            {/* Horizontal Nav Bar for Batches */}
            <Box sx={{ display: 'flex', gap: 2, marginBottom: '20px', overflowX: 'auto' }}>
                {batches.map((batch) => (
                    <Button
                        key={batch.id}
                        variant={batch.id === selectedBatchId ? 'contained' : 'outlined'}
                        onClick={() => setSelectedBatchId(batch.id)}
                    >
                        {batch.name}
                    </Button>
                ))}
            </Box>

            {/* Leaderboard for the Selected Batch */}
            <Paper elevation={3} sx={{ padding: '20px' }}>
                <Typography variant="h6" gutterBottom>
                    {selectedBatch ? selectedBatch.name : 'Select a Batch'}
                </Typography>
                {selectedBatch && (
                    <List>
                        {selectedBatch.leaders.map((leader) => (
                            <ListItem key={leader.id}>
                                <Typography>
                                    {leader.name} - {leader.score} points
                                </Typography>
                            </ListItem>
                        ))}
                    </List>
                )}
            </Paper>
        </Box>
    );
};

export default LeaderBoard;

import React, { useEffect, useState } from 'react';
import {
    Box,
    Typography,
    Grid,
    Card,
    CardContent,
    Tooltip,
    Select,
    MenuItem,
    Avatar,
} from '@mui/material';
import { SelectChangeEvent } from '@mui/material';
import axios from 'axios';



interface Trainer {
    trainerId: number;
    name: string;
    email: string;
    phoneNumber: string;
    specialization: string | null;
}

const API_BASE_URL = 'http://localhost:5220';

const fetchTrainers = async () => {
    console.log(`${API_BASE_URL}/fetchtrainers'`)
    const response = await axios.post(`${API_BASE_URL}/fetchtrainers`);
    return response.data;
};

const TrainerTab: React.FC = () => {
    const [trainers, setTrainers] = useState<Trainer[]>([]);
    const [filteredTrainers, setFilteredTrainers] = useState<Trainer[]>([]);
    const [specializations, setSpecializations] = useState<string[]>([]);
    const [selectedSpecialization, setSelectedSpecialization] = useState<string>('All');

    useEffect(() => {
        fetchTrainers()
        
            .then((data: Trainer[]) => {
                setTrainers(data);
                setFilteredTrainers(data);
                console.log("API Response:", data);
                const distinctSpecializations = [
                    'All',
                    ...Array.from(
                        new Set(data.map((trainer) => (trainer.specialization || 'Unknown').trim()))
                    ),
                ];
                setSpecializations(distinctSpecializations);
            })
            .catch((error: any) => console.error('Error fetching trainers:', error));
    }, []);

    const handleFilterChange = (event: SelectChangeEvent<string>) => {
        const selected = event.target.value;
        setSelectedSpecialization(selected);

        const filtered =
            selected === 'All'
                ? trainers
                : trainers.filter(
                      (trainer) => (trainer.specialization || 'Unknown').trim() === selected
                  );
        setFilteredTrainers(filtered);
    };

    return (
        <Box
            sx={{
                minHeight: '100vh',
                background: 'linear-gradient(to bottom, #FFA500, #FFFFFF)',
                padding: '20px',
                flex: 1,
            }}
        >
            <Box
                sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '20px',
                }}
            >
                <Typography variant="h4" sx={{ color: '#FFFFFF' }}>
                    Trainer List
                </Typography>

                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                 <Typography variant="h6" sx={{ color: '#FFFFFF' }}>
                                        Filter by Specialization:
                                    </Typography>
                    <Select
                        value={selectedSpecialization}
                        onChange={handleFilterChange}
                        sx={{
                            marginRight: '20px',
                            backgroundColor: '#FFFFFF',
                            color: '#FFA500',
                            minWidth: '200px',
                            borderRadius: '10px',
                        }}
                    >
                        {specializations.map((specialization, index) => (
                            <MenuItem key={index} value={specialization}>
                                {specialization}
                            </MenuItem>
                        ))}
                    </Select>
                    <Typography variant="h6" sx={{ color: '#FFFFFF' }}>
                        Total: {filteredTrainers.length}
                    </Typography>
                </Box>
            </Box>

            <Grid container spacing={3} justifyContent="center">
                {filteredTrainers.map((trainer) => (
                    <Grid item xs={12} sm={6} md={4} key={trainer.trainerId}>
                        <Tooltip
                            title={
                                <Typography variant="body2">
                                    <strong>Phone:</strong> {trainer.phoneNumber || 'N/A'}
                                </Typography>
                            }
                            arrow
                        >
                            <Card
                                sx={{
                                    borderRadius: '10px',
                                    boxShadow: '0 4px 10px rgba(0, 0, 0, 0.2)',
                                    backgroundColor: '#FFFFFF',
                                    padding: '10px',
                                    display: 'flex',
                                    alignItems: 'center',
                                }}
                            >
                                <Avatar
                                    sx={{
                                        width: 60,
                                        height: 60,
                                        marginRight: '15px',
                                        backgroundColor: '#FFA500',
                                        color: '#FFFFFF',
                                    }}
                                >
                                    {trainer.name.charAt(0)}
                                </Avatar>
                                <CardContent sx={{ flex: 1 }}>
                                    <Typography variant="h6" sx={{ color: '#FFA500' }}>
                                        {trainer.name}
                                    </Typography>
                                    <Typography variant="body1" sx={{ margin: '10px 0' }}>
                                        <strong>Email:</strong> {trainer.email}
                                    </Typography>
                                    <Typography variant="body2" sx={{ marginBottom: '5px' }}>
                                        <strong>Specialization:</strong>{' '}
                                        {trainer.specialization || 'Unknown'}
                                    </Typography>
                                </CardContent>
                            </Card>
                        </Tooltip>
                    </Grid>
                ))}
            </Grid>
        </Box>
    );
};

export default TrainerTab;

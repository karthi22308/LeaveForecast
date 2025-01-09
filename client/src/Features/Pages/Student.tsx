import React, { useState, useEffect } from 'react';
import { Box, Button, Paper, IconButton, Tooltip, Typography } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import PeopleIcon from '@mui/icons-material/People';
import SportsKabaddiIcon from '@mui/icons-material/SportsKabaddi';
import AssessmentIcon from '@mui/icons-material/Assessment';
import TopBarStudent from '../../Components/TopBarStudent';
import Assignmentstab from '../../Features/Pages/student/assignments';
import Recommendationtab from '../../Features/Pages/student/Recommendation';
import Leadertab from '../../Features/Pages/leaderboard';

const StudentPage: React.FC = () => {
    useEffect(() => {
        document.body.classList.add('body-admin');
        return () => {
            document.body.classList.remove('body-admin');
        };
    }, []);

    const [value, setValue] = useState(0);
    const [navExpanded, setNavExpanded] = useState(true);

    const navItems = [
        { label: "My Assignments", icon: <PeopleIcon />, component: <Assignmentstab /> },
        { label: "Leaderboard", icon: <SportsKabaddiIcon />, component: <Leadertab /> },
        { label: "AI Recommendations", icon: <AssessmentIcon />, component: <Recommendationtab /> }
        
    ];

    return (
        <div>
            <TopBarStudent />
            <Box sx={{ display: 'flex' }}>
                <Paper
                    elevation={3}
                    sx={{
                        minWidth: navExpanded ? '200px' : '100px',  // Consistent minWidth for the sidebar
                        width: navExpanded ? '200px' : '100px',
                        transition: 'width 0.3s',
                        overflow: 'hidden',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'flex-start',
                        height: '100vh',
                        bgcolor: 'rgba(255, 255, 255, 0.7)'
                    }}
                >
                    <IconButton onClick={() => setNavExpanded(!navExpanded)} sx={{ alignSelf: 'flex-end', display: 'flex', alignItems: 'center' }}>
                        <MenuIcon />
                        {!navExpanded && <Typography variant="caption" sx={{ ml: 1 }}>Menu</Typography>}
                    </IconButton>
                    {navItems.map((item, index) => (
                        <Tooltip title={navExpanded ? '' : item.label} key={item.label}>
                            <Button
                                variant={value === index ? 'contained' : 'outlined'}
                                color="primary"
                                onClick={() => setValue(index)}
                                sx={{ justifyContent: 'center', width: '100%', flexDirection: 'column' }}
                                startIcon={item.icon}
                            >
                                {!navExpanded && <Typography variant="caption">{item.label}</Typography>}
                                {navExpanded && item.label}
                            </Button>
                        </Tooltip>
                    ))}
                </Paper>
                <Box sx={{ flexGrow: 1 }}>
                    {navItems[value].component}
                </Box>
            </Box>
        </div>
    );
};

export default StudentPage;

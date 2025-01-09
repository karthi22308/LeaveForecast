import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import { useNavigate } from 'react-router-dom'; // Import useNavigate

const TopBar: React.FC = () => {
  const navigate = useNavigate(); // Hook for navigation

  const handleLogout = () => {
    // Add your logout logic here, such as clearing user session data
    console.log('Logout clicked');
    navigate('/login'); // Navigate to login page
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Skill Navigator Application Candidate
        </Typography>
        <Button color="inherit" onClick={handleLogout}>
          Logout
        </Button>
      </Toolbar>
    </AppBar>
  );
};

export default TopBar;

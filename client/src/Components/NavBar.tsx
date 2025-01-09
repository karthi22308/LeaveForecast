import React from 'react';
import { Box, Button } from '@mui/material';

interface NavBarProps {
  currentPage: 'login' | 'register';
}

const NavBar: React.FC<NavBarProps> = ({ currentPage }) => (
  <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, margin: 2 }}>
    <Button
      variant={currentPage === 'login' ? 'contained' : 'outlined'}
      color="primary"
      href="/login"
    >
      Login
    </Button>
    <Button
      variant={currentPage === 'register' ? 'contained' : 'outlined'}
      color="primary"
      href="/register"
    >
      Register
    </Button>
  </Box>
);

export default NavBar;

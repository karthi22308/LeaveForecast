import React, { useState, useEffect } from 'react';
import { Box, TextField, Button, Typography, CircularProgress } from '@mui/material';
import { validateUser } from './api';
import { useNavigate } from 'react-router-dom';
import Topbar from '../../Components/TopBar'; 
import Navbar from '../../Components/NavBar';


const Login: React.FC = () => {
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  useEffect(() => {
    document.body.classList.add('body-login');
    return () => {
        document.body.classList.remove('body-login');
    };
}, []);
  const navigate = useNavigate();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setCredentials({ ...credentials, [name]: value });
    
    
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    const data = await validateUser(credentials.username, credentials.password);
    try {
      
      console.log('Login successful:', data);
      console.log('Login successful:', data); // Check what data looks like
      console.log('Role received:', data.Role); 
      if (data.role === 'Admin') {
        navigate('/admin'); // Redirect to admin page
      } else if (data.role === 'Student') {
        navigate('/student'); // Redirect to student page
      }else if (data.role === 'Trainer') {
          navigate('/trainer'); // Redirect to student page
      } else {
        throw new Error('Unauthorized role');
      }
    } catch (err) {
      console.error('Login failed:', err);
      setError('Invalid username or password'+err+data);
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = () => {
    navigate('/register'); // Redirect to register page
  };

  return (
    <div>
    <Topbar />
    <Navbar currentPage="login"/>
    <Box sx={{ maxWidth: 400,
      margin: 'auto',
      textAlign: 'center',
      padding: 4,
      position: 'relative',
      left: '20%'}}>
      <Typography variant="h4" gutterBottom>
        Skill Navigator Login
      </Typography>
      {error && <Typography color="error" gutterBottom>{error}</Typography>}
      <form onSubmit={handleSubmit}>
        <TextField
          fullWidth
          margin="normal"
          label="Username"
          name="username"
          value={credentials.username}
          onChange={handleChange}
        />
        <TextField
          fullWidth
          margin="normal"
          label="Password"
          type="password"
          name="password"
          value={credentials.password}
          onChange={handleChange}
        />
        <Button type="submit" variant="contained" color="primary" fullWidth disabled={loading}>
          {loading ? <CircularProgress size={24} /> : 'Login'}
        </Button>
      </form>
      <Button 
        variant="outlined" 
        color="secondary" 
        fullWidth 
        sx={{ mt: 2 }} 
        onClick={handleRegister}
      >
        Register
      </Button>
    </Box>
    </div>
  );
};

export default Login;




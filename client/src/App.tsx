import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Login from './Features/auth/Login';
import Register from './Features/auth/Register';
import Student from './Features/Pages/Student';
import Admin from './Features/Pages/admin';
import Trainer from './Features/Pages/trainer';
//import Dashboard from './Features/dashboard/Dashboard';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        {/* Redirect root to Login */}
        <Route path="/" element={<Navigate to="/login" />} />
        
        {/* Define routes */}
        <Route path="/Login" element={<Login />} />
        <Route path="/Register" element={<Register />} />
        <Route path="/Student" element={<Student />} /> 
        <Route path="/admin" element={<Admin />} /> 
        <Route path="/trainer" element={<Trainer />} /> 
      </Routes>
    </Router>
  );
};

export default App;

import React,{useEffect} from 'react';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import '../../index.css';
import {
  Box,
  TextField,
  Button,
  Grid,
  Typography,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Checkbox,
  ListItemText,
} from '@mui/material';
import { registerUser } from './api';
import { useNavigate } from 'react-router-dom';
import Topbar from '../../Components/TopBar'; // Adjust the path as necessary
import Navbar from '../../Components/NavBar';

const Register: React.FC = () => {
    useEffect(() => {
      document.body.classList.add('body-login');
      return () => {
          document.body.classList.remove('body-login');
      };
  }, []);
  const formik = useFormik({
    initialValues: {
      username: '',
      password: '',
      email: '',
      role: '',
      degree: '',
      specialization: '',
      phoneNumber: '',
      programmingLanguages: [] as string[], // Updated field name
      internshipDetails: '',
      coursesCompleted: '',
      linkedInProfileLink: '',
      certifications: '',
    },
    validationSchema: Yup.object({
      username: Yup.string().required('Required'),
      password: Yup.string().min(8, 'Password must be at least 8 characters').required('Required'),
      email: Yup.string().email('Invalid email').required('Required'),
      phoneNumber: Yup.string().matches(/^\d{10}$/, 'Phone number must be 10 digits'),
    }),
    onSubmit: async (values) => {
      try {
        const valuesToSubmit = {
          ...values,
          programmingLanguages: values.programmingLanguages.join(','),
        };
        const data = await registerUser(valuesToSubmit);
        console.log('Registration successful:', data);
        alert('Registration Successful! Please login.');
      } catch (err) {
        console.error('Registration failed:', err);
        alert('Registration failed. Please try again.' + err);
      }
    },
  });

  const navigate = useNavigate();
  const handleRegister = () => {
    navigate('/login'); // Redirect to login page
  };

  const languageOptions = ['Java', 'C++', 'C', 'Python'];
  const internshipOptions = ['TCS', 'Accenture', 'Infosys', 'Wipro', 'Cognizant', 'IBM', 'Capgemini', 'HCL', 'Tech Mahindra', 'Oracle'];
  const degreeOptions = ['B.E', 'B.Sc', 'M.Sc', 'M.Tech', 'Ph.D'];
  const specializationOptions = ['CSE', 'ECE', 'Mech', 'EEE', 'Civil'];

  return (
    <div>
      <Topbar />
      <Navbar currentPage="register" />
      <Box sx={{ maxWidth: 800, margin: 'auto', textAlign: 'center', padding: 4, position: 'relative', left: '20%' }}>
        <Typography variant="h4" gutterBottom>
          Register
        </Typography>
        <form onSubmit={formik.handleSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Username"
                name="username"
                value={formik.values.username}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                helperText={formik.touched.username && formik.errors.username}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Password"
                type="password"
                name="password"
                value={formik.values.password}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                helperText={formik.touched.password && formik.errors.password}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Email"
                name="email"
                value={formik.values.email}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                helperText={formik.touched.email && formik.errors.email}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Phone Number"
                name="phoneNumber"
                value={formik.values.phoneNumber}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                helperText={formik.touched.phoneNumber && formik.errors.phoneNumber}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel htmlFor="role-select">Role</InputLabel>
                <Select
                  id="role-select"
                  name="role"
                  value={formik.values.role}
                  onChange={formik.handleChange}
                  onBlur={formik.handleBlur}
                >
                  <MenuItem value="Trainer">Trainer</MenuItem>
                  <MenuItem value="Student">Student</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel htmlFor="degree-select">Degree</InputLabel>
                <Select
                  id="degree-select"
                  name="degree"
                  value={formik.values.degree}
                  onChange={formik.handleChange}
                  onBlur={formik.handleBlur}
                >
                  {degreeOptions.map((degree) => (
                    <MenuItem key={degree} value={degree}>
                      {degree}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel htmlFor="specialization-select">Specialization</InputLabel>
                <Select
                  id="specialization-select"
                  name="specialization"
                  value={formik.values.specialization}
                  onChange={formik.handleChange}
                  onBlur={formik.handleBlur}
                >
                  {specializationOptions.map((specialization) => (
                    <MenuItem key={specialization} value={specialization}>
                      {specialization}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            {formik.values.role === 'Student' && (
              <>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel htmlFor="programming-languages-select">Programming Languages Known</InputLabel>
                    <Select
                      id="programming-languages-select"
                      multiple
                      name="programmingLanguages"
                      value={formik.values.programmingLanguages}
                      onChange={(e) => formik.setFieldValue('programmingLanguages', e.target.value)}
                      renderValue={(selected) => (selected as string[]).join(', ')}
                    >
                      {languageOptions.map((lang) => (
                        <MenuItem key={lang} value={lang}>
                          <Checkbox checked={formik.values.programmingLanguages.includes(lang)} />
                          <ListItemText primary={lang} />
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel htmlFor="internship-details-select">Internship Details</InputLabel>
                    <Select
                      id="internship-details-select"
                      name="internshipDetails"
                      value={formik.values.internshipDetails}
                      onChange={formik.handleChange}
                    >
                      {internshipOptions.map((company) => (
                        <MenuItem key={company} value={company}>
                          {company}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Courses Completed"
                    name="coursesCompleted"
                    value={formik.values.coursesCompleted}
                    onChange={formik.handleChange}
                    onBlur={formik.handleBlur}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="LinkedIn Profile Link"
                    name="linkedInProfileLink"
                    value={formik.values.linkedInProfileLink}
                    onChange={formik.handleChange}
                    onBlur={formik.handleBlur}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Certifications"
                    name="certifications"
                    value={formik.values.certifications}
                    onChange={formik.handleChange}
                    onBlur={formik.handleBlur}
                  />
                </Grid>
              </>
            )}
          </Grid>
          <Button type="submit" variant="contained" color="primary" fullWidth>
            Register
          </Button>
        </form>
        <Button
          variant="outlined"
          color="secondary"
          fullWidth
          sx={{ mt: 2 }}
          onClick={handleRegister}
        >
          Back to Login
        </Button>
      </Box>
    </div>
  );
};

export default Register;

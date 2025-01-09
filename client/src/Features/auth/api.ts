import axios from 'axios';

const API_BASE_URL = 'http://localhost:5220'; 

export const validateUser = async (username: string, password: string) => {
  const response = await axios.post(`${API_BASE_URL}/validate`, { 
  "username" :username,
  "password" :password });
  console.log(response)
  return response.data;
};
export const fetchstudents = async () => {
  const response = await axios.post(`${API_BASE_URL}/fetchstudents`);
  console.log(response)
  return response.data;
};


export const registerUser = async (userData: Record<string, string | number>) => {
  const response = await axios.post(`${API_BASE_URL}/register`, userData);
  return response.data;
};

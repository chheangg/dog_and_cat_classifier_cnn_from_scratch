import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:3000/api',
  timeout: 5000,
});

export async function uploadImage(file: File) : Promise<{ result: number[] }> {
  const formData = new FormData();
  formData.append('file', file);

  return (await api.post('/model/infer', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })).data
}
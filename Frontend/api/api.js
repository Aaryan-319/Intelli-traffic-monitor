export const getData = async () => {
    const response = await fetch('http://localhost:5000/api/data');
    const data = await response.json();
    return data;
  };
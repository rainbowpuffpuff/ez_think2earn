// src/theme.js
import { createTheme } from '@mui/material/styles';

// Define the custom color palette
const theme = createTheme({
  palette: {
    primary: {
      main: '#673AB7', // Deep Purple
    },
    secondary: {
      main: '#FFC107', // Amber
    },
    background: {
      default: '#F5F5F5', // Light Gray
      paper: '#FFFFFF', // White for Paper components
    },
    accent: {
      main: '#00BCD4', // Cyan
    },
  },
  typography: {
    fontFamily: 'Montserrat, sans-serif',
    h1: {
      fontWeight: 700,
      color: '#673AB7',
    },
    h2: {
      fontWeight: 500,
      color: '#673AB7',
    },
    body1: {
      fontWeight: 300,
      color: '#333333',
    },
    button: {
      textTransform: 'none', // Disable uppercase transformation
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          padding: '10px 20px',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          padding: '20px',
        },
      },
    },
  },
});

export default theme;


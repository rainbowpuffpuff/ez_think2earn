// src/App.js
import React from 'react';
import FileUpload from './FileUpload';
import { AppBar, Toolbar, Typography, Container } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme';

function App() {
  return (
    <ThemeProvider theme={theme}>
      {/* AppBar for Navigation */}
      <AppBar position="static">
        <Toolbar>
          <UploadFileIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div">
            EZKL ONNX Processor
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="md" sx={{ mt: 5, mb: 5 }}>
        <Typography variant="h4" align="center" gutterBottom>
          Run a private network on public data. Upload your model.
        </Typography>
        <FileUpload />
      </Container>
    </ThemeProvider>
  );
}

export default App;


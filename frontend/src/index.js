// src/index.js
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import CssBaseline from '@mui/material/CssBaseline';

ReactDOM.render(
  <React.StrictMode>
    {/* CssBaseline kickstarts an elegant, consistent, and simple baseline to build upon. */}
    <CssBaseline />
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);


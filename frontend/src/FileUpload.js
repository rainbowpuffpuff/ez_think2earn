// src/FileUpload.js
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

/**
 * Helper component to truncate and display JSON data.
 * Displays up to 'maxItems' per subfield and indicates truncation if necessary.
 */
function TruncatedField({ title, data, maxItems = 5 }) {
  const truncateArray = (arr) => {
    if (arr.length <= maxItems) {
      return arr;
    }
    return [...arr.slice(0, maxItems), '... truncated ...'];
  };

  const truncateObject = (obj) => {
    const keys = Object.keys(obj);
    if (keys.length <= maxItems) {
      return obj;
    }
    const truncatedKeys = [...keys.slice(0, maxItems), '... truncated ...'];
    const truncatedObj = {};
    truncatedKeys.forEach((key) => {
      truncatedObj[key] = obj[key];
    });
    return truncatedObj;
  };

  const recursiveTruncate = (data) => {
    if (Array.isArray(data)) {
      const truncated = truncateArray(data);
      return truncated.map((item) =>
        typeof item === 'object' && item !== null ? recursiveTruncate(item) : item
      );
    } else if (typeof data === 'object' && data !== null) {
      const truncated = truncateObject(data);
      const result = {};
      Object.keys(truncated).forEach((key) => {
        const value = truncated[key];
        result[key] = typeof value === 'object' && value !== null ? recursiveTruncate(value) : value;
      });
      return result;
    }
    return data; // Primitive data types
  };

  let displayData;
  if (Array.isArray(data)) {
    displayData = truncateArray(data);
    displayData = displayData.map((item) =>
      typeof item === 'object' && item !== null ? recursiveTruncate(item) : item
    );
  } else if (typeof data === 'object' && data !== null) {
    displayData = truncateObject(data);
    const result = {};
    Object.keys(displayData).forEach((key) => {
      const value = displayData[key];
      result[key] = typeof value === 'object' && value !== null ? recursiveTruncate(value) : value;
    });
    displayData = result;
  } else {
    displayData = data; // Primitive data types
  }

  return (
    <Box sx={{ mt: 2, textAlign: 'left' }}>
      <Typography variant="subtitle1" gutterBottom color="#673AB7">
        {title}
      </Typography>
      <Paper sx={{ padding: 2, backgroundColor: '#EDE7F6' }}>
        <pre style={{ overflowX: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {JSON.stringify(displayData, null, 2)}
        </pre>
      </Paper>
    </Box>
  );
}

function FileUpload() {
  const [outputs, setOutputs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [inputData, setInputData] = useState(null);
  const [witnessData, setWitnessData] = useState(null);

  /**
   * Callback function when a file is dropped or selected.
   * Handles the file upload and response processing.
   */
  const onDrop = useCallback(
    (acceptedFiles) => {
      // Validate that a file is selected
      if (acceptedFiles.length === 0) {
        setError('Please upload a valid .onnx file.');
        return;
      }

      setLoading(true);
      setOutputs([]);
      setError('');
      setInputData(null);
      setWitnessData(null);

      const file = acceptedFiles[0];
      const formData = new FormData();
      formData.append('file', file);

      axios
        .post('http://localhost:5000/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })
        .then((res) => {
          // Store the returned data in states
          setOutputs(res.data.outputs);
          setInputData(res.data.input_data);
          setWitnessData(res.data.witness_data);
          setLoading(false);
        })
        .catch((err) => {
          console.error(err);
          if (err.response && err.response.data && err.response.data.error) {
            setError(`Error: ${err.response.data.error}`);
          } else {
            setError('An unexpected error occurred.');
          }
          setLoading(false);
        });
    },
    [] // No dependencies as we removed task, subjectId, runId
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: '.onnx',
    multiple: false,
  });

  return (
    <Box sx={{ textAlign: 'center', maxWidth: 800, margin: '0 auto', padding: 2 }}>
      {/* Title */}
      <Typography variant="h4" gutterBottom color="#673AB7">
        Upload Your ONNX Model
      </Typography>

      {/* Drag-and-Drop Upload Area */}
      <Paper
        {...getRootProps()}
        elevation={isDragActive ? 6 : 3}
        sx={{
          padding: 4,
          border: '2px dashed #673AB7',
          backgroundColor: isDragActive ? '#EDE7F6' : '#FFFFFF',
          cursor: 'pointer',
          transition: 'background-color 0.3s, box-shadow 0.3s',
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 60, color: '#673AB7' }} />
        <Typography variant="h6" sx={{ mt: 2, color: '#333333' }}>
          {isDragActive
            ? 'Drop the .onnx file here...'
            : 'Drag & drop a .onnx file here, or click to select file'}
        </Typography>
        <Button variant="contained" color="primary" sx={{ mt: 2 }}>
          Choose File
        </Button>
      </Paper>

      {/* Loading Indicator */}
      {loading && (
        <Box sx={{ mt: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <CircularProgress color="primary" />
          <Typography variant="body1" sx={{ mt: 2, color: '#673AB7' }}>
            Processing your model...
          </Typography>
        </Box>
      )}

      {/* Error Message */}
      {error && (
        <Alert severity="error" sx={{ mt: 4 }}>
          {error}
        </Alert>
      )}

      {/* Processing Steps Output */}
      {outputs.length > 0 && (
        <Box sx={{ mt: 4, textAlign: 'left' }}>
          <Typography variant="h5" gutterBottom color="#673AB7">
            Processing Steps:
          </Typography>
          <Paper sx={{ padding: 2, backgroundColor: '#F3E5F5' }}>
            <List>
              {outputs.map((output, index) => (
                <ListItem key={index}>
                  <ListItemText primary={output} />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Box>
      )}

      {/* Display Input JSON Data with Truncation */}
      {inputData && (
        <Box sx={{ mt: 4, textAlign: 'left' }}>
          <Typography variant="h5" gutterBottom color="#673AB7">
            Input JSON:
          </Typography>
          <Paper sx={{ padding: 2, backgroundColor: '#EDE7F6' }}>
            {Object.keys(inputData).map((key, index) => (
              <TruncatedField key={index} title={key} data={inputData[key]} maxItems={5} />
            ))}
          </Paper>
        </Box>
      )}

      {/* Display Witness JSON Data with Truncation */}
      {witnessData && (
        <Box sx={{ mt: 4, textAlign: 'left' }}>
          <Typography variant="h5" gutterBottom color="#673AB7">
            Witness JSON:
          </Typography>
          <Paper sx={{ padding: 2, backgroundColor: '#EDE7F6' }}>
            {Object.keys(witnessData).map((key, index) => (
              <TruncatedField key={index} title={key} data={witnessData[key]} maxItems={5} />
            ))}
          </Paper>
        </Box>
      )}
    </Box>
  );
}

export default FileUpload;


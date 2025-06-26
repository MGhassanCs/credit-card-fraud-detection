import React, { useState } from "react";
import {
  Container, Typography, TextField, Button, Grid, Tooltip, Box, Alert
} from "@mui/material";
import axios from "axios";

const featureInfo = "Anonymized feature from PCA. Original meaning is not public.";

const initialState = {};
for (let i = 1; i <= 28; i++) initialState[`V${i}`] = 0.0;
initialState["Amount"] = 0.0;

function App() {
  const [features, setFeatures] = useState(initialState);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFeatures({ ...features, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError(null);
    try {
      const response = await axios.post("http://localhost:8000/predict", features);
      setResult(response.data.fraud_probability);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Could not connect to API. Is the FastAPI server running?"
      );
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        💳 Credit Card Fraud Detection
      </Typography>
      <Typography variant="body1" color="textSecondary" sx={{ mb: 2 }}>
        Features V1–V28 are anonymized principal components from the original dataset.
        Their exact meanings are not public, but they are derived from transaction details.
      </Typography>
      <Box component="form" onSubmit={handleSubmit} sx={{ mb: 3 }}>
        <Grid container spacing={2}>
          {[...Array(28)].map((_, i) => (
            <Grid item xs={12} sm={6} md={4} key={`V${i + 1}`}>
              <Tooltip title={featureInfo} arrow>
                <TextField
                  label={`V${i + 1}`}
                  name={`V${i + 1}`}
                  type="number"
                  value={features[`V${i + 1}`]}
                  onChange={handleChange}
                  fullWidth
                  size="small"
                  inputProps={{ step: "any" }}
                />
              </Tooltip>
            </Grid>
          ))}
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              label="Transaction Amount (USD)"
              name="Amount"
              type="number"
              value={features.Amount}
              onChange={handleChange}
              fullWidth
              size="small"
              inputProps={{ step: "any" }}
            />
          </Grid>
        </Grid>
        <Button
          type="submit"
          variant="contained"
          color="primary"
          sx={{ mt: 3 }}
          fullWidth
        >
          Predict Fraud Probability
        </Button>
      </Box>
      {result !== null && (
        <Alert severity={result > 0.5 ? "error" : "success"}>
          Fraud Probability: <b>{(result * 100).toFixed(2)}%</b>
          {result > 0.5 ? " ⚠️ High risk of fraud!" : " ✅ Low risk of fraud."}
        </Alert>
      )}
      {error && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Container>
  );
}

export default App;
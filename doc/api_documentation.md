markdown
# API Documentation

## Endpoints

### POST /predict
Makes a sales prediction based on input data.

#### Request Body
```json
{
    "date": "2024-01-01",
    "customer_id": 1001,
    "product_id": 5001,
    ...
}
```

#### Response
```json
{
    "predicted_sales": 99.98,
    "confidence": 0.95
}
```

### GET /health
Health check endpoint.

# API Reference

## Base URL


---

## **Authentication**

Currently, the API does not require authentication for simplicity. However, in a production environment, it's recommended to implement authentication mechanisms like API keys or OAuth tokens.

---

## **Endpoints**

### **1. POST `/api/handle_query`**

Handles incoming customer queries and routes them to the appropriate agent.

#### **Description**

Processes the user's message and returns a response generated by one of the specialized agents:

- `QueryAgent` for general queries.
- `OrderAgent` for order-related inquiries.
- `RecommendationAgent` for product recommendations.

#### **Request Headers**

- `Content-Type: application/json`

#### **Request Body**

```json
{
  "message": "Your question here",
  "context": {
    "user_id": "optional_user_id",
    "session_id": "optional_session_id"
  }
}

--------------------------------------

Response
Success (200 OK)

json
Copy code
{
  "response": "Agent's reply",
  "agent": "AgentName",
  "timestamp": "2023-10-15T14:28:00Z"
}
response (string): The reply from the agent.
agent (string): The name of the agent that handled the query.
timestamp (string): The time the response was generated.
Error (4xx or 5xx)

json
Copy code
{
  "error": "Error Message",
  "details": "Additional details if any"
}
Example Request
bash
Copy code
curl -X POST http://localhost:8000/api/handle_query \
     -H "Content-Type: application/json" \
     -d '{
           "message": "I want to track my order",
           "context": {"user_id": "12345"}
         }'
Example Response
json
Copy code
{
  "response": "Your order #101 is currently in transit and will arrive on 2023-10-18.",
  "agent": "OrderAgent",
  "timestamp": "2023-10-15T14:28:00Z"
}
2. GET /api/orders/{order_id}
Retrieves details about a specific order.

Description
Provides order status and other relevant information based on the order_id.

Path Parameters
order_id (integer, required): The unique identifier of the order.
Response
Success (200 OK)

json
Copy code
{
  "order_id": 101,
  "status": "Shipped",
  "estimated_delivery": "2023-10-18",
  "items": [
    {"product_id": 1, "name": "Product A", "quantity": 2}
  ]
}
Error (404 Not Found)

json
Copy code
{
  "error": "Order Not Found",
  "details": "No order found with ID 999"
}
Example Request
bash
Copy code
curl -X GET http://localhost:8000/api/orders/101
3. GET /api/products
Retrieves a list of products, optionally filtered by query parameters.

Query Parameters
category (string, optional): Filter products by category.
search (string, optional): Search products by name or description.
Response
Success (200 OK)

json
Copy code
{
  "products": [
    {"id": 1, "name": "Product A", "category": "Electronics"},
    {"id": 3, "name": "Product C", "category": "Electronics"}
  ]
}
Example Request
bash
Copy code
curl -X GET "http://localhost:8000/api/products?category=Electronics"
4. POST /api/recommendations
Provides personalized product recommendations based on user input.

Request Headers
Content-Type: application/json
Request Body
json
Copy code
{
  "user_input": "I'm looking for a new laptop",
  "user_id": "12345"
}
user_input (string, required): The user's interest or query.
user_id (string, optional): The unique identifier of the user for personalization.
Response
Success (200 OK)

json
Copy code
{
  "recommendations": [
    {"id": 5, "name": "Laptop Model X", "category": "Electronics"},
    {"id": 6, "name": "Laptop Model Y", "category": "Electronics"}
  ]
}
Example Request
bash
Copy code
curl -X POST http://localhost:8000/api/recommendations \
     -H "Content-Type: application/json" \
     -d '{
           "user_input": "I'm looking for a new laptop",
           "user_id": "12345"
         }'
Error Handling
The API uses standard HTTP status codes to indicate the success or failure of an API request.

200 OK: The request was successful.
400 Bad Request: The request was invalid or cannot be otherwise served.
404 Not Found: The requested resource could not be found.
500 Internal Server Error: An error occurred on the server.
Error Response Format
json
Copy code
{
  "error": "Error Type",
  "details": "Detailed error message"
}
Headers
Content-Type: Indicates the media type of the resource. Should be application/json.
Date: The date and time when the response was generated.
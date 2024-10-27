# Agent Architecture

This system utilizes three specialized agents to handle different types of customer interactions efficiently.

## QueryAgent

- **Purpose**: Handles general customer queries that do not fall into specific categories like orders or recommendations.
- **Functionality**:
  - Uses natural language processing to interpret queries.
  - Provides informative responses or directs users to relevant resources.

## OrderAgent

- **Purpose**: Manages all order-related inquiries.
- **Functionality**:
  - Checks order status.
  - Processes cancellations or modifications.
  - Interfaces with the order management database.

## RecommendationAgent

- **Purpose**: Offers personalized product recommendations.
- **Functionality**:
  - Analyzes user input and context.
  - Utilizes product data to suggest relevant items.
  - Enhances user engagement and upselling opportunities.

## Workflow

1. **User Input**: The user submits a query through the chat interface.
2. **Input Logging**: The system logs the input for analytics.
3. **Agent Selection**: The main application routes the query to the appropriate agent based on keywords.
4. **Response Generation**: The agent processes the query and generates a response.
5. **Response Delivery**: The formatted response is sent back to the user's chat interface.
6. **Escalation**: If an agent cannot handle the query, it flags it for human intervention.

---


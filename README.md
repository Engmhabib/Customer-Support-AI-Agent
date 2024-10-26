# Customer-Support-AI-Agent
An AI-powered customer support automation system using Crew AI and LangChain for natural language processing. This system provides automated responses to customer queries, personalized product recommendations, and efficient escalation management.

## Features

- Automated customer support response
- Multi-agent collaboration for complex queries
- Product recommendation engine
- Analytics dashboard
- Escalation to human agents for complex cases

## Getting Started

Follow the instructions in the [Setup Guide](docs/SetupGuide.md) to set up and run the project locally.

## Documentation

- [Setup Guide](docs/SetupGuide.md)
- [Agent Architecture](docs/AgentArchitecture.md)
- [API Reference](docs/APIReference.md)

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

![image](https://github.com/user-attachments/assets/4ccd3b5b-6f81-49bf-8a4e-6d57c097d9ac)


+---------------------+        +---------------------+        +---------------------+
|                     |        |                     |        |                     |
|     Frontend        |  <---> |      Backend        |  <---> |      Agents         |
| (HTML, CSS, JS)     |        |    (FastAPI)        |        |  (Crew AI, NLP)     |
|                     |        |                     |        |                     |
+---------------------+        +---------------------+        +---------------------+
         ^                              |                              |
         |                              |                              |
         |                              v                              v
         |                     +---------------------+        +---------------------+
         |                     |                     |        |                     |
         |                     |    Data Layer       |        |  External Services  |
         |                     |  (JSON, Database)   |        |  (Crew AI, LangChain)|
         |                     |                     |        |                     |
         |                     +---------------------+        +---------------------+
         |                              ^
         |                              |
         +------------------------------+

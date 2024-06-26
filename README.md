# AI Doc Companion

This sample project demonstrates the implementation of the Retrieval Augmented Generation (RAG) method, leveraging `Amazon Bedrock` for Large Language Model (LLM) capabilities. The text embeddings generated by the titan text embeeding model, are stored in `Amazon Opensearch Service` with support for a vector engine.

## Features
- Utilizes RAG method for efficient text retrieval and generation.
- Empowers users to upload extensive PDF documents.
- Enables natural language search for various information within the documents.

## Technologies Used
- **Amazon Bedrock:** Employs LLM and text embedding model for language generation and text embedding.
- **Amazon Opensearch Service:** Stores text embeddings with vector engine support.
- **Streamlit:** Frontend development framework.

## How it Works
1. Users upload PDF documents to the system.
2. Amazon Bedrock processes the documents, generating text embeddings.
3. Text embeddings are stored in Amazon Opensearch Service.
4. Frontend, built using Streamlit, allows users to perform natural language searches.
5. Opensearch retrieves relevant documents based on the user's query.

## Potential Applications
- Academic research.
- Legal document analysis.
- Knowledge management systems.
- Information retrieval in various domains.

## Prerequisites

Before using this project, ensure you meet the following prerequisites:

1. **Amazon Opensearch Domain Setup:**
   - Create an Amazon Opensearch domain with master user enabled.
   - Obtain the necessary information such as the domain endpoint, master username, and password.

2. **Access to Amazon Bedrock Service:**
   - Ensure you have access to the Amazon Bedrock service to leverage its Large Language Model capabilities.

3. **DynamoDB Table Setup:**
   - Configure a DynamoDB table with a single attribute `SessionId`, using KeyType `HASH`, for storing chat history.

4. **Environment Setup:**
   - Create an environment file named `.env` with the following contents:

     ```plaintext
        OPENSEARCH_DOMAIN_ENDPOINT= [Domain Endpoint of your created OpenSearch domain]
        OPENSEARCH_USERNAME= [Master username of your created OpenSearch domain]
        OPENSEARCH_PASSWORD= [Master password of your created OpenSearch domain]
        INDEX_NAME= [Name for the index to be created in Opensearch]
        EMBEDDING_MODEL_ID= [Name of the embedding model to be used, e.g., amazon.titan-embed-g1-text-02]
        CLAUDE_SONNET_MODEL= [Name of the large language model to be used, e.g., anthropic.claude-3-sonnet-20240229-v1:0]
        ACCEPT=application/json
        CONTENT_TYPE=application/json
        DYNAMODB_TABLE=DynamSessionTable
     ```

## Usage
1. Install dependencies using `pip install -r requirements.txt`.
2. Run the Streamlit application using `streamlit run app.py`.
3. Upload PDF documents.
4. Input natural language queries.
5. Retrieve relevant information from uploaded documents.

## Demo
https://github.com/AnishMachamasi/AI-Doc-Companion/assets/98002255/aff9140c-bcf3-438b-a9d1-745ed3daab27

## Contributions
Contributions are welcome! Feel free to submit pull requests or raise issues for any improvements or features.

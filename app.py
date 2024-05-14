import streamlit as st
import json
from dotenv import load_dotenv
import os
import pandas as pd
from opensearchpy.helpers import bulk
from preprocessing_and_embedding.preprocess import (
    get_data,
    chunk_data,
    restructure_chunks,
)
from preprocessing_and_embedding.opensearch import (
    getOpenSearchClient,
    CreateIndex,
    checkIfIndexIsCreated,
    createIndexMapping,
    get_uploaded_document_list,
)

from preprocessing_and_embedding.bedrock import (
    get_bedrock_client,
    invoke_bedrock_embedding_model,
)

from chat_pipeline.chat_pipeline import ChatPipeline

# Load environment variables from .env file
load_dotenv()

OPENSEARCH_DOMAIN_ENDPOINT = os.getenv("OPENSEARCH_DOMAIN_ENDPOINT")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID")
ACCEPT = os.getenv("ACCEPT")
CONTENT_TYPE = os.getenv("CONTENT_TYPE")

auth = (OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD)

opensearchClient = getOpenSearchClient(OPENSEARCH_DOMAIN_ENDPOINT, auth)
bedrockClient = get_bedrock_client()


def main():
    st.title("AI Doc Companion")

    # Check if the button has been clicked
    if "button_clicked" not in st.session_state:
        st.session_state["button_clicked"] = False

    # Button to trigger the function
    if st.button("Click to show or hide uploaded documents!!"):
        # Toggle button click state
        st.session_state["button_clicked"] = not st.session_state["button_clicked"]

    # If button has been clicked, show the list
    if st.session_state["button_clicked"]:
        # Call the function to get the list
        item_list = get_uploaded_document_list(opensearchClient)

        # Display each item
        for item in item_list:
            st.write(item)

    # Left plane for Upload Document button
    st.sidebar.title("Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Choose a PDF file", accept_multiple_files=True, type="pdf"
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            data = get_data(uploaded_file)

            # chunking text using langchain's RecursiveCharacterTextSplitter
            chunks = chunk_data(data)

            # Restructuring chunks
            docs = restructure_chunks(chunks, uploaded_file.name)

            # Checking if the index with same name have already been created
            exists = checkIfIndexIsCreated(opensearchClient, INDEX_NAME)

            if not exists:
                # Creating Index in OpenSearch
                response = CreateIndex(opensearchClient, INDEX_NAME)

                # Mapping Different field in Created Index
                response = createIndexMapping(opensearchClient, INDEX_NAME)

            else:
                print("index exists.")

            # embedding creation
            for doc in docs:
                payload = {"inputText": doc["text"]}
                body = json.dumps(payload)

                embedding = invoke_bedrock_embedding_model(
                    bedrockClient, body, EMBEDDING_MODEL_ID, ACCEPT, CONTENT_TYPE
                )
                doc["vector_field"] = embedding
                doc["_index"] = INDEX_NAME

            success, failed = bulk(opensearchClient, docs)
            print("Number of Successfully indexed document:", success)
            print("Number of failed document to be indexed:", failed)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask question to your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        response = ChatPipeline(user_question=prompt).process_query(
            user_question=prompt
        )
        document_source = []
        for d in response["source_documents"]:
            document_name = d.metadata["document_name"].replace("/tmp/", "")
            page_number = d.metadata["page_number"]
            page_number = int(page_number) + 1
            documentSource = f"{document_name}, PageNo. {page_number}"
            document_source.append(documentSource)

        response = {
            "query": response["question"],
            "answer": response["answer"],
            "document Source": document_source,
        }

        # Extract document sources from the response
        document_sources = []
        page_numbers = []
        for item in response["document Source"]:
            source, page = item.split(", PageNo. ")
            document_sources.append(source.strip())
            page_numbers.append(page.strip())

        # Create a DataFrame from the document sources and page numbers
        df = pd.DataFrame(
            {"Document Source": document_sources, "Page Number": page_numbers}
        )

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(f"**Answer:** {response['answer']}")
            st.markdown("**Document Sources:**")
            st.table(df)

        response_message = f"""{response["answer"]}\n\nDocument Sources:\n\n"""
        for source in response["document Source"]:
            response_message += source + "\n\n"
        st.session_state.messages.append(
            {"role": "assistant", "content": response_message}
        )


if __name__ == "__main__":
    main()

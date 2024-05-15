from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
import boto3
import os
from dotenv import load_dotenv
from preprocessing_and_embedding.bedrock import get_bedrock_client

# Load environment variables from .env file
load_dotenv()

# We are also providing a different chat history retriever which outputs the history as a Claude chat (ie including the \n\n)
_ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}

OPENSEARCH_DOMAIN_ENDPOINT = os.getenv("OPENSEARCH_DOMAIN_ENDPOINT")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID")
CLAUDE_SONNET_MODEL = os.getenv("CLAUDE_SONNET_MODEL")
ACCEPT = os.getenv("ACCEPT")
CONTENT_TYPE = os.getenv("CONTENT_TYPE")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE")

auth = (OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD)
bedrockClient = get_bedrock_client()


class ChatPipeline:
    def __init__(self, user_question):
        self.question = user_question

    @staticmethod
    def create_bedrock_embedding(bedrockClient: boto3.client) -> BedrockEmbeddings:
        """
        Creates bedrock embedding using titan
        """
        bedrock_embeddings_client = BedrockEmbeddings(
            client=bedrockClient,
            model_id=EMBEDDING_MODEL_ID,
        )
        return bedrock_embeddings_client

    @classmethod
    def get_opensearch_vector_Search_client(
        cls, index_name: str
    ) -> OpenSearchVectorSearch:
        """
        Returns LLM opensearch client
        """
        docsearch = OpenSearchVectorSearch(
            index_name=index_name,
            embedding_function=cls.create_bedrock_embedding(bedrockClient),
            opensearch_url=f"https://{OPENSEARCH_DOMAIN_ENDPOINT}",
            http_auth=auth,
            is_aoss=False,
        )
        return docsearch

    @staticmethod
    def create_bedrock_llm(bedrock_client: boto3.client):
        """
        Returns langchain bedrock model
        """
        bedrock_llm = BedrockChat(
            model_id=CLAUDE_SONNET_MODEL,
            client=bedrock_client,
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0,
                "top_k": 250,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman:"],
            },
        )
        return bedrock_llm

    @staticmethod
    def _get_chat_history(chat_history):
        buffer = ""
        for dialogue_turn in chat_history:
            if isinstance(dialogue_turn, BaseMessage):
                role_prefix = _ROLE_MAP.get(
                    dialogue_turn.type, f"{dialogue_turn.type}: "
                )
                buffer += f"\n{role_prefix}{dialogue_turn.content}"
            elif isinstance(dialogue_turn, tuple):
                human = "\n\nHuman: " + dialogue_turn[0]
                ai = "\n\nAssistant: " + dialogue_turn[1]
                buffer += "\n" + "\n".join([human, ai])
            else:
                raise ValueError(
                    f"Unsupported chat history format: {type(dialogue_turn)}."
                    f" Full chat history: {chat_history} "
                )
        return buffer

    @classmethod
    def process_query(cls, user_question):
        question = user_question
        # the condense prompt for Claude
        condense_prompt_claude = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
                Chat History:
                {chat_history}
                Follow Up Input: {question}
                Standalone question:"""
        )

        message_history = DynamoDBChatMessageHistory(
            table_name=DYNAMODB_TABLE,
            session_id="0",
        )

        memory_chain = ConversationBufferWindowMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True,
            k=1,
            output_key="answer",
        )

        opensearch_vector_search_client = cls.get_opensearch_vector_Search_client(
            INDEX_NAME
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm=cls.create_bedrock_llm(bedrockClient),
            retriever=opensearch_vector_search_client.as_retriever(
                search_kwargs={"k": 5}
            ),  # search_type = "similarity", search_kwargs = { "k": 23 }
            memory=memory_chain,
            get_chat_history=cls._get_chat_history,
            return_source_documents=True,
            # verbose=True,
            condense_question_prompt=condense_prompt_claude,
            chain_type="stuff",  # 'refine',
            # max_tokens_limit=300
        )
        # the LLMChain prompt to get the answer. the ConversationalRetrievalChange does not expose this parameter in the constructor
        qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(
            """
            **Task**
            <task>
            Your task is to answer the user question properly. If user ask you about documents, answer regarding them. If user greets you or ask you different question, answer them accordingly like returning greeting back.
            </task>
            
            **Profile**
            <profile>
            1. Language: You will communicate in English only.
            2. You are a question answer expert.
            </profile>
            
            **Rules**
            <rules>
            1. Maintain a consistent and professional persona throughout the interaction.
            2. Always analyse the question first and according to question perform different task.
            3. If user question's answer lies in context, use context to answer them. Remember to analyse the provided context properly before replying anything.
            4. Do not to give fake answer.
            5. In your answer, never say that you are answering from the context. Be as human as possible like how a human would answer. For example, never answer by saying "Based on the provided context,".
            </rules>
            
            <question>
            {question}
            <question>

            <context>
            {context}
            </context>
            """
        )
        result = qa.invoke({"question": question})
        return result

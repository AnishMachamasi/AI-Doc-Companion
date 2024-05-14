from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# get pdf data
def get_data(pdf_file):
    temp_file_path = str(pdf_file.name)
    with open(temp_file_path, "wb") as file:
        file.write(pdf_file.getvalue())

    loader = PyPDFLoader(temp_file_path)
    data = loader.load()
    return data


# chunking the document
def chunk_data(data):
    textSplitternltk = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=300, length_function=len
    )
    chunks = textSplitternltk.split_documents(data)
    return chunks


def restructure_chunks(chunks, fileName):
    reference_name = fileName.split(".")[0]

    # convert chunks into the form of dictionary
    docs = []
    for idx, doc in enumerate(chunks):
        document = {
            "text": doc.page_content + f" {reference_name}",
            "document_name": doc.metadata["source"],
            "page_number": doc.metadata["page"] + 1,
        }
        docs.append(document)

    print(docs)

    return docs

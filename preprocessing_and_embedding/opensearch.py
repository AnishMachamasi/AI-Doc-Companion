from opensearchpy import OpenSearch, RequestsHttpConnection


def getOpenSearchClient(domainEndpoint, auth):
    client = OpenSearch(
        hosts=[{"host": domainEndpoint, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300,
    )

    return client


def CreateIndex(client, indexName):
    settings = {
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil",
                "knn.algo_param.ef_search": 100,
            }
        }
    }
    res = client.indices.create(index=indexName, body=settings, ignore=[400])
    return bool(res["acknowledged"])


def createIndexMapping(client, indexName):
    response = client.indices.put_mapping(
        index=indexName,
        body={
            "properties": {
                "text": {"type": "text"},
                "document_name": {"type": "text"},
                "page_number": {"type": "text"},
                "vector_field": {"type": "knn_vector", "dimension": 1536},
            }
        },
    )

    return bool(response["acknowledged"])


# check if the index is created
def checkIfIndexIsCreated(client, indexName):
    exists = client.indices.exists(indexName)
    return exists


def get_uploaded_document_list(openserchClient):
    documents = []
    queryCount = {"size": 1000, "query": {"match_all": {}}}
    # get response of the query from the opensearch
    response = openserchClient.search(body=queryCount, index="chatdocument")
    hits = response["hits"]["hits"]

    for hit in hits:
        documents.append(hit.get("_source").get("document_name"))

    documents = list(set(documents))
    result_list = [f"{index}. {item}" for index, item in enumerate(documents, start=1)]
    return result_list
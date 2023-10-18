# Deploying the AIGC Development Environment 

## Prepare 
Before you begin, please ensure that your environment meets the requirements for running KubeBlocks. KubeBlocks is a cloud-native data infrastructure management platform based on Kubernetes. So you will need a Kubernetes service provided by a cloud provider such as EKS (AWS), GKE (Google), ACK (Alibaba), and so on. You can check a list of all supported cloud providers from [here](https://kubeblocks.io/docs/preview/user_docs/try-out-on-playground/try-kubeblocks-on-cloud#preparation).

Another more convenient way is to deploy a mini Kubernetes environment on your laptop. You can find the specific configuration requirements [here](https://kubeblocks.io/docs/preview/user_docs/try-out-on-playground/try-kubeblocks-on-your-laptop#before-you-start).

1. install **kbcli**(the cli tools assisting you to interact with KubeBlocks)

    For specific installation instructions, please refer to the [official website](https://kubeblocks.io/docs/preview/user_docs/installation/install-with-kbcli/install-kbcli#install-kbcli-1).
2. install **KubeBlocks**

`kbcli kubeblocks install --version 0.7.0-beta.6`

## Enable Kubeblocks Vector database
KubeBlocks supports the management of vector databases, such as Qdrant, Milvus, and Weaviate. In this chapter, we take Qdrant as an example to show how to build AIGC demo with Kubeblocks.

1. enable `qdrant` addon
    
   using `kbcli addon enable qdrant` enable qdrant addon. 
   
   You can use `kbcli clusterdefinition list` to check if the addon is ready.
   ![cd-list](/aigc/img/cd-list.png)
2. create qdrant cluster

    using `kbcli cluster create my-qdrant --cluster-defnition qdrant` and `kbcli cluster list` to check if the cluster is Running. This process typically takes 5-8 minutes, depending on your network conditions.
    ![cluster-create](/aigc/img/cluster-create.png)

## Deploy private LLM in KubeBlocks

using `kbcli addon enable vllm` and `kbcli cluster create --cluster-definition vllm` will create a priavate LLM in KubeBlocks.

notes: vLLM is a quantized LLM model, and even though it has been quantized, it still puts some strain on running smoothly in a regular laptop's Docker environment. We recommend choosing an appropriate cloud environment for the large model

## Demo in Jupyter Application
In addition, KubeBlocks also provides Jupyter application that support most AIGC libraries, including llama-index,lang-chain,transform, and more. You can start developing and debugging your AIGC demo in your Kubernetes environment with simple commands.

1.  using `kbcli addon enable jupyter-notebook` enable jupyter-notebook application
2. using `kbcli dashboard open jupyter-notebook` open the Jupyter dashboard.
![open-jupyter](/aigc/img/open-jupyter.png)

### An AI question-and-answer assistant demo in KubeBlocks AIGC infrastructure
Next,I will demonstrate a demo of building an intelligent AI question-and-answer assistant for the KubeBlocks user-docs on the AIGC infrastructure built on KubeBlocks.

**All the following code will be executed in Jupyter**

#### Prepare for LLM application
In a LLM Aplication, We need to convert the text input from the user into vectors and store them in a vector database. So, we need to implement the following steps in the code:
1. load text to vector embedding model.
```python
from typing import Any, Dict, List
from text2vec import SentenceModel
from llama_index import LangchainEmbedding
from llama_index.readers.file.markdown_reader import MarkdownReader
from langchain.embeddings.base import Embeddings


class Text2VecEmbedding(Embeddings):
    def __init__(self):
        self.model = SentenceModel('shibing624/text2vec-base-chinese')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.model.encode(text)
        
embedding_model = LangchainEmbedding(Text2VecEmbedding())
vector_size = 768
reader = MarkdownReader()
```
![prepare](/aigc/img/prepare.png)
2. create vector database client.   
```python
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams,Distance
```
When creating a client to connect to the vector database, we need to know the backend address of that database. You can use the `kbcli cluster describe` command to view the vector database information in KubeBlocks.
![cluster-describe](/aigc/img/cluster-describe.png)
```python
"""
`kbcli cluster describe <qdrant_cluster_name>` get the qdrant server's information 
"""
url = "my-qdrant-qdrant.default.svc.cluster.local"
port = 6333
grpc_port = 6334
distance = "Cosine"

client = qdrant_client.QdrantClient(
                url=url,
                port=port,
                prefer_grpc=False,
                https=False,
                timeout=1000, 
            )
client.recreate_collection(collection_name="demo",vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ))

connector = QdrantVectorStore(
            client=client,
            collection_name="demo",
            vectors_config=VectorParams(size=vector_size, distance=distance),
)
```

#### Upload our user-docs
Now, we will upload some KubeBlocks user docs and load them into our vector database using the previously prepared embedding model.
![upload-files](/aigc/img/upload-files.png)
```python
from llama_index.data_structs.data_structs import Node
from llama_index.vector_stores.types import NodeWithEmbedding
from llama_index.schema import NodeRelationship, RelatedNodeInfo
import os
from typing import List

def process_file(file):
    docs = reader.load_data(file)

    nodes: List[NodeWithEmbedding] = []
    for doc in docs:
        vector = embedding_model.get_text_embedding(doc.text)
        doc.embedding = vector
        node = Node(
            text=doc.text,
            doc_id=doc.doc_id,
        )
        node.relationships = {
            NodeRelationship.SOURCE: RelatedNodeInfo(
                node_id=node.node_id, metadata={"source": file}
            )
        }
        nodes.append(NodeWithEmbedding(node=node, embedding=vector))

    addPoints = connector.add(nodes)

directory = "./files"
for root, dirs, files in os.walk(directory):
    for file in files:
        file_path = os.path.join(root, file)
        process_file(file_path)
```
And We have currently added visualization support for the Qdrant. You can use 'kubectl' to port-forward the Qdrant service to your local machine and view the vector database's status in a web page to confirm that the documents have been loaded into the vector database.

- `kubectl port-forward services/my-qdrant-qdrant 10000:3000 6333:6333`("3000" is the web-ui service port and "6333" is the database service port.)
- open `127.0.0.1:10000` in your browser.
![qdrant-web-ui](/aigc/img/qdrant-web-ui.png)

#### Query with LLM
Finally, we will implement the functionality to query with private LLM. In this step, we will embed the query question, perform a 'similarity match' query in the vector database, and then use a custom prompt to call our deployed private LLM API to obtain the query result.


```python
from langchain import PromptTemplate
from pydantic import BaseModel
import json
import requests
import openai
import os

query_str = "如何在windows中安装kbcli?"
query_contents = client.search(collection_name="demo",
              query_vector=embedding_model.get_text_embedding(query_str),
              with_vectors=True,
              limit=3,
              score_threshold=0.5,
              search_params={'exact': False, 'hnsw_ef': 128},
              consistency="majority"
             )
pack_context = ""
for query in query_contents:
            payload = query.payload or {}
            text = query.payload.get("text") or json.loads(
                payload["_node_content"]
            ).get("text")
            pack_context += text

'''
you can custom your own prompt
'''
prompt_template = """
上下文信息如下:
----------------\n
{context}
\n--------------------\n

根据提供的上下文信息,然后回答问题：{query}。

请确保回答准确和详细。
"""
prompt = PromptTemplate.from_template(prompt_template)
prompt_str = prompt.format(query=query_str, context=pack_context)
# check our prompt_str
print(prompt_str)
```
![check-prompt](/aigc/img/check-prompt.png)

```python
# use our private LLM API 
LLM_API = "http://a780a53170a7140f58efd575b03d60b5-667ac4219aac185d.elb.ap-northeast-1.amazonaws.com:8000/generate_stream"
data = {
    "prompt": prompt_str,
}
response = requests.post(LLM_API, json=data,stream=True)

for chunk in response.iter_lines(chunk_size=2048, decode_unicode=False, delimiter=b"\0"):
    if chunk:
        data = json.loads(chunk.decode("utf-8"))
        print(data.text)
```


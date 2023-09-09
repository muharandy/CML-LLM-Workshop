# LLM Workshop

In this workshop we will be using a privately hosted LLM model exposed as REST API. The REST API mimics OpenAI REST API specification so we will use several examples throughout the workshop that can be used as a drop in replacement of OpenAI. We will be going through several examples:
- Deploying chatGPT like application using streamlit
- Using completion API on a notebook
- Using completion API with langchain on a notebook
- Using langchain to develop Retrieval Augmented Generation on a notebook
- Building chatPDF from scratch using streamlit

## Deploying chatGPT like application using streamlit

We are not going to develop this from scratch. We will simply deploy the example code as an application in CML. This is to show what to expect and what can be done after we finish the workshop.

* Go to examples folder and open `start_chat.py`
    * This is the entry point script that we will be calling when the application starts
    * You are free to inspect and go through the code but we will not explain it in detail for now.
* Open "Applications" page from the CML left-hand side menu
* Click "New Application"
* Give it a name as "Chat[XX]" with XX is your workshop username (for example ChatUser01)
* Create subdomain as "chat[xx]-app" (for example chatuser01-app)
* Select `start_chat.py` as the script
* Select 2vCPU and 4GB memory as the resource profile 
* Hit "Create Application"

Wait for the application to be deployed. Once it is deployed and started, you can play around with it

## Using Completion API on a notebook

Now that you have seen what you can do with a privately hosted LLM model, we will start learning from the basic: accessing the LLM REST API from a notebook.

Start a notebook session and install `openai` client library. We are not using OpenAI service here, we are just using their client library to hit our privately hosted LLM server.


```python
!pip install -q openai
```

Next we will start with setting up the required environment variables to access our privately hosted LLM server.


```python
import openai
import os

# Modify OpenAI's API key, API base, model accordingly.
#os.environ['OPENAI_API_KEY']="sk-111111111111111111111111111111111111111111111111"
#os.environ['OPENAI_API_BASE']=os.environ["LLM_API_SERVER_BASE"]
openai.api_key = "sk-111111111111111111111111111111111111111111111111"
openai.api_base = os.environ["LLM_API_SERVER_BASE"]
model = "x"
```

It's all set! Now we can try using the `Completion` API. You can refer to the documentation here:
- https://platform.openai.com/docs/api-reference/completions


```python
openai.Completion.create(
            model=model,
            temperature=1.31,
            top_p=0.14,
            repetition_penalty=1.17,
            top_k=49,
            max_tokens=1000,
            prompt="Hello How are you")
```




    <OpenAIObject text_completion id=conv-1692943828948771584 at 0x7fa8a866afc0> JSON: {
      "id": "conv-1692943828948771584",
      "object": "text_completion",
      "created": 1692943828,
      "model": "Llama-2-13B-chat-GPTQ",
      "choices": [
        {
          "index": 0,
          "finish_reason": "stop",
          "text": "?\nI'm doing well, thanks for asking! It's always nice to connect with someone new. What brings you here today? Do you have any questions or topics you'd like to discuss? I'm all ears (or eyes, rather)!",
          "logprobs": null
        }
      ],
      "usage": {
        "prompt_tokens": 4,
        "completion_tokens": 54,
        "total_tokens": 58
      }
    }



Our fake OpenAI also allows us to call `ChatCompletion` API which is more suitable for conversational application. More details on the documentation here:
- https://platform.openai.com/docs/api-reference/chat


```python
openai.ChatCompletion.create(
            model=model,
            temperature=1.31,
            top_p=0.14,
            repetition_penalty=1.17,
            top_k=49,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ])
```




    <OpenAIObject chat.completions id=chatcmpl-1692763315431015424 at 0x7f207065ba10> JSON: {
      "id": "chatcmpl-1692763315431015424",
      "object": "chat.completions",
      "created": 1692763315,
      "model": "Llama-2-13B-chat-GPTQ",
      "choices": [
        {
          "index": 0,
          "finish_reason": "stop",
          "message": {
            "role": "assistant",
            "content": "Hello there! I'm here to help answer any questions you may have. What would you like to know?"
          }
        }
      ],
      "usage": {
        "prompt_tokens": 34,
        "completion_tokens": 23,
        "total_tokens": 57
      }
    }



Let's try another one. One of the hardest question humanity has ever encountered.


```python
openai.ChatCompletion.create(
            model=model,
            temperature=1.31,
            top_p=0.14,
            repetition_penalty=1.17,
            top_k=49,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! it's about to rain today. Should I bring an umbrella or a raincoat?"}
            ])
```




    <OpenAIObject chat.completions id=chatcmpl-1692943842413918976 at 0x7fa8a866a7f0> JSON: {
      "id": "chatcmpl-1692943842413918976",
      "object": "chat.completions",
      "created": 1692943842,
      "model": "Llama-2-13B-chat-GPTQ",
      "choices": [
        {
          "index": 0,
          "finish_reason": "stop",
          "message": {
            "role": "assistant",
            "content": "Based on the current weather forecast, there is a high chance of heavy rain showers throughout the day. To stay dry and comfortable, I would recommend bringing both an umbrella and a raincoat. The umbrella will provide protection from the rain while you are walking or standing, while the raincoat will keep you dry and warm during any prolonged exposure to the elements. Additionally, wearing a raincoat can also help to protect your clothes from getting wet and ruined."
          }
        }
      ],
      "usage": {
        "prompt_tokens": 55,
        "completion_tokens": 103,
        "total_tokens": 158
      }
    }



## Using completion API with langchain on a notebook

Next we are going to integrate with Langchain. It is designed to simplify the creation of applications using large language models. More information on langhchain can be found here:
- https://python.langchain.com/docs/get_started/introduction.html

Let's start with installing `langchain`


```python
!pip install -q langchain
```

Once it's installed we will start with setting up the necessary environment variables


```python
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

os.environ["OPENAI_API_KEY"] = "sk-111111111111111111111111111111111111111111111111"
os.environ["OPENAI_API_BASE"] = os.environ["LLM_API_SERVER_BASE"]
model = "x"
```

One key difference here with how we did it previously is we are now using the OpenAI wrapper from langchain. This allows us to use other functionality provided by langchain which we will try more later. For now, let's use one of langchain callbacks feature to stream the generated text even inside a notebook. Notice the `StreamingStdOutCallbackHandler()` being passed as one of the `callbacks` arguments


```python
llm = OpenAI(
            model=model,
            temperature=1.31,
            top_p=0.14,
            max_tokens=1000,
            streaming=True, 
            callbacks=[StreamingStdOutCallbackHandler()])
```


```python
resp = llm("tell me about Jakarta in one sentence")
```

    ?
    Jakarta is a bustling metropolis with a rich history, diverse culture, and modern attractions, from colonial-era landmarks to vibrant street food markets.

## Using langchain to develop Retrieval Augmented Generation on a notebook

We are now going to do something much more interesting. We will use langhchain to allow our LLM to answer questions based on the context given in a supplied document. This approach is also known as Retrieval Augmented Generation (RAG). We will be using a vector database to store the context as for fast retrieval during the Q&A chain.

Let's start with installing the required libraries


```python
!pip install -q langchain chromadb pypdf sentence_transformers pysqlite3-binary tiktoken
```

Similar with what we did previously we are going to set OpenAI endpoint to our privately hosted LLM server. For the embeddings, we will be using OpenAIEmbeddings here. However, since ours is not a real OpenAI API, it is actually using `all-mpnet-base-v2` under the hood.


```python
import os
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "sk-111111111111111111111111111111111111111111111111"
os.environ["OPENAI_API_BASE"] = os.environ["LLM_API_SERVER_BASE"]
model = "x"
```

We need to do this to workaround the issue of ChromaDB sqlite dependency. Reference:
- https://docs.trychroma.com/troubleshooting#sqlite


```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

Next, we will be using `WebBaseLoader` to load our context for Q&A. This is just an example, we can also load PDF or Text files. We just need to make sure that we use the right loaders. Once the context is loaded, we will split the text and store it in a vector store. We will be using ChromaDB for the vector store in this tutorial


```python
loader = WebBaseLoader("https://blog.cloudera.com/applying-fine-grained-security-to-apache-spark/")
data = loader.load()
```


```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)
```


```python
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
```

The content of the web page that we loaded earlier was splitted and converted into vector embeddings and loaded into ChromaDB. To simulate that the vector store works, we will show how we can retrieve information from the vector store using similarity search. When we build our QA chain later on, the chain will perform similarity search on our vector store and provide it as additional context to the LLM. So essentially what the LLM does is "answering" the question by summarizing the context given as part of the prompt.


```python
question = "What is spark secure access?"
docs = vectorstore.similarity_search(question)
print(docs)
```

    [Document(page_content='mixed results.\xa0 One approach is to use 3rd party tools (such as Privacera)\xa0 that integrate with Spark. However, it not only increases costs but requires duplication of policies and yet another external tool to manage. Other approaches also fall short by serving as partial solutions to the problem. For example, EMR plus Lake Formation makes a compromise by only providing column level security but not controlling row filtering.', metadata={'language': 'en-US', 'source': 'https://blog.cloudera.com/applying-fine-grained-security-to-apache-spark/', 'title': 'Applying Fine Grained Security to Apache Spark - Cloudera Blog'}), Document(page_content='Fine grained access control (FGAC) with Spark', metadata={'language': 'en-US', 'source': 'https://blog.cloudera.com/applying-fine-grained-security-to-apache-spark/', 'title': 'Applying Fine Grained Security to Apache Spark - Cloudera Blog'}), Document(page_content='Introducing Spark Secure Access Mode\nStarting with CDP 7.1.7 SP1 (announced earlier this year in March), we introduced a new access mode with Spark that adheres to the centralized FGAC policies defined within SDX. In the coming months we will enhance this to make it even easier with minimal to no code changes in your applications, while being performant and without limiting the Spark APIs used.', metadata={'language': 'en-US', 'source': 'https://blog.cloudera.com/applying-fine-grained-security-to-apache-spark/', 'title': 'Applying Fine Grained Security to Apache Spark - Cloudera Blog'}), Document(page_content='underlying files circumventing more fine grained access that would otherwise limit rows or columns.', metadata={'language': 'en-US', 'source': 'https://blog.cloudera.com/applying-fine-grained-security-to-apache-spark/', 'title': 'Applying Fine Grained Security to Apache Spark - Cloudera Blog'})]



```python
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
```


```python
question = "What is spark secure access?"
output = qa_chain({"query": question},return_only_outputs=True)
print(output['result'])
```

    Based on the context provided, "Spark Secure Access" refers to a new access mode introduced in CDP 7.1.7 SP1 for Apache Spark, which adheres to centralized Fine Grained Access Control (FGAC) policies defined within SDX. This access mode enables fine-grained access control over data stored in underlying files, without limiting the Spark APIs used and with minimal to no code changes required in applications.



```python

```

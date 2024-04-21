import os
import textwrap
from langchain import HuggingFaceHub
from langchain import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

QUESTION = "What is are ScoreType and EvalType in this code ?.\n\n"

INPUT_DIR = 'content/Minic'
DOC_CHUNK_SIZE = 1500
DOC_CHUNK_OVERLAP = 150
MAX_TOKEN_GENERATED = 256
CONVERT_DIR = 'content/converted_codebase'
DEVICE = 'cpu'
QDRANT_DIR = 'content/local_qdrant'
QDRANT_COLLECTION = 'my_docs'
EMBEDDINGS = 'BAAI/bge-small-en-v1.5'
MODEL = 'codellama/CodeLlama-7b-Instruct-hf'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
PROMPT = """You are a helpful modern C++ coding assistant. 
You should only answer the question once and not have any text after the answer is done. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information. 
Your answer will by short, not more than 100 words, you won't repeat yourself, but you can also include examples of code."""

INSTRUCTION = """CONTEXT:\n\n {context}\n\nQuestion: {question}"""

def get_prompt(instruction=INSTRUCTION, new_system_prompt=PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def filter_files(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if "Source" in root and (file.endswith('.cpp') or file.endswith('.hpp') or file.endswith('.h')):
                print(file)
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                new_root = os.path.join(dst_dir, os.path.dirname(rel_path))
                os.makedirs(new_root, exist_ok=True)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            data = f.read()
                    except UnicodeDecodeError:
                        print(f"Failed to decode the file: {file_path}")
                        continue
                # use .txt extension for the TextLoader
                new_file_path = os.path.join(new_root, file + '.txt')
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(data)

def wrap_text_preserve_newlines(text, width=80):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

print("Loading embeddings")
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDINGS,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings":True}
)

print("Converting input files")
filter_files(INPUT_DIR, CONVERT_DIR)

print("Loading files")
src_dir = "content/converted_codebase"
loader = DirectoryLoader(src_dir, show_progress=True, loader_cls=TextLoader)
repo_files = loader.load()
print(f"Number of files loaded: {len(repo_files)}")
if len(repo_files) < 1:
    exit(1)

print("Splitting files into chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=DOC_CHUNK_SIZE, chunk_overlap=DOC_CHUNK_OVERLAP)
documents = text_splitter.split_documents(documents=repo_files)
print(f"Number of documents : {len(documents)}")
if len(documents) < 1:
    exit(1)

# fix extension for metadata informations
for doc in documents:
    old_path_with_txt_extension = doc.metadata["source"]
    new_path_without_txt_extension = old_path_with_txt_extension.replace(".txt", "")
    doc.metadata.update({"source": new_path_without_txt_extension})

print("Populating vector database")
qdrant = Qdrant.from_documents(
    documents,
    embeddings,
    path=QDRANT_DIR,
    collection_name=QDRANT_COLLECTION,
    force_recreate=True
)

retriever = qdrant.as_retriever()

print("Loading model")
llm = HuggingFacePipeline.from_model_id(
    model_id=MODEL,
    task="text-generation",
    device_map=DEVICE,
    pipeline_kwargs={"max_new_tokens": MAX_TOKEN_GENERATED}
)

print("Looking for matching documents")
found_docs = qdrant.similarity_search(QUESTION)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
print(format_docs(found_docs))

print("Building prompt")
prompt_template = get_prompt()
llama_prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

print("Building the chain")
chain = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=llm,
    retriever=retriever,
    verbose=True,
    return_source_documents=True,
    chain_type_kwargs={"prompt": llama_prompt, "verbose": True}
)

print("Querying chain")
result = chain(QUESTION)

print("Processing answer")
process_llm_response(result)

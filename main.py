from fastapi import FastAPI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model_name="llama-3.3-70b-versatile")

DB=None
app=FastAPI()
@app.post('/generate')
def get_rag_op(JD:str) -> str:
    global DB
    if DB is None:
        data=None
        with open('shl_assessment_links.json', 'r') as file:
            data = json.load(file)
        
        URLs=[joda['url'] for joda in data]
        print("starting docs loading")
        initial_docs=WebBaseLoader(
            web_paths=URLs,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("col-12 col-md-8",)
                )
            ),
        ).load()
        print("docs loaded")
        
        DB=Chroma.from_documents(initial_docs,embedding=GPT4AllEmbeddings(),collection_name="RAG")

    docs=DB.similarity_search(JD,k=10)
    prompt=f"""
        You are a helpful QA assistant with access to multiple assessments for various job roles.
        from the provided docs use relevant documents and generate a well crafted response such that It looks like a Human answer.
        if the context is empty return "Assessment not available", but do not give incorrect response.\n\n
        <docs>{docs}</docs>\n
        <Job requirement>{JD}</job requirement>
        Answer:[]"""
    res=llm.invoke(prompt)
    return res.content

# import uvicorn
# uvicorn.run(app, host='localhost', port=8000)
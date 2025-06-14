from fastapi import FastAPI, Form,HTTPException,UploadFile,File
from typing import Literal, Optional,List
import uuid
import asyncio

from langchain.prompts import PromptTemplate
from pydantic import BaseModel,Field
from langchain.output_parsers import PydanticOutputParser
from pathlib import Path
import os
from DataPreprocessing.llm import get_llm_model
from fastapi.responses import JSONResponse
api=FastAPI()
import configparser
from langchain_core.messages import SystemMessage,HumanMessage
from DataPreprocessing.data_convert import document_to_text

from langchain_text_splitters import RecursiveCharacterTextSplitter
config=configparser.ConfigParser()
config.read('config.properties')

#below will be used to 
class Information(BaseModel):
    topic:str=Field(...,description='Unique topic from document')
    short_description:str=Field(...,description='Short Description about the context')
    long_description:str=Field(...,description='Long Description about the context')
    overview:str=Field(...,description='High-level summary of the content')
    metadata:str=Field(...,description='One topic,About the author or Date/time')
    #Field(...) → The ... means it’s required.
    #description="..." → A label for humans. It explains what this field is for.
  
parser = PydanticOutputParser(pydantic_object=Information)  
    
 #Directory to store uploaded folder
uploads = "uploads"
output_folder_base = "output_folder"
os.makedirs(uploads,exist_ok=True)
    
#encoding



@api.post("/get_info")
async def get_basic_info (
               query:str=Form(None),
                prompt:str=Form(None),
                session_id:str=Form(None),
                ocr_method:str=Form('tesseract'),
                files: List[UploadFile] = File(...)
):
    
    session_id=uuid.uuid4()
    
       # Create directories for session
    cur_uploads_path = f'{uploads}/{session_id}' #where all folder will get uploaded
    output_folder_path = f'{output_folder_base}/{session_id}'   #where all output json will be stored
    os.makedirs(cur_uploads_path,exist_ok=True) 
    
    if len(files)==0:
        raise HTTPException(status_code=400,detail="NO FILE PROVIDED")
    print(f'New session created :{session_id}')
    
    for file in files:
        file_location = f'{cur_uploads_path}/{file.filename}'
        with open(file_location, "wb") as f:
            f.write(await file.read())
    
    
    
    if query:
        prompt = f"""You are a smart AI assistant. A user is specifically looking for information about **'{query}'** in the content extracted from the uploaded document.\n\n"
        
        "Your task is to generate a structured JSON response containing the following keys:\n"
        "- `topic`: The main subject of the document, focusing on the query.\n"
        "- `short_description`: A brief explanation addressing the query if relevant.\n"
        "- `long_description`: A detailed answer or exposition centered on the query, using all available context from the document.\n"
        "- `overview`: A high-level summary of the document's overall content, highlighting parts that relate to the query.HEre add almost all the content possible\n"
        "- `metadata`: Any relevant details such as author, document title, creation date, section name, etc.\n\n"

        "-Focus **primarily** on the content that is most relevant to the query.\n"
        "- If the query is explicitly mentioned, extract detailed and relevant information.\n"
        "- If the query is **not directly mentioned** but **contextually related**, infer intelligently and clearly note this.\n"
        "- If the query is **not relevant at all** to the document's content, then politely indicate that in the output, and provide a general summary instead.\n\n"

        "Make sure the output is factual, concise, and captures the essence of the query within the document context."""
         
    else:
       prompt = f"""You are a smart AI assistant. Generate structured JSON output with the following keys:"
                "topic, short_description, long_description, overview, and metadata. "
                "Each key should be filled with relevant information extracted from the provided document."""
    
        
    
    model_name=config['Model']['model_name_gemini']
    llm=get_llm_model(model_name)
    
    print(len(files))
    print(f'query:{query}')
    get_results=await asyncio.gather(*[async_from_get_basic_info(query,llm,file,ocr_method,session_id,prompt,output_folder_path,cur_uploads_path) for file in files])
    return get_results
    
async def async_from_get_basic_info(query,llm,file,ocr_method,session_id,prompt1,output_folder_path,cur_uploads_path):
    try:
        return await asyncio.to_thread(generate_get_basic_info,query,llm,file,ocr_method,session_id,prompt1,output_folder_path,cur_uploads_path)
    except Exception as e:
        print("Exception occured ",e)
        
        
def generate_get_basic_info(query,llm,file,ocr_method,session_id,prompt1,output_folder_path,cur_uploads_path):
    
    # Save uploaded files
    
    """file_location = f'{cur_uploads_path}/{file.filename}'
    with open(file_location, "wb") as f:
        f.write( file.read())"""
    print(f"uploaded_folder_path:{cur_uploads_path}")
    all_text_content=document_to_text(file,ocr_method,session_id,output_folder_path,cur_uploads_path)
    
    
    #chunking_text
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, #2000 chars per chunk
    chunk_overlap=400,  
    length_function=len,
    is_separator_regex=False)
    
    texts = text_splitter.create_documents(all_text_content)
    prompt2=f"""You are a highly capable AI assistant.
Below is the **extracted context** from a document (could be a PDF, DOCX, etc.). Your job is to analyze this text and use it to respond to user queries with factual, structured, and insightful responses.
Extracted Document Content:
{texts}
Your task is to:
1. **Carefully read and understand** the content provided above.
2. Use this context to generate accurate answers to any user query.
3. If a user provides a query, find **relevant parts of the text** that directly or indirectly relate to it.
4. If the query is not mentioned, look for **conceptually related** content and explain your reasoning.
5. If the query has **no relevance** to the content, say so clearly but still summarize the document helpfully.
"""f"""
You are a highly capable AI assistant.
Below is the **extracted context** from a document (could be a PDF, DOCX, etc.). Your job is to analyze this text and use it to respond to user queries with factual, structured, and insightful responses.
Extracted Document Content:
{texts}
Your task is to:
1. **Carefully read and understand** the content provided above.
2. Use this context to generate accurate answers to any user query.
3. If a user provides a query, find **relevant parts of the text** that directly or indirectly relate to it.
4. If the query is not mentioned, look for **conceptually related** content and explain your reasoning.
5. If the query has **no relevance** to the content, say so clearly but still summarize the document helpfully.
"""
   
    prompt=prompt1+prompt2
    llm=llm.with_structured_output(Information)
    output=llm.invoke(prompt)
    #print(f'output:{output}')
    return output
    
    
    

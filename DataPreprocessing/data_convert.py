from docx import Document as DocxDocument
from pathlib import Path
import os
import fitz  # PyMuPDF
import shutil
from PIL import Image
import pytesseract
import configparser
import os

config=configparser.ConfigParser()
config.read('config.properties')

try:
    tess_path = config["Model"]["tesseract"]
except KeyError:
    raise KeyError(
        "Missing [Model] section or 'tesseract' key in config.properties"
    )
tess_path = os.path.expanduser(os.path.expandvars(tess_path))


pytesseract.pytesseract.tesseract_cmd = tess_path


def document_to_text(file,ocr_method, session_id, output_folder_path, cur_uploads_path):
    cur_uploads_path = Path(cur_uploads_path)
    output_folder_path = Path(output_folder_path)
    text = ""
    #print("file",file)
    for file_path in cur_uploads_path.iterdir():
        extension = file_path.suffix.lower()
        #print(f"Processing: {file_path.name} | Type: {extension}")
        
        if extension in ['.png','.jpg','.jpeg']:
            try:
                text = pytesseract.image_to_string(Image.open(file_path))
            except Exception as e:
                print(f"Exception occured: {e}")
                
        elif extension == '.docx':
            print('Reading .docx using python-docx')
            doc = DocxDocument(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif extension in ['.doc', '.odt', '.rtf', '.txt']:
            print(f'Reading text from {extension} as plain text')
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

        elif extension == '.pdf':
            print('Reading .pdf using PyMuPDF')
            with fitz.open(file_path) as pdf_doc:
                for page in pdf_doc:
                    text += page.get_text()

        else:
            raise ValueError(f"Unsupported file extension: {extension}")


    os.makedirs(output_folder_path, exist_ok=True)

    # Save to .md file
    md_filename = f"{session_id}.md"
    md_path = output_folder_path / md_filename
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(text)

    # Cleanup
    shutil.rmtree(cur_uploads_path)

    return text

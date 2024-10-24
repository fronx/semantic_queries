from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from pdf2image import convert_from_path
from PIL import Image
import io
import base64
import json

def upload_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        upload_response = client.files.create(file=pdf_file, purpose='user_data')
        # print(upload_response)
        file_id = upload_response.id
    
    return file_id

def ocr_pdf(file_id):
    prompt = """
        Please OCR this PDF file and split the content into the following sections:
        - Header: The top section of the page, typically containing titles or headers.
        - Footer: The bottom section of the page, often containing page numbers or footnotes.
        - Main text: The primary content of the page.

        Convert headlines as headlines and provide the output in the following JSON format:
        {{
            "page_number": page_number,
            "header": "Text of the header",
            "footer": "Text of the footer",
            "main_text": "Main content of the page",
            "headlines": ["Headline 1", "Headline 2", ...]
        }}
        """

    response = client.chat.completions.create(model="gpt-4",
        messages=[{
            "role": "user",
            "content": prompt,
            "attachments": [
                { "file_id": file_id }
            ]
        }])

    return response


if __name__ == '__main__':
    # file_id = upload_pdf('./pdfs/GW 9 Hegel, Phänomenologie des Geistes [unlocked].pdf')
    # file_path = './pdfs/Zeitlichkeit bei Bergson 4.pdf'
    # file_id = upload_pdf(file_path)
    print(ocr_pdf('file-4AXYiDJAoDqx1GR2fN9CRBXk'))

import pdfplumber
from PyPDF2 import PdfReader
from openai import OpenAI
import json
import argparse
import requests
from io import BytesIO
import os

class PDFParser:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_text_content(self, plumber_pdf):
        """Extract text content from all PDF pages"""
        text_content = ""
        try:
            for page in plumber_pdf.pages:
                text = page.extract_text()
                if text:
                    text_content += text + "\n"
        except Exception as e:
            print(f"Warning: Error extracting text from page: {str(e)}")
        return text_content

    def parse_pdf(self, pdf_path=None, pdf_url=None):
        """Parse PDF from either local file or URL"""
        try:
            if pdf_url:
                # Handle URL-based PDF
                response = requests.get(pdf_url)
                pdf_bytes = BytesIO(response.content)
                plumber_pdf = pdfplumber.open(pdf_bytes)
            else:
                # Handle local PDF file
                plumber_pdf = pdfplumber.open(pdf_path)

            # Extract text content
            text_content = self.extract_text_content(plumber_pdf)
            
            # Close the plumber PDF
            plumber_pdf.close()

            if not text_content.strip():
                return {"error": "No text content found in PDF"}

            return self.extract_structured_data(text_content)

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def extract_structured_data(self, text):
        """Use GPT-4 to extract structured data from the text"""
        messages = [
            {
                "role": "system",
                "content": """You are an expert at extracting information from documents.
                Extract relevant identification information while maintaining accuracy.
                If you can't find specific fields, exclude them from the response."""
            },
            {
                "role": "user",
                "content": f"""Extract any of the following information if present in the text:
                - Name
                - Date
                - Title
                - Keywords
                - Blog Name
                - Other relevant identification fields
                
                Only include fields that are clearly present in the text.
                Do not make assumptions or include uncertain information.
                Format the response as a clean JSON object.
                
                Text to analyze:
                {text}"""
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=messages,
                temperature=0,
                max_tokens=500,
                response_format={ "type": "json_object" }  # Enforce JSON response
            )
            
            # Get the response content
            content = response.choices[0].message.content
            
            # Ensure we have valid JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, return a cleaned up version of the response
                return {
                    "error": "Failed to parse structured data",
                    "raw_content": content
                }

        except Exception as e:
            print(f"Warning: Error in GPT processing: {str(e)}")
            return {
                "error": "Failed to process document content",
                "details": str(e)
            }

def main():
    parser = argparse.ArgumentParser(description="Extract information from PDF documents")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--url", type=str, help="URL to the PDF file")
    
    args = parser.parse_args()
    
    if not args.pdf_path and not args.url:
        print("Please provide either --pdf_path or --url")
        return

    try:
        pdf_parser = PDFParser()
        result = pdf_parser.parse_pdf(
            pdf_path=args.pdf_path,
            pdf_url=args.url
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
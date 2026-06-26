from pypdf import PdfReader

def pdf_to_text(pdf_path):

    try:
        reader = PdfReader(pdf_path)

        pages = []

        for page in reader.pages:

            content = page.extract_text()

            if content:
                pages.append(content)

        return "\n".join(pages)

    except FileNotFoundError:
        raise FileNotFoundError("File not found.")

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":

    pdf_path = "test.pdf"

    text = pdf_to_text("C:\\Users\\SE\\Documents\\Sehaj program folders\\My Own Projects\\Ai_study_assistant\\rag_backend\\test.pdf")

    print(text)
from docx import Document


def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


try:
    text = read_docx(r"D:\Projects\BDH\23UG013_ProjectPhase.docx")
    with open("doc_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully wrote {len(text)} characters to doc_content.txt")
except Exception as e:
    print(f"Error: {e}")

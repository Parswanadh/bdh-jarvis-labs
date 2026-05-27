from docx import Document


def read_docx_full(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


try:
    text = read_docx_full(r"D:\Projects\BDH\23UG013_ProjectPhase (1).docx")
    with open("doc_full_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully wrote {len(text)} characters to doc_full_content.txt")
except Exception as e:
    print(f"Error: {e}")

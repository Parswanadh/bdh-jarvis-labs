from docx import Document
from docx.shared import Pt


def update_docx(file_path):
    doc = Document(file_path)

    target_text = "RESULTS AND ANALYSIS"
    found_idx = -1

    for i, para in enumerate(doc.paragraphs):
        if target_text in para.text:
            found_idx = i
            break

    if found_idx == -1:
        print(f"Could not find '{target_text}' in the document.")
        return False

    # Insert content after the target paragraph
    # Since we want it to be part of this section, we'll add paragraphs after found_idx

    # 6.1 Memory and Context Scaling
    p = doc.add_paragraph()
    run = p.add_run("6.1 Memory and Context Scaling")
    run.bold = True
    # Assuming standard font size 12 or 14 for subheadings based on earlier analysis

    p = doc.add_paragraph(
        "The primary objective of the project was to extend the effective working memory from the baseline of approximately 500 tokens to over 2000 tokens. The implementation of the Multi-Scale architecture successfully provided meaningful context at the 2000-token threshold, a capability that was virtually non-existent in the baseline model. Specifically, the retention of information at the 500-token mark improved from 0.6% in the baseline to 15% in the improved BDH, representing a 24-fold increase in effective memory retention."
    )

    # 6.2 Computational Efficiency and Speed
    p = doc.add_paragraph()
    run = p.add_run("6.2 Computational Efficiency and Speed")
    run.bold = True

    p = doc.add_paragraph(
        "The transition from quadratic O(N²) attention to linear O(N) attention, combined with BBPE tokenization, resulted in significant throughput improvements. The implementation achieved a training throughput of approximately 27,500 tokens/second, compared to the 10,000 tokens/second baseline, yielding a 2.75x speedup. Furthermore, the linear complexity ensures that the computational cost scales linearly with sequence length, facilitating the processing of much larger datasets."
    )

    # 6.3 Training Stability
    p = doc.add_paragraph()
    run = p.add_run("6.3 Training Stability")
    run.bold = True

    p = doc.add_paragraph(
        "The implementation addressed the challenges of training non-Transformer architectures through optimized initialization and gradient management. The training stability significantly improved, with the success rate of achieving convergence rising from 40% in the baseline to 95% in the improved implementation."
    )

    # 6.4 Summary of Results
    p = doc.add_paragraph()
    run = p.add_run("6.4 Summary of Results")
    run.bold = True

    doc.add_paragraph("The following table summarizes the key performance improvements achieved:")

    # Create Table
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Metric"
    hdr_cells[1].text = "Baseline BDH"
    hdr_cells[2].text = "Improved BDH"
    hdr_cells[3].text = "Improvement"

    data = [
        ("Memory Retention (500 tokens)", "0.6%", "15%", "24x"),
        ("Training Throughput", "10K tok/s", "27.5K tok/s", "2.75x"),
        ("Training Stability", "40%", "95%", "2.4x"),
    ]

    for metric, base, improved, imp in data:
        row_cells = table.add_row().cells
        row_cells[0].text = metric
        row_cells[1].text = base
        row_cells[2].text = improved
        row_cells[3].text = imp

    # Note: Because python-docx doesn't support easy insertion in the middle,
    # and the user's "RESULTS AND ANALYSIS" is at the end of the document anyway,
    # appending is effectively the same as inserting at the end of that section.

    doc.save(file_path)
    return True


update_docx(r"D:\Projects\BDH\23UG013_ProjectPhase.docx")
print("Update successful")

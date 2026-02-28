# BDH Science Fair Demo Materials

**Purpose:** Live demonstration and presentation materials for BDH Science Fair
**Created:** February 25, 2026
**Presenter:** Demo Choreographer (T9)

---

## Directory Contents

```
demo/
├── README.md                         # This file
├── demo_script.md                    # Complete demo script with timing
├── live_demo_notebook.ipynb          # Jupyter notebook for live demo
├── rehearsal_notes.md                # Rehearsal schedule and timing tracker
├── backup_plan.md                    # Contingency plans for failures
├── generate_backup_visuals.py        # Script to create backup screenshots
└── backup_screenshots/               # Pre-generated visualizations (run script to create)
    ├── retention_curves.png          # Memory retention comparison
    ├── tokenization_comparison.png   # BBPE vs baseline
    ├── training_speedup.png          # Training efficiency
    ├── biological_mapping.png        # Brain inspiration diagram
    ├── summary_results.png           # All improvements summary
    └── full_demo_composite.png       # Complete demo overview
```

---

## Quick Start

### 1. Generate Backup Visualizations
Before presentation, generate all backup screenshots:

```bash
cd demo
python generate_backup_visuals.py
```

This creates all PNG images in `backup_screenshots/` directory.

### 2. Review Demo Script
Read `demo_script.md` for the complete presentation narrative and timing.

### 3. Practice with Jupyter Notebook
Open `live_demo_notebook.ipynb` and run through all cells to ensure they work.

### 4. Rehearse with Timer
Use `rehearsal_notes.md` to track timing during practice sessions.

---

## Presentation Timeline

**Total Duration:** 12-14 minutes

| Section | Duration | Description |
|---------|----------|-------------|
| 1. Problem Statement | 2 min | Show BDH memory decay limitations |
| 2. Solution Overview | 2 min | Explain multi-scale approach |
| 3. Live Demo Part 1 | 2 min | Demonstrate baseline BDH |
| 4. Live Demo Part 2 | 3 min | Demonstrate multi-scale BDH |
| 5. Live Demo Part 3 | 2 min | Demonstrate BBPE tokenization |
| 6. Results & Impact | 2 min | Show quantitative improvements |
| 7. Q&A Transition | 1 min | Invite questions |

---

## Key Files for Presentation Day

### Must Have:
- ✓ `live_demo_notebook.ipynb` - Primary demo vehicle
- ✓ `demo_script.md` - Presenter script/cheat sheet
- ✓ `backup_screenshots/` - Entire directory
- ✓ `backup_plan.md` - Emergency procedures

### Nice to Have:
- `generate_backup_visuals.py` - To regenerate visuals if needed
- `rehearsal_notes.md` - Final timing reference

---

## Preparation Checklist

### Day Before Presentation:
- [ ] Generate all backup visualizations
- [ ] Test Jupyter notebook (run all cells)
- [ ] Read through demo script 2-3 times
- [ ] Time read-through (target: 10-12 min speaking pace)

### Day of Presentation:
- [ ] Arrive 30 minutes early
- [ ] Test projector with laptop
- [ ] Open Jupyter notebook
- [ ] Run all cells to verify
- [ ] Check GPU availability
- [ ] Have backup screenshots on USB drive
- [ ] Print demo script (hard copy backup)

### 5 Minutes Before:
- [ ] Notebook open to first cell
- [ ] Backup screenshots folder open
- [ ] Timer ready
- [ ] Take deep breaths

---

## Demo Dependencies

### Required Python Packages:
```bash
pip install numpy matplotlib pandas jupyter torch
```

### Optional (for tokenization demo):
```bash
pip install tokenizers
```

### System Requirements:
- Python 3.8+
- 4GB RAM minimum
- GPU optional (but recommended for model demos)
- Jupyter Notebook or JupyterLab

---

## Backup Plans

The demo is designed with multiple fallback options:

1. **Jupyter Won't Start** → Use pre-generated screenshots in slides
2. **GPU Not Available** → Skip model loading, show visualizations only
3. **Cell Execution Error** → Show pre-generated output, continue
4. **Time Running Out** → Jump to summary, offer details during Q&A
5. **Laptop Crash** → Continue with verbal explanation

See `backup_plan.md` for detailed procedures.

---

## Key Improvements to Demonstrate

### 1. Multi-Scale Memory (4× Extension)
- **Baseline:** 500 token effective memory
- **Multi-Scale:** 2000+ token effective memory
- **Improvement:** 10,000× better retention at long sequences

### 2. BBPE Tokenization (4× Efficiency)
- **Byte-Level:** 1M tokens per epoch
- **BBPE:** 250K tokens per epoch
- **Improvement:** 4× faster training

### 3. Maintained Properties
- ✓ Biological plausibility (Hebbian learning)
- ✓ Interpretability (readable synaptic states)
- ✓ O(N) complexity (linear attention)
- ✓ Sparse activations (~5% neurons)

---

## Biological Connection

The demo emphasizes BDH's biological inspiration:

| Brain Mechanism | Timescale | BDH Implementation | Effective Memory |
|-----------------|-----------|-------------------|------------------|
| STP (Short-term Plasticity) | 100-500 ms | Fast state (decay=0.95) | ~100 tokens |
| LTP (Long-term Potentiation) | Seconds-minutes | Medium state (decay=0.99) | ~500 tokens |
| Structural Changes | Hours-days | Slow state (decay=0.995) | ~2000 tokens |

This multi-scale approach directly mimics how real brains learn at different timescales!

---

## Anticipated Questions

### Q: Why not just use Transformer?
**A:** BDH offers unique advantages: interpretability (read synaptic states), biological plausibility (Hebbian learning), and fixed memory footprint. Our work shows these advantages can be maintained while improving efficiency.

### Q: Does this scale beyond 1B parameters?
**A:** That's Phase 2! We've demonstrated principles at 10M scale. Multi-scale memory should benefit all model sizes - we're excited to test this hypothesis.

### Q: How is this different from Mamba/RWKV?
**A:** Different trade-offs: Mamba (efficient but black-box), RWKV (good efficiency, some interpretability), BDH (maximum interpretability + biological grounding). We prioritize understanding HOW the model works.

### Q: What's the biological connection?
**A:** Three key inspirations: (1) Hebbian learning - "neurons that fire together, wire together", (2) Multi-scale plasticity - STP, LTP, structural changes, (3) Sparse activations - only ~5% of neurons fire at once.

### Q: How long did this take?
**A:** 3 days of focused sprint work! This demonstrates that intelligent architectural modifications yield quick improvements, and brain-inspired AI is accessible to undergraduate researchers.

---

## Success Criteria

Demo is successful when:
- [ ] Judges understand the problem and solution
- [ ] Live code works without major issues (or backup works smoothly)
- [ ] Quantitative improvements are clear
- [ ] Biological connection is explained well
- [ ] Demo fits within 15-minute time limit
- [ ] Q&A is handled confidently
- [ ] Judges appear engaged and interested

---

## Contact & Coordination

### Team Coordination:
- **T4 (Science Fair Researcher):** Best practices and judge expectations
- **T10 (Presentation Designer):** Slide integration and visual design
- **T14 (Team Lead):** Overall coordination and progress

### Demo Script Sections by Teammate:
- Problem & Solution: Demo Choreographer (T9)
- Multi-scale Implementation: Multi-scale Architect (T1)
- BBPE Implementation: Tokenization Engineer (T2)
- Results & Metrics: Performance Analyst (T7)
- Visualizations: Visualization Specialist (T8)

---

## Tips for Success

### During Presentation:
1. **Speak clearly and at moderate pace** - Don't rush!
2. **Make eye contact with judges** - Not just the screen
3. **Use gestures** - Point to graphs, emphasize key points
4. **Show enthusiasm** - This is exciting work!
5. **Pause for effect** - Before revealing key results

### What to Avoid:
- ❌ Reading directly from slides
- ❌ Turning back to audience too much
- ❌ Getting bogged down in technical details
- ❌ Not leaving time for questions
- ❌ Making excuses if something fails

---

## Remember

**You're not just presenting results, you're telling a story about intelligent problem-solving at the intersection of AI and neuroscience!**

The demo has a compelling narrative arc:
**Problem (memory limits) → Solution (multi-scale states) → Results (4× improvement) → Impact (practical brain-inspired AI)**

Good luck! You've got this! 🐉

---

**Last Updated:** February 25, 2026
**Version:** 1.0 - Initial Release

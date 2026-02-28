# Live Demo Best Practices Guide
**BDH Science Fest Sprint**
**Date:** 2026-02-25
**Prepared by:** Science Fair Researcher (T12)

---

## The "Tour Guide" Philosophy

A live demo is not a lecture - it's a guided tour. You're the expert showing visitors around your creation.

**Key Principle:** Your audience should never wonder "What am I looking at?" or "What's happening now?"

---

## Pre-Demo Checklist

### Technical Preparation ✓

**24 Hours Before:**
- [ ] Run full demo 3 times end-to-end
- [ ] Time each section (target: 5-7 minutes total)
- [ ] Test all visualizations render correctly
- [ ] Verify dataset files exist and load correctly
- [ ] Check Python environment and dependencies
- [ ] Pre-compute all expensive operations

**1 Hour Before:**
- [ ] Restart computer (fresh state)
- [ ] Open all needed files and applications
- [ ] Set browser tabs to demo materials
- [ ] Check internet connection (if needed)
- [ ] Position backup device nearby
- [ ] Test projector/display connection

**Right Before:**
- [ ] Close unnecessary applications
- [ ] Clear desktop clutter
- [ ] Set font sizes to 18pt+
- [ ] Have water nearby
- [ ] Take 3 deep breaths

---

## Environment Setup

### Jupyter Notebook Configuration

```python
# SET THESE AT THE TOP OF YOUR NOTEBOOK

# Display settings for readability
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3

# Pandas display settings
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 50)

# NumPy settings for reproducibility
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# PyTorch settings (if using)
import torch
torch.manual_seed(42)
```

### Font Size Guide

```
Code Editor:          18-20 pt
Jupyter Notebook:     16-18 pt
Terminal Output:      14-16 pt
Presentation Slides:  24+ pt
```

**Rule:** If you have to squint to read it, the judges can't read it either.

---

## Demo Structure

### Recommended Flow (5-7 minutes)

```
┌─────────────────────────────────────────────────────────┐
│ INTRODUCTION (30 seconds)                               │
│ "Today I'll show you how Multi-Scale BDH achieves      │
│  13x better memory retention."                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ BASELINE DEMO (1 minute)                                │
│ "First, let's see how the baseline BDH performs..."    │
│ - Load baseline model                                   │
│ - Show state decay visualization                        │
│ - Run quick retention test                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ MULTI-SCALE DEMO (2 minutes)                            │
│ "Now let's see our Multi-Scale approach..."            │
│ - Load multi-scale model                                │
│ - Show three-scale state matrix                         │
│ - Run same retention test                               │
│ - Show side-by-side comparison                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ EXPLANATION (1 minute)                                  │
│ "Let me show you WHY this works..."                    │
│ - Highlight key code sections                           │
│ - Explain the three decay rates                         │
│ - Show neuroscience connection                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ INTERACTIVE ELEMENT (1 minute)                          │
│ "Would you like to see it remember something specific?"│
│ - Custom query demonstration                            │
│ - Live state matrix inspection                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ CONCLUSION (30 seconds)                                 │
│ "So we've achieved 13x improvement with only 3%        │
│  overhead - making brain-inspired AI more viable."     │
└─────────────────────────────────────────────────────────┘
```

---

## Scripting Your Demo

### Introduction Template

```
"Hi! Today I'm going to show you Multi-Scale BDH in action.

As you can see on my screen, I have a BDH model loaded.
BDH is a brain-inspired AI architecture that uses Hebbian learning -
'neurons that fire together, wire together.'

The challenge we're addressing is that BDH has a memory wall.
After just 500 tokens, it forgets 95% of information.
Let me show you what that looks like..."
```

### During Demo - What to Say

**When loading code:**
```
"Now I'm loading our baseline BDH model. This is the original
architecture with a single synaptic state."
```

**When running computations:**
```
"I'm now processing a 2000-token sequence about Paris.
Watch how the synaptic state matrix evolves..."
```

**When showing results:**
```
"As you can see from this heatmap, the state is mostly dark -
meaning the information has decayed. The model has forgotten
the key facts it was given earlier."
```

**When showing comparison:**
```
"Now let's look at our Multi-Scale version side-by-side.
On the left, the baseline - mostly dark. On the right,
Multi-Scale - you can see clear clusters of active memory
in the slow-decay state. This is the 13x improvement in action."
```

---

## Handling Code in Demos

### DO: Highlight Key Sections

Use visual emphasis to draw attention:

```python
# BAD: Just showing the code
def forward(self, x):
    state = self.state * decay + lr * (q @ k.t())

# GOOD: Highlight the important part
def forward(self, x):
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # KEY: Multi-scale update with 3 decay rates
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fast = self.fast_state * 0.90 + lr * (q @ k.t())
    medium = self.medium_state * 0.99 + lr * (q @ k.t())
    slow = self.slow_state * 0.999 + lr * (q @ k.t())
    #        ^^^^^^^^^^^
    #        BIO-GROUNDED DECAY RATES
```

### DON'T: Show Walls of Text

```python
# AVOID: Long functions without context
def multi_scale_bdh_forward(query, key, value, fast_decay, medium_decay,
                            slow_decay, learning_rate, use_normalization,
                            dropout_rate, activation_function, ...):
    # 50 lines of complex logic
```

**Instead:** Break it down and show pieces

```python
# BETTER: Show one concept at a time
# Step 1: Fast state update (short-term memory)
fast_state = fast_state * 0.90 + lr * attention

# Step 2: Medium state update (long-term potentiation)
medium_state = medium_state * 0.99 + lr * attention

# Step 3: Slow state update (structural consolidation)
slow_state = slow_state * 0.999 + lr * attention
```

---

## Visual Excellence

### Heatmap Best Practices

```python
# GOOD: Clear, readable heatmap
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Consistent color scale across all plots
vmin, vmax = 0, 1

im1 = axes[0].imshow(fast_state, vmin=vmin, vmax=vmax, cmap='viridis')
axes[0].set_title('Fast State (0.90)', fontsize=16)
axes[0].set_xlabel('Token Position', fontsize=12)
axes[0].set_ylabel('Memory Dimension', fontsize=12)

im2 = axes[1].imshow(medium_state, vmin=vmin, vmax=vmax, cmap='viridis')
axes[1].set_title('Medium State (0.99)', fontsize=16)

im3 = axes[2].imshow(slow_state, vmin=vmin, vmax=vmax, cmap='viridis')
axes[2].set_title('Slow State (0.999)', fontsize=16)

# Add colorbar
fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.6)

plt.suptitle('Multi-Scale Synaptic States', fontsize=18, y=1.02)
plt.tight_layout()
```

### Plot Best Practices

```python
# GOOD: Clear comparison plot
plt.figure(figsize=(12, 6))

# Plot lines with thick, visible lines
plt.plot(tokens_baseline, retention_baseline,
         'o-', linewidth=3, markersize=10,
         label='Baseline BDH', color='#e74c3c')

plt.plot(tokens_multiscale, retention_multiscale,
         's-', linewidth=3, markersize=10,
         label='Multi-Scale BDH', color='#2ecc71')

# Clear labels
plt.xlabel('Token Position', fontsize=14)
plt.ylabel('Memory Retention (%)', fontsize=14)
plt.title('Memory Retention Comparison', fontsize=16)

plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.ylim([0, 15])  # Focus on relevant range
```

---

## Interactive Elements

### Let Judges Drive

After your scripted demo, offer interactivity:

```
"Now, I'd like to show you something specific.
Is there anything you'd like to see the model remember?"

OR

"You mentioned [topic from their question].
Let me show you how our model handles that case..."
```

### Live Inspection

Show you can inspect the model:

```
"One of the advantages of BDH is interpretability.
Let me show you exactly what the model is storing..."

[Display state matrix with annotations]

"These bright spots here? That's the model remembering
the capital of France we mentioned 1000 tokens ago.
We can actually SEE what it's remembering."
```

---

## Handling Demo Failures

### The Golden Rule

**Never apologize profusely. Never say "this never happens."**

### Recovery Strategies

#### Strategy 1: The Pre-baked Pivot
```
"We're seeing a slight variation from our training data.
This actually highlights an interesting aspect of [related concept].

Let me show you the results from our controlled benchmark runs..."
[Switch to saved outputs]
```

#### Strategy 2: The Teaching Moment
```
"Interesting - this demonstrates a known behavior at extreme inputs.
Let me adjust the parameters to show you the typical case..."
[Adjust and continue]
```

#### Strategy 3: The Honest Approach
```
"I'm getting an unexpected result here. This is why we do research!
Let me show you what the expected output looks like..."
[Show saved version]
```

### Always Have Backup

**Minimum backup materials:**
- Saved Jupyter notebook with all outputs
- Screenshots of key visualizations
- PDF of slides with results
- Video recording of successful demo run (ideal)

---

## BDH-Specific Demo Tips

### The "Memory Test" Demo

```python
# Create a memorable test case
test_sequence = """
The BDH model was created by researchers at MIT in 2024.
It uses Hebbian learning inspired by neuroscience.
The key innovation is synaptic state matrices.
Now, after 1000 tokens of filler text...
[... 1000 tokens ...]
Question: What year was BDH created and by whom?
"""

# Show retrieval
response = model.generate(test_sequence + "\nAnswer:")
print(response)  # Should retrieve: "2024 by MIT researchers"
```

### The State Matrix Inspection

```python
# Show WHERE information is stored
def inspect_memory(model, query):
    # Get attention weights
    attention = model.get_attention(query)

    # Show which scales retain the information
    print("Fast state activation:", attention['fast'].sum().item())
    print("Medium state activation:", attention['medium'].sum().item())
    print("Slow state activation:", attention['slow'].sum().item())

    # Visualize
    plt.imshow(attention['slow'].detach().numpy())
```

### The BBPE Tokenization Demo

```python
# Show universal coverage
test_cases = [
    "Hello World",                    # English
    "こんにちは世界",                   # Japanese
    "Привет мир",                      # Russian
    "مرحبا بالعالم",                    # Arabic
    "🧠💡🔬",                           # Emojis
    "print('Hello')"                   # Code
]

for text in test_cases:
    tokens = bbpe.encode(text)
    print(f"{text:20} → {len(tokens)} tokens")
```

---

## Timing Tips

### If You're Running Short (< 5 min)

- Skip the baseline demo, jump straight to comparison
- Shorten the code explanation
- Skip interactive elements
- Focus on the key result visualization

### If You're Running Long (> 7 min)

- Don't show every single code cell
- Skip secondary visualizations
- Move interactive elements to Q&A period
- Wrap up sooner

---

## Body Language During Demo

### DO:
- Face the audience, not the screen
- Use pointer/hand gestures to guide attention
- Make eye contact while code runs
- Stand to the side of the screen, not in front
- Smile and show enthusiasm

### DON'T:
- Read from the screen
- Block the audience's view
- Turn your back to the audience
- Stand still - move purposefully
- Look nervous if something takes time

---

## Quick Reference: Demo Day Commands

```bash
# Start your demo environment
cd D:/projects/BDH
conda activate bdh-env

# Start Jupyter
jupyter notebook demo/live_demo_notebook.ipynb

# Or run the demo script directly
python demo/run_science_fair_demo.py
```

---

## Final Checklist for Demo Day

**Right Before Starting:**
- [ ] Deep breath
- [ ] Smile
- [ ] Make eye contact
- [ ] "Thank you for coming..."
- [ ] Start with the hook

**During Demo:**
- [ ] Narrate clearly
- [ ] Point to relevant code
- [ ] Explain visualizations
- [ ] Check for understanding
- [ ] Maintain energy

**After Demo:**
- [ ] "Any questions?"
- [ ] Thank the judges
- [ ] Offer to show more
- [ ] End on high note

---

## Remember: The Demo is Your Superpower

Most science fair projects use static posters. You have a **live, working demonstration**. This is your competitive advantage.

- It proves authenticity
- It generates excitement
- It demonstrates technical skill
- It creates memorable experiences

**You've built something amazing. Now show it off with confidence!**
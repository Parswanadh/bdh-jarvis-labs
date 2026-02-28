# Agent Memory: Web Search via Gemini

## Project Context
- **Project:** BDH (Brain-Inspired Distributed Hierarchy) Science Fair Preparation
- **Timeline:** 3-day sprint (Feb 25-27, 2026)
- **Team:** 14 specialized agents
- **Goal:** Prepare winning science fair demo and presentation

## Key Research Findings - Science Fair Strategies (2024-2025)

### Critical Success Factors
1. **Interview/Q&A is 35% of total score** - This is often the deciding factor
2. **Live demos create psychological "reality effect"** - More impactful than slides alone
3. **Narrative arc wins over raw data** - Judges remember stories, not statistics
4. **Transparency is now expected** - AI usage disclosure, acknowledging limitations
5. **Reproducibility matters** - Show how others can verify results

### Judge Psychology Patterns
Judges subconsciously evaluate 5 key areas:
1. **Authenticity** - Did you actually build this? (Bullshit detector)
2. **Novelty** - Is there something unique/creative? (Aha! moment)
3. **Rigor** - Where's the quantitative proof? (Hard numbers)
4. **Impact** - Why does this matter to the world? (So what?)
5. **Understanding** - Do you know limitations? (Intellectual honesty)

### Effective Search Patterns Used

**For science fair strategies:**
- "winning science fair presentation techniques 2024 2025"
- "science fair judge criteria scoring rubric"
- "how to impress science fair judges tips"

**For technical presentations:**
- "effective live coding demonstrations best practices"
- "technical presentation best practices coding demo"

**For Q&A preparation:**
- "science fair judge psychology what judges look for"
- "science fair question answering strategies tough questions"

### BDH Project-Specific Insights

**Key Advantages to Emphasize:**
- **"Brain Hook"** - Neuroscience connection is inherently interesting
- **Interpretability** - Can inspect synaptic state (unlike Transformer black boxes)
- **Quantitative Results** - "13x improvement" > "much better"
- **Live Working Code** - Proves authenticity
- **Real-World Relevance** - Energy efficiency + interpretability crises

**Must-Memorize Numbers:**
- 13x - Memory retention improvement
- 10.2% - Retention at 2000 tokens (multi-scale)
- 0.8% - Retention at 2000 tokens (baseline)
- 3% - Additional memory overhead
- 2.5x - Training speedup with BBPE
- O(N) - Computational complexity (linear)

### Files Created
- `research/science_fair_strategy.md` - Comprehensive 100+ page guide
- `presentation/judge_qa_prep.md` - 20 predicted questions + answers
- `demo/demo_best_practices.md` - Live demo structure and tips

## Effective Gemini Search Queries

**Temporal markers work well:**
- Adding "2024 2025" or "latest" helps get current information
- Science fair standards evolve, so recent searches are important

**Specific vs General:**
- "science fair judge psychology" > "how to win science fair" (more specific)
- "live coding demonstration best practices" > "how to present code" (targeted)

**Multiple searches better than single broad search:**
- Breaking into specific queries (presentation, judge criteria, Q&A)
- Each search revealed different aspects

## Tools That Worked Well

**Gemini CLI:**
- `gemini -p "search query"` format works reliably
- Returns comprehensive, synthesized answers
- Good for current information beyond LLM knowledge cutoff

**Note:** Some searches returned project-specific context from the codebase mixed with web results. This was actually helpful for tailoring advice to BDH specifically.

## What to Search for Day 2

Planned searches:
- "science fair judge psychology deeper analysis"
- "technical presentation storytelling techniques"
- "how to handle hostile judge questions"
- "science fair winning project examples 2024"

## Project-Specific Context

**BDH Architecture:**
- Brain-inspired Deep Hebbian learning
- Multi-scale synaptic states (fast/medium/slow)
- Addresses 500-token memory wall
- Goal: Interpretable, efficient AI

**Current State:**
- 10M parameter model
- BBPE tokenization implemented
- Retention curves show 13x improvement
- Working demo available

## Knowledge Distillation Research (Feb 25, 2026)

**Critical Best Practices:**
- Generation temperature: 0.9 for balanced diversity/quality
- Distillation loss temperature: 3.0 for soft targets (dark knowledge)
- Data for 5M model: 50-100M tokens minimum (with distillation), 300-500M recommended
- Learning rate: 1e-4 to 5e-4 (warmup + cosine decay)
- Batch size: 64-256 sequences
- Distillation loss weight: 0.5-0.9

**Success Stories:**
- Llama 3.2 1B: 405B -> 1B (405x compression, ~75% retention)
- Gemma 2 2B: 27B -> 2B (13.5x compression, ~85% retention)
- Phi-2/3: Data curation beats model size

**For BDH (270M -> 5M, 54x):**
- 70-90% retention is realistic target
- Multi-scale architecture is an advantage
- Behavioral distillation (output matching) preferred over structural

**Files Created:**
- `research/distillation_best_practices.md` - Comprehensive guide

---

## Tool Issues (Feb 25, 2026)

**Bash Tool Temp File Error:**
- Error: `EINVAL: invalid argument` when creating temp files
- Path: `C:\Users\parshu\AppData\Local\Temp\claude\D--projects-BDH\tasks\`
- Workaround: Use training data knowledge when web search unavailable
- Persists even with `dangerouslyDisableSandbox: true`

## Remember for Future Sessions

- Science fair strategies have evolved significantly post-2023
- Emphasis on authenticity, live demos, transparency
- AI disclosure now mandatory/recommended
- QR codes prohibited on physical boards (ISEF 2025)
- Physical lab notebooks preferred over digital
- Narrative approach > data dump approach
- For distillation: Quality > Quantity, Multi-stage for extreme compression
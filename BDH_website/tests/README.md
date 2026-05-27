# BDH Website — Playwright Tests

## Setup
```bash
npm install
npx playwright install chromium
```

## Run Tests
```bash
# All tests
npm test

# With UI
npm run test:ui

# Specific browser
npx playwright test --project=chromium

# View report
npm run test:report
```

## Test Coverage (18 tests)
| Test | What it verifies |
|------|-----------------|
| Page loads | Correct title, no errors |
| Hero section | Headline + both CTA buttons visible |
| Navigation links | All anchor links present and valid |
| Architecture cards | All 6 component cards render |
| Flow diagram | All 7 flow steps visible |
| Status columns | What Works / In Progress / Planned |
| Open questions | All questions render |
| Footer | Links to Paper, GitHub, Research Docs |
| External links | target="_blank" + rel="noopener" |
| Dark mode toggle | Click toggles .dark class |
| No console errors | Zero pageerror events |
| Feature cards | All 6 features visible |
| Mobile (375px) | Hamburger menu visible |
| Tablet (768px) | Desktop nav visible |
| Desktop (1280px) | 3-column grid, all 6 cards |
| Paper link | Points to arXiv:2509.26507 |
| Skip link | Accessibility skip-to-content |
| Mobile menu | Opens/closes correctly |

## Config
- `playwright.config.ts`: Chromium + Mobile Chrome + Mobile Safari projects
- Auto-starts `http-server` on port 8080
- Traces on retry, HTML reporter

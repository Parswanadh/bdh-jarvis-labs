import { test, expect } from '@playwright/test';

test.describe('BDH Website', () => {
  test('page loads with correct title', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/BDH Research/);
  });

  test('hero section has headline and CTAs', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('Architecting the');
    await expect(page.locator('h1')).toContainText('Singularity');
    const heroSection = page.locator('section').first();
    await expect(heroSection.getByRole('link', { name: 'Read the Paper' })).toBeVisible();
    await expect(heroSection.getByRole('link', { name: 'Live Demo' })).toBeVisible();
    await expect(heroSection.getByRole('link', { name: 'Explore the Code' })).toBeVisible();
  });

  test('navigation links scroll to sections', async ({ page }) => {
    await page.goto('/');
    const navLinks = ['Architecture', 'Research', 'AGI Pathway', 'Status'];
    const desktopNav = page.locator('nav .hidden.md\\:flex');
    for (const label of navLinks) {
      const link = desktopNav.getByRole('link', { name: label });
      await expect(link).toBeVisible();
      const href = await link.getAttribute('href');
      expect(href).toMatch(/^#[a-z-]+$/);
    }
  });

  test('architecture section renders all component cards', async ({ page }) => {
    await page.goto('/');
    const cards = [
      'Linear Attention',
      'Hebbian State Matrix',
      'Multi-Scale Memory',
      'Multiplicative Gating',
      'ReLU Sparsity',
      'RYS Surgery',
      'Mimetic Init',
      'Byte-Level Processing',
    ];
    const archSection = page.locator('#architecture');
    for (const card of cards) {
      await expect(archSection.getByRole('heading', { name: card })).toBeVisible();
    }
  });

  test('architecture flow diagram renders', async ({ page }) => {
    await page.goto('/');
    const flowDiagram = page.locator('#flow-diagram');
    await expect(flowDiagram.getByText('Input')).toBeVisible();
    await expect(flowDiagram.getByText('Embedding')).toBeVisible();
    await expect(flowDiagram.getByText('Linear Attn')).toBeVisible();
    await expect(flowDiagram.getByText('Hebbian State')).toBeVisible();
    await expect(flowDiagram.getByText('ReLU FFN')).toBeVisible();
    await expect(flowDiagram.getByText('Mult. Gate')).toBeVisible();
    await expect(flowDiagram.getByText('Output')).toBeVisible();
  });

  test('status section has three columns', async ({ page }) => {
    await page.goto('/');
    const statusSection = page.locator('#status');
    await expect(statusSection.getByRole('heading', { name: 'What Works' })).toBeVisible();
    await expect(statusSection.getByRole('heading', { name: 'In Progress' })).toBeVisible();
    await expect(statusSection.getByRole('heading', { name: 'Planned' })).toBeVisible();
    await expect(statusSection.getByText('Linear attention via Q')).toBeVisible();
  });

  test('open questions section renders', async ({ page }) => {
    await page.goto('/');
    const questionsSection = page.locator('#questions');
    await expect(questionsSection.getByText('Open Questions')).toBeVisible();
    await expect(questionsSection.getByText('Does the low-rank Hebbian approximation')).toBeVisible();
    await expect(questionsSection.getByText('Can BDH scale beyond 1B parameters')).toBeVisible();
  });

  test('footer renders with links', async ({ page }) => {
    await page.goto('/');
    const footer = page.locator('footer');
    await expect(footer.getByText('BDH Research — Open Source')).toBeVisible();
    await expect(footer.getByRole('link', { name: 'Paper' })).toBeVisible();
    await expect(footer.getByRole('link', { name: 'GitHub' })).toBeVisible();
    await expect(footer.getByRole('link', { name: 'Research Docs' })).toBeVisible();
  });

  test('external links open in new tab', async ({ page }) => {
    await page.goto('/');
    const externalLinks = page.locator('a[target="_blank"]');
    const count = await externalLinks.count();
    expect(count).toBeGreaterThan(0);
    for (let i = 0; i < count; i++) {
      const rel = await externalLinks.nth(i).getAttribute('rel');
      expect(rel).toContain('noopener');
    }
  });

  test('dark mode toggle works', async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('bdh-theme', 'light');
    });
    await expect(page.locator('html')).not.toHaveClass(/dark/);

    await page.click('#theme-toggle');
    await expect(page.locator('html')).toHaveClass(/dark/);
  });

  test('no console errors on page load', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('/');
    expect(errors).toEqual([]);
  });

  test('features section has 6 cards', async ({ page }) => {
    await page.goto('/');
    const features = ['Linear Scaling', 'Fixed Memory', 'Interpretable States', 'Biologically Grounded', 'Knowledge Distillation', 'Open Research'];
    const featuresSection = page.locator('#feature-cards');
    for (const feature of features) {
      await expect(featuresSection.getByText(feature)).toBeVisible();
    }
  });

  test('responsive: mobile layout (375px)', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('/');
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Open menu' })).toBeVisible();
  });

  test('responsive: tablet layout (768px)', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('.hidden.md\\:flex')).toBeVisible();
  });

  test('responsive: desktop layout (1280px)', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto('/');
    await expect(page.locator('h1')).toBeVisible();
    const cards = page.locator('#arch-cards > div');
    await expect(cards).toHaveCount(8);
  });

  test('paper link points to correct arXiv', async ({ page }) => {
    await page.goto('/');
    const paperLink = page.getByRole('link', { name: 'arXiv:2509.26507' });
    await expect(paperLink).toHaveAttribute('href', 'https://arxiv.org/abs/2509.26507');
  });

  test('skip to content link exists', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.skip-link')).toBeVisible();
    const href = await page.locator('.skip-link').getAttribute('href');
    expect(href).toBe('#main');
  });

  test('mobile hamburger menu opens and closes', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('/');
    const menuBtn = page.getByRole('button', { name: 'Open menu' });
    const menu = page.locator('#mobile-menu');

    await expect(menu).not.toHaveClass(/open/);

    await menuBtn.click();
    await expect(menu).toHaveClass(/open/);

    await menu.getByRole('link', { name: 'Architecture' }).click();
    await expect(menu).not.toHaveClass(/open/);
  });
});

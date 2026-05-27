process.env.PWTEST_CACHE_DIR = 'D:/playwright-cache';
process.env.TMP = 'D:/playwright-cache/tmp';
process.env.TEMP = 'D:/playwright-cache/tmp';
delete process.env.PLAYWRIGHT_BROWSERS_PATH;

const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  outputDir: 'D:/playwright-cache/test-results',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:8080',
    trace: 'off',
    channel: 'chrome',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        channel: 'chrome',
        viewport: { width: 1280, height: 720 },
      },
    },
  ],
  webServer: {
    command: 'npx http-server . -p 8080 -s',
    url: 'http://localhost:8080',
    reuseExistingServer: !process.env.CI,
  },
});

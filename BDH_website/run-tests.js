process.env.PWTEST_CACHE_DIR = 'D:/playwright-cache';
process.env.PLAYWRIGHT_BROWSERS_PATH = 'D:/playwright-browsers';

const { spawn } = require('child_process');
const path = require('path');

const playwright = path.join(__dirname, 'node_modules', '@playwright', 'test', 'cli.js');
const env = {
  ...process.env,
  PWTEST_CACHE_DIR: 'D:/playwright-cache',
  TMP: 'D:/playwright-cache/tmp',
  TEMP: 'D:/playwright-cache/tmp',
};
const child = spawn('node', [playwright, 'test', '--project=chromium', '--reporter=list', ...process.argv.slice(2)], {
  stdio: 'inherit',
  env,
});

child.on('close', (code) => process.exit(code));

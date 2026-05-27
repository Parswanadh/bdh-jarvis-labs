@echo off
set PWTEST_CACHE_DIR=D:\playwright-cache
set PLAYWRIGHT_BROWSERS_PATH=D:\playwright-browsers
npx playwright test %*

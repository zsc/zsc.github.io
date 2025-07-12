// screenshot.js
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// --- ⚙️ CONFIGURATION - EDIT THESE VALUES ---

// 1. Your Google AI (Gemini) API Key.
const GEMINI_API_KEY = process.env.GEMINI_API_KEY

// 2. The date range you want to capture. Format: YYYY-MM-DD
const START_DATE = '2025-06-20';
const END_DATE = '2025-06-30';

// 3. Path to your HTML file and output directory.
const HTML_FILE_PATH = 'index.html';
const OUTPUT_DIR = 'screenshots';

// --- END OF CONFIGURATION ---


/**
 * Formats a Date object into a YYYY-MM-DD string.
 * @param {Date} date The date to format.
 * @returns {string} The formatted date string.
 */
function formatDate(date) {
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const dd = String(date.getDate()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd}`;
}

/**
 * The main function to run the screenshot process.
 */
async function run() {
    if (!GEMINI_API_KEY || GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY') {
        console.error('❌ Error: Please set your GEMINI_API_KEY in the configuration section of the script.');
        return;
    }

    // Ensure output directory exists
    if (!fs.existsSync(OUTPUT_DIR)) {
        fs.mkdirSync(OUTPUT_DIR);
    }

    console.log('🚀 Starting Puppeteer...');
    const browser = await puppeteer.launch({
        // headless: false, // Uncomment for debugging to see the browser UI
    });

    try {
        const startDate = new Date(START_DATE);
        const endDate = new Date(END_DATE);

        for (let d = startDate; d <= endDate; d.setDate(d.getDate() + 1)) {
            const currentDate = new Date(d); // Create a copy to avoid mutation issues
            const dateString = formatDate(currentDate);
            const page = await browser.newPage();

            console.log(`\nProcessing date: ${dateString}...`);

            // Set a consistent viewport for screenshots
            await page.setViewport({ width: 800, height: 1080 });

            // Construct the file URL with the API key parameter
            const fileUrl = `file://${path.resolve(HTML_FILE_PATH)}?apiKey=${GEMINI_API_KEY}`;
            await page.goto(fileUrl, { waitUntil: 'networkidle0' });

            // --- Interact with the page ---

            // 1. Set the date in the date picker
            await page.evaluate((date) => {
                document.getElementById('date-picker').value = date;
            }, dateString);
            console.log('  - Set date picker value.');

            // 2. Click the fetch button
            await page.click('#fetch-button');
            console.log('  - Clicked fetch button. Waiting for content...');

            // 3. Wait for the content to load.
            // We wait for a specific element that appears only after the API call is successful.
            // This is more reliable than waiting for a fixed time.
            try {
                await page.waitForSelector('#content-container h2', { timeout: 30000 }); // 30-second timeout
                console.log('  - Content loaded successfully.');

                // 4. Take a screenshot of the main container
                const container = await page.$('.container');
                if (container) {
                    const screenshotPath = path.join(OUTPUT_DIR, `${dateString}.png`);
                    await container.screenshot({ path: screenshotPath });
                    console.log(`  ✅ Screenshot saved to: ${screenshotPath}`);
                } else {
                    console.error('  - ❌ Could not find the .container element to screenshot.');
                }
            } catch (error) {
                console.error(`  - ❌ Timed out waiting for content for date ${dateString}. The API call may have failed or taken too long.`);
                const errorContent = await page.$eval('#error-container', el => el.textContent).catch(() => 'No error message visible.');
                console.error(`  - Page error message: ${errorContent}`);
            }

            await page.close();
        }
    } catch (error) {
        console.error('An unexpected error occurred:', error);
    } finally {
        await browser.close();
        console.log('\n🎉 All tasks complete. Browser closed.');
    }
}

run();

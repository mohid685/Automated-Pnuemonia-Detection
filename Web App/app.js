const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const app = express();

// Configure EJS
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Middleware
app.use(express.static('public'));
app.use(express.urlencoded({ extended: true }));

// Visual Analysis Page with results
app.get('/visual-analysis', (req, res) => {
    // If there are no results in the session, redirect to home
    if (!req.app.locals.lastResults) {
        return res.redirect('/');
    }
    res.render('visual-analysis', {
        results: req.app.locals.lastResults
    });
});

// Home route
app.get('/', (req, res) => {
    res.render('index', {
        results: null,
        error: null
    });
});

// Analyze route with improved error handling
app.post('/analyze', (req, res) => {
    const pythonScriptPath = path.join(__dirname, 'predict.py');
    const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    const pythonProcess = spawn(pythonExecutable, [pythonScriptPath, '--batch']);

    let data = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (chunk) => {
        data += chunk.toString();
    });

    pythonProcess.stderr.on('data', (chunk) => {
        errorData += chunk.toString();
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error(`Python script exited with code ${code}`);
            console.error(`Error output: ${errorData}`);
            return res.render('index', {
                results: null,
                error: `Python script failed with error: ${errorData || 'Unknown error'}`
            });
        }

        try {
            const results = JSON.parse(data);

            if (results.error) {
                return res.render('index', {
                    results: null,
                    error: results.error
                });
            }

            // Process sample images
            if (results.samples) {
                results.samples = results.samples.map(sample => {
                    return {
                        ...sample,
                        image: `data:image/jpeg;base64,${sample.image}`
                    };
                });
            }

            // Store results in app locals for the visual analysis page
            req.app.locals.lastResults = results;

            res.render('index', {
                results,
                error: null
            });
        } catch (err) {
            console.error('Error parsing results:', err);
            res.render('index', {
                results: null,
                error: 'Failed to process analysis results. Please check server logs.'
            });
        }
    });
});

const PORT = 1900;
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
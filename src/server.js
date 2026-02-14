// File: src/server.js
import app from './app.js';
import { PORT } from './config/env.js';
import pool from './config/db.js';

// Start Server
const server = app.listen(PORT, async () => {
    try {
        // Test DB Connection
        const [rows] = await pool.query('SELECT CURRENT_TIMESTAMP as now');
        console.log(`Server running on port ${PORT}`);
        console.log(`Database connected: ${rows[0].now}`); // Rows array, first element
    } catch (err) {
        console.error('Failed to connect to the database', err);
        process.exit(1);
    }
});

// Graceful Shutdown
const shutdown = () => {
    console.log('Shutting down server...');
    server.close(() => {
        console.log('Server closed');
        pool.end(); // MySQL pool end isn't always async in the same way, but it works
        console.log('Database pool closed');
        process.exit(0);
    });
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

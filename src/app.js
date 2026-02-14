// File: src/app.js
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import cookieParser from 'cookie-parser';

import authRoutes from './routes/auth.routes.js';
import detectionRoutes from './routes/detection.routes.js';
import zoneRoutes from './routes/zone.routes.js';
import analyticsRoutes from './routes/analytics.routes.js';
import predictionRoutes from './routes/prediction.routes.js';
import actionRoutes from './routes/action.routes.js';

import { errorHandler } from './middlewares/error.middleware.js';

const app = express();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors());
app.use(helmet());
app.use(morgan('dev'));
app.use(cookieParser());

// Routes
// Prefix all API routes with /api as requested
app.use('/api/auth', authRoutes);
app.use('/api/detections', detectionRoutes);
app.use('/api/zones', zoneRoutes);
app.use('/api/analytics', analyticsRoutes);
app.use('/api/predictions', predictionRoutes);
app.use('/api/actions', actionRoutes);

// Health Check
app.get('/health', (req, res) => {
    res.status(200).json({ status: 'OK', timestamp: new Date() });
});

// Error Handling
app.use(errorHandler);

export default app;

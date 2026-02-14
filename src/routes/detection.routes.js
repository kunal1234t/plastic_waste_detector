// File: src/routes/detection.routes.js
import { Router } from 'express';
import * as detectionController from '../controllers/detection.controller.js';
// import { authenticateToken } from '../middlewares/auth.middleware.js';

const router = Router();

// Public for ML script simplicity, or uncomment middleware to secure
router.post('/', detectionController.createDetection);

export default router;

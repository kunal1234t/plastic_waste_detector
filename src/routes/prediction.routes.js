// File: src/routes/prediction.routes.js
import { Router } from 'express';
import * as predictionController from '../controllers/prediction.controller.js';
import { authenticateToken } from '../middlewares/auth.middleware.js';

const router = Router();

router.use(authenticateToken);

router.get('/', predictionController.getPredictions);

export default router;

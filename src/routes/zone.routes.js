// File: src/routes/zone.routes.js
import { Router } from 'express';
import * as zoneController from '../controllers/zone.controller.js';
import { authenticateToken } from '../middlewares/auth.middleware.js';

const router = Router();

router.use(authenticateToken);

router.get('/', zoneController.getAllZones);
router.get('/:id', zoneController.getZoneById);

export default router;

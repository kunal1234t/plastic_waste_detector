// File: src/routes/action.routes.js
import { Router } from 'express';
import * as actionController from '../controllers/action.controller.js';
import { authenticateToken } from '../middlewares/auth.middleware.js';
import { requireRole } from '../middlewares/role.middleware.js';

const router = Router();

router.use(authenticateToken);

router.post('/', requireRole('admin'), actionController.dispatchAction);
router.get('/', actionController.getActions); // Viewers can see actions? Let's say yes.

export default router;

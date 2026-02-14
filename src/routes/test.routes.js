// routes/test.routes.js
import { Router } from "express";
import { requireAuth } from "../middlewares/auth.middleware.js";

const router = Router();

router.get("/private", requireAuth, (req, res) => {
    res.json({
        message: "Auth works",
        user: req.user
    });
});

export default router;

// File: src/services/analytics.service.js
import pool from '../config/db.js';

export const getGlobalAnalytics = async () => {
    // Aggregate data: Total detections, Average Risk, Top Polluted Zones
    const totalDetectionsQuery = 'SELECT COUNT(*) as count FROM detections';
    const avgRiskQuery = 'SELECT AVG(risk_score) as avg FROM zone_metrics';
    const topZonesQuery = `
        SELECT z.name, zm.risk_score 
        FROM zones z 
        JOIN zone_metrics zm ON z.id = zm.zone_id 
        ORDER BY zm.risk_score DESC 
        LIMIT 5
    `;
    const plasticsByTypeQuery = `
        SELECT plastic_type, COUNT(*) as count 
        FROM detections 
        GROUP BY plastic_type
    `;

    const [totalRes] = await pool.query(totalDetectionsQuery);
    const [avgRes] = await pool.query(avgRiskQuery);
    const [topZonesRes] = await pool.query(topZonesQuery);
    const [typesRes] = await pool.query(plasticsByTypeQuery);

    return {
        totalDetections: totalRes[0].count,
        averageRiskScore: parseFloat(avgRes[0].avg || 0).toFixed(2),
        topPollutedZones: topZonesRes,
        wasteComposition: typesRes
    };
};

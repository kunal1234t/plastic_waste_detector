// File: src/config/db.js
import mysql from "mysql2/promise";
import { DB_CONFIG } from "./env.js";

const pool = mysql.createPool({
    host: DB_CONFIG.host,
    user: DB_CONFIG.user,
    password: DB_CONFIG.password,
    database: DB_CONFIG.database,
    port: DB_CONFIG.port,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

export default pool;

<?php
session_start();
require_once 'config.php';

// Redirect to unified diagnosis page with wheat mode
if (!isset($_SESSION['user_id'])) {
    header("Location: auth.php");
    exit();
}

// Check if there's a result to display
if (isset($_GET['status']) && isset($_GET['file']) && isset($_GET['img'])) {
    // Preserve result parameters in redirect
    header("Location: predict_plant.php?mode=wheat&status=" . urlencode($_GET['status']) . "&file=" . urlencode($_GET['file']) . "&img=" . urlencode($_GET['img']));
} else {
    // Simple redirect to wheat mode
    header("Location: predict_plant.php?mode=wheat");
}
exit();
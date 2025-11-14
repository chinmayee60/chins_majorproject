<?php
session_start();
require_once 'config.php';

// SECURITY CHECK: If user is not logged in, redirect to auth.php
if (!isset($_SESSION['user_id'])) {
    header("Location: auth.php");
    exit();
}

$user_id = $_SESSION['user_id'];
$username = $_SESSION['username'];

// Handle LOGOUT
if (isset($_GET['logout'])) {
    session_destroy();
    header("Location: auth.php");
    exit();
}

// Handle prediction status messages
$status_message = '';
$result_content = '';
if (isset($_GET['status']) && $_GET['status'] == 'success' && isset($_GET['file'])) {
    $file_path = htmlspecialchars($_GET['file']);
    if (file_exists($file_path)) {
        $result_content = file_get_contents($file_path);
        $status_message = "Analysis Complete! Your latest report has been generated.";
    } else {
        $status_message = "Prediction complete, but the result file could not be accessed.";
    }
}

// Fetch prediction history for the CURRENT USER ONLY
$history = [];
$stmt = $conn->prepare("SELECT model_type, result_file, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10");
$stmt->bind_param("i", $user_id);
$stmt->execute();
$result = $stmt->get_result();
while ($row = $result->fetch_assoc()) {
    $history[] = $row;
}
$stmt->close();
?>

<!DOCTYPE html>
<html lang="en">
<?php include 'includes/global_head_scripts.php'; ?>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Agro AI - Professional Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="style.css"> 
    <style>
        /* Specific CSS overrides for this page */
        /* NOTE: Ensure all rules from Step 1 are in your style.css */
        .main-content-area {
            margin-left: 280px; 
            padding-top: 30px;
        }
        .navbar {
            display: none; /* Hide the main horizontal navbar */
        }
    </style>
</head>
<body>

<div class="fixed-sidebar-menu">
    <div class="p-4 border-bottom">
        <h5 class="offcanvas-title" style="color: #009472; font-weight: bold;">AGRIVISION AI</h5>
    </div>
    <nav class="nav nav-pills flex-column mb-auto flex-grow-1">
      <a class="nav-link active" href="index.php">Dashboard Home</a>
      <a class="nav-link" href="predict_plant.php">🌱 Leaf Disease Detection</a>
      <a class="nav-link" href="predict_wheat.php">🌾 Stem/Root Disease Detection</a>
      <a class="nav-link" href="recommend_crop.php">📊 Crop Recommendation</a>
      <a class="nav-link" href="recommend_map.php">🗺️ Map Recommendation</a>
    </nav>
    <div class="p-4 border-top">
        <p class="small text-muted mb-1">
            Logged in as: <strong><?php echo htmlspecialchars($username); ?></strong>
        </p>
        <a href="index.php?logout=true" class="btn btn-sm btn-outline-dark w-100">Logout</a>
    </div>
</div>

<div class="container-fluid main-content-area">
    <h1 class="display-5 mb-5 text-center" style="font-weight: bold;">Welcome to the Crop Intelligence Center</h1>

    <?php if ($status_message): ?>
        <div class="alert alert-info alert-dismissible fade show" role="alert">
            <?php echo $status_message; ?>
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
        <div class="card mb-5 shadow-sm">
            <div class="card-header bg-success text-white">Latest Analysis Report</div>
            <div class="card-body">
                <pre style="white-space: pre-wrap; font-size: 0.9em;"><?php echo htmlspecialchars($result_content); ?></pre>
            </div>
        </div>
    <?php endif; ?>

    <h3 class="mt-4 mb-3 text-secondary" style="font-family: Georgia, serif; font-size: 1.5rem; border-bottom: 1px solid #eee; padding-bottom: 10px;">
        Prediction History
    </h3>
    
    <div class="history-list">
        <?php if (!empty($history)): ?>
            <ul class="list-group">
                <?php foreach ($history as $item): ?>
                    <li class="list-group-item d-flex justify-content-between align-items-center shadow-sm">
                        <span>
                            <span class="badge bg-success text-white me-2"><?php echo ucfirst($item['model_type']); ?></span>
                            Analysis completed on <?php echo date("Y-m-d H:i:s", strtotime($item['timestamp'])); ?>
                        </span>
                        <button class="btn btn-sm btn-outline-success view-result-btn" data-bs-toggle="modal" data-bs-target="#resultModal" data-file="<?php echo htmlspecialchars($item['result_file']); ?>">View Report</button>
                    </li>
                <?php endforeach; ?>
            </ul>
        <?php else: ?>
            <div class="alert alert-warning text-center mt-3">No prediction history found.</div>
        <?php endif; ?>
    </div>
</div>

<div class="modal fade" id="resultModal" tabindex="-1" aria-hidden="true">
  <div class="modal-dialog modal-lg">
    <div class="modal-content">
      <div class="modal-header bg-success text-white">
        <h5 class="modal-title">Full Analysis Report</h5>
        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <pre id="modalResultContent" style="white-space: pre-wrap; font-size: 0.95em;"></pre>
      </div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
$(document).ready(function() {
    $('.view-result-btn').on('click', function() {
        var filePath = $(this).data('file');
        
        $.ajax({
            url: filePath,
            type: 'GET',
            dataType: 'text',
            success: function(data) {
                $('#modalResultContent').text(data);
            },
            error: function() {
                $('#modalResultContent').text('Error loading file content. Check file permissions on the results/ folder.');
            }
        });
    });
});
</script>

</body>
</html>
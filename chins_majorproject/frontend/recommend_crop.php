<?php
session_start();
require_once 'config.php';
if (!isset($_SESSION['user_id'])) { header("Location: auth.php"); exit(); }
$username = $_SESSION['username'];

// Check for successful prediction result passed via URL
$has_result = false;
$result_data = [];
$uploaded_image_path = ''; // Not used for crop model

if (isset($_GET['status']) && $_GET['status'] == 'success' && isset($_GET['file'])) {
    $file_path = htmlspecialchars($_GET['file']);
    if (file_exists($file_path)) {
        $has_result = true;
        
        // 1. Read and Parse the saved text file content
        $raw_content = file_get_contents($file_path);
        
        $lines = explode("\n", $raw_content);
        $result_data['Advice'] = '';
        $result_data['Input'] = '';
        $in_advice_section = false;

        foreach ($lines as $line) {
            if (strpos($line, 'Recommended Crop:') !== false) {
                $result_data['Crop'] = trim(str_replace('Recommended Crop:', '', $line));
            } elseif (strpos($line, 'Input Conditions:') !== false) {
                $result_data['Input'] = trim(str_replace('Input Conditions:', '', $line));
            } elseif (strpos($line, '--- AI Generated Advice ---') !== false) {
                $in_advice_section = true;
            } elseif ($in_advice_section) {
                $result_data['Advice'] .= $line . "\n";
            }
        }
        
        // Data for display
        $crop_name = $result_data['Crop'] ?? 'Could not determine crop.';
        $input_conditions = $result_data['Input'] ?? 'N/A';
        
        // Try to decode JSON input for cleaner display
        $input_display = json_decode($input_conditions, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            $input_display = $input_conditions; // Fallback to raw string
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<?php include 'includes/global_head_scripts.php'; ?>
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Crop Recommendation Tool</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet"/>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
    <script>
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    colors: {
                        "primary": "#0df259", "background-light": "#f5f8f6", "background-dark": "#102216",
                        "text-light": "#0d1c12", "text-dark": "#e7f4eb", "surface-light": "#ffffff",
                        "surface-dark": "#1a2e21", "border-light": "#cee8d7", "border-dark": "#2a4a35",
                        "subtle-light": "#499c65", "subtle-dark": "#94c2a5",
                    },
                    fontFamily: { "display": ["Inter", "sans-serif"] },
                    borderRadius: { "DEFAULT": "0.5rem", "lg": "0.75rem", "xl": "1rem", "full": "9999px" },
                },
            },
        }
    </script>
<style>
        .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; font-size: 20px; }
        /* Fixed Sidebar Implementation */
        .fixed-sidebar-menu {
            position: fixed; top: 0; left: 0; width: 280px; height: 100vh;
            background-color: #f5f8f6; color: #0d1c12; z-index: 1030; padding-top: 20px; border-right: 1px solid #cee8d7;
            display: flex; flex-direction: column;
        }
        .main-content-area { margin-left: 280px; }
        .fixed-sidebar-menu .nav-link { padding: 15px 1.5rem; color: #0d1c12 !important; font-weight: 500; }
        .fixed-sidebar-menu .nav-link.active {
            background-color: rgba(13, 242, 89, 0.2); color: #0d1c12 !important;
            border-left: 3px solid #0df259; font-weight: 700;
        }
        .top-navbar { display: none; }
        .form-input-custom {
            @apply form-input w-full rounded-lg border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark p-2 text-sm text-text-light dark:text-text-dark focus:outline-none focus:ring-1 focus:ring-primary transition-colors;
        }
    </style>
</head>
<body class="bg-background-light dark:bg-background-dark font-display text-text-light dark:text-text-dark">

<div class="fixed-sidebar-menu">
    <div class="px-4 py-3 border-b border-border-light dark:border-border-dark">
        <h5 class="text-xl font-bold" style="color: #0d1c12;">SMART AGRO AI</h5>
    </div>
    <nav class="nav flex-column flex-1 py-4">
      <a class="nav-link" href="index.php">Dashboard Home</a>
      <a class="nav-link" href="predict_plant.php">🌱 Plant Disease Detection</a>
      <a class="nav-link" href="predict_wheat.php">🌾 Wheat Disease Detection</a>
      <a class="nav-link active" href="recommend_crop.php">📊 Crop Recommendation</a>
      <a class="nav-link" href="recommend_map.php">🗺️ Map Recommendation</a>
    </nav>
    <div class="p-4 border-top border-border-light dark:border-border-dark">
        <p class="text-sm text-text-light/70 dark:text-text-dark/70 mb-1">
            Logged in as: <strong><?php echo htmlspecialchars($username); ?></strong>
        </p>
        <a href="index.php?logout=true" class="flex w-full items-center justify-center rounded-lg h-10 bg-primary/20 text-text-light dark:bg-primary/30 dark:text-text-dark hover:bg-primary/30 dark:hover:bg-primary/40 transition-colors text-sm font-bold">Logout</a>
    </div>
</div>

<div class="main-content-area flex flex-1 justify-center py-10 sm:py-16">
    <div class="w-full max-w-5xl px-4 sm:px-6 lg:px-8">
        
        <div class="flex flex-col items-center gap-2 mb-10 text-center">
            <p class="text-4xl font-black tracking-tighter sm:text-5xl text-text-light dark:text-text-dark">Crop Recommendation Tool</p>
            <p class="max-w-xl text-base text-subtle-light dark:text-subtle-dark">Input the soil and environmental parameters below for analysis via Port 5002.</p>
        </div>

        <section id="upload-section" style="<?php echo $has_result ? 'display: none;' : ''; ?>" class="flex justify-center">
            <div class="w-full max-w-3xl p-6 rounded-xl border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark">
                <form action="action_predict.php" method="POST">
                    <input type="hidden" name="model_type" value="crop">

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div class="flex flex-col gap-1"><label class="text-sm font-medium" for="N">Nitrogen (N) (kg/ha)</label><input type="number" step="any" class="form-input-custom" name="N" required value="90"></div>
                        <div class="flex flex-col gap-1"><label class="text-sm font-medium" for="P">Phosphorus (P) (kg/ha)</label><input type="number" step="any" class="form-input-custom" name="P" required value="42"></div>
                        <div class="flex flex-col gap-1"><label class="text-sm font-medium" for="K">Potassium (K) (kg/ha)</label><input type="number" step="any" class="form-input-custom" name="K" required value="43"></div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div class="flex flex-col gap-1"><label class="text-sm font-medium" for="temperature">Temperature (°C)</label><input type="number" step="any" class="form-input-custom" name="temperature" required value="20.87"></div>
                        <div class="flex flex-col gap-1"><label class="text-sm font-medium" for="humidity">Humidity (%)</label><input type="number" step="any" class="form-input-custom" name="humidity" required value="82.0"></div>
                        <div class="flex flex-col gap-1"><label class="text-sm font-medium" for="pH">pH Value</label><input type="number" step="any" class="form-input-custom" name="pH" required value="6.5"></div>
                    </div>

                    <div class="flex flex-col gap-1 mb-6">
                        <label class="text-sm font-medium" for="rainfall">Rainfall (mm)</label>
                        <input type="number" step="any" class="form-input-custom" name="rainfall" required value="202.93">
                    </div>
                    
                    <button type="submit" class="flex w-full items-center justify-center rounded-lg h-12 px-5 bg-primary text-background-dark text-base font-black shadow-lg hover:brightness-110 transition-all">
                        <span class="truncate">Get Crop Recommendation</span>
                    </button>
                </form>
            </div>
        </section>
        
        <section id="results-section" style="<?php echo $has_result ? '' : 'display: none;'; ?>">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12 p-4">
                
                <div class="flex flex-col justify-center gap-6 md:col-span-2">
                    <div class="flex flex-col items-center text-center">
                        <p class="text-sm font-bold uppercase tracking-wider text-primary">Recommendation Result</p>
                        <h2 class="text-5xl font-black tracking-tighter text-text-light dark:text-text-dark mt-1"><?php echo htmlspecialchars($crop_name); ?></h2>
                    </div>

                    <div class="p-4 rounded-lg bg-surface-dark/5 dark:bg-surface-dark border border-border-light dark:border-border-dark">
                        <h3 class="text-lg font-bold mb-3">Input Conditions Analyzed</h3>
                        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm text-subtle-light dark:text-subtle-dark">
                            <?php if (is_array($input_display)): ?>
                                <?php foreach ($input_display as $key => $value): ?>
                                    <p><strong class="text-text-light dark:text-text-dark"><?php echo htmlspecialchars($key); ?>:</strong> <?php echo htmlspecialchars($value); ?></p>
                                <?php endforeach; ?>
                            <?php else: ?>
                                <p class="col-span-4">Raw Input: <?php echo htmlspecialchars($input_display); ?></p>
                            <?php endif; ?>
                        </div>
                    </div>
                    
                    <div class="p-4 rounded-lg bg-surface-dark/5 dark:bg-surface-dark border border-border-light dark:border-border-dark">
                        <h3 class="text-lg font-bold mb-3">Cultivation Guide (AI Generated)</h3>
                        <div class="text-sm text-subtle-light dark:text-subtle-dark leading-relaxed whitespace-pre-wrap max-h-72 overflow-y-auto">
                            <?php echo htmlspecialchars($result_data['Advice']); ?>
                        </div>
                    </div>

                    <div class="flex flex-col sm:flex-row items-center gap-3 pt-4">
                        <a href="index.php" class="flex w-full sm:w-auto items-center justify-center rounded-lg h-11 px-5 bg-transparent border border-border-light dark:border-border-dark text-text-light dark:text-text-dark hover:bg-black/5 dark:hover:bg-white/5 text-sm font-bold transition-all">
                            <span class="material-symbols-outlined mr-2 !text-lg">home</span>
                            Go to Dashboard
                        </a>
                        
                        <button onclick="window.location.href='recommend_crop.php';" class="flex w-full sm:w-auto items-center justify-center rounded-lg h-11 px-5 bg-primary text-background-dark text-sm font-bold shadow-sm hover:brightness-110 transition-all">
                            <span class="material-symbols-outlined mr-2 !text-lg">restart_alt</span>
                            Run New Recommendation
                        </button>
                    </div>
                </div>
            </div>
        </section>

    </div>
</div>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
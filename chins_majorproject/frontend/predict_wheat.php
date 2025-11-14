<?php
session_start();
require_once 'config.php';
if (!isset($_SESSION['user_id'])) { header("Location: auth.php"); exit(); }
$username = $_SESSION['username'];

// Check for successful prediction result passed via URL
$has_result = false;
$result_data = [];
$uploaded_image_path = isset($_GET['img']) ? htmlspecialchars($_GET['img']) : 'results/default_wheat_placeholder.jpg'; 


if (isset($_GET['status']) && $_GET['status'] == 'success' && isset($_GET['file'])) {
    $file_path = htmlspecialchars($_GET['file']);
    if (file_exists($file_path)) {
        $has_result = true;
        
        // 1. Read and Parse the saved text file content
        $raw_content = file_get_contents($file_path);
        
        $lines = explode("\n", $raw_content);
        $result_data['Advice'] = '';
        $in_advice_section = false;

        foreach ($lines as $line) {
            if (strpos($line, 'Detected Disease:') !== false) {
                $result_data['Disease'] = trim(str_replace('Detected Disease:', '', $line));
            } elseif (strpos($line, 'Confidence:') !== false) {
                $result_data['Confidence'] = trim(str_replace('Confidence:', '', $line));
            } elseif (strpos($line, '--- AI Generated Advice ---') !== false) {
                $in_advice_section = true;
            } elseif ($in_advice_section) {
                $result_data['Advice'] .= $line . "\n";
            }
        }
        
        // Data for display
        $disease_name = $result_data['Disease'] ?? 'Diagnosis Complete';
        $confidence = $result_data['Confidence'] ?? 'N/A';
        $severity = (float)str_replace(['%', ' '], '', $confidence) > 85 ? 'High' : 'Moderate';
        $severity_color = $severity === 'High' ? 'bg-red-500' : 'bg-yellow-500';
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<?php include 'includes/global_head_scripts.php'; ?>
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Wheat Health - Diagnosis Tool</title>
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
    </style>
</head>
<body class="bg-background-light dark:bg-background-dark font-display text-text-light dark:text-text-dark">

<div class="fixed-sidebar-menu">
    <div class="px-4 py-3 border-b border-border-light dark:border-border-dark">
        <h5 class="text-xl font-bold" style="color: #0d1c12;">AGRIVISION AI</h5>
    </div>
    <nav class="nav flex-column flex-1 py-4">
      <a class="nav-link" href="index.php">Dashboard Home</a>
      <a class="nav-link" href="predict_plant.php">🌱 Plant Disease Detection</a>
      <a class="nav-link active" href="predict_wheat.php">🌾 Stem Disease Detection</a>
      <a class="nav-link" href="recommend_crop.php">📊 Crop Recommendation</a>
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
            <p class="text-4xl font-black tracking-tighter sm:text-5xl text-text-light dark:text-text-dark">Diagnose Wheat Health</p>
            <p class="max-w-xl text-base text-subtle-light dark:text-subtle-dark">Upload an image of the wheat stem or root for specific infection diagnosis via Port 5001.</p>
        </div>

        <section id="upload-section" style="<?php echo $has_result ? 'display: none;' : ''; ?>">
            <form action="action_predict.php" method="POST" enctype="multipart/form-data" id="wheat-diagnosis-form">
                <input type="hidden" name="model_type" value="wheat">
                
                <div class="flex flex-col p-4">
                    <div class="flex flex-col items-center gap-6 rounded-xl border-2 border-dashed border-border-light dark:border-border-dark px-6 py-14 bg-surface-light dark:bg-surface-dark">
                        
                        <div class="flex flex-col items-center justify-center rounded-full size-16 bg-primary/20 text-primary">
                            <span class="material-symbols-outlined !text-3xl">agriculture</span>
                        </div>
                        
                        <div class="flex max-w-md flex-col items-center gap-2">
                            <p class="text-lg font-bold tracking-tight text-text-light dark:text-text-dark text-center" id="upload-status-text">Drag & drop an image or click to upload</p>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark text-center">Model: `wheat_stem_model_final.pth`</p>
                        </div>
                        
                        <label for="image-upload-input" class="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-11 px-5 bg-primary text-background-dark text-sm font-bold shadow-sm hover:brightness-110 transition-all">
                            <span class="truncate">Select Image</span>
                        </label>
                        
                        <input type="file" id="image-upload-input" name="image" accept="image/jpeg, image/png" required style="display: none;">
                    </div>
                </div>

                <div class="flex justify-center p-4">
                    <button type="submit" class="flex w-full max-w-md items-center justify-center rounded-lg h-12 px-5 bg-primary text-background-dark text-base font-black shadow-lg hover:brightness-110 transition-all">
                        <span class="truncate">Diagnose Now</span>
                    </button>
                </div>
            </form>
        </section>
        
        <section id="results-section" style="<?php echo $has_result ? '' : 'display: none;'; ?>">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-12 p-4">
                <div class="w-full h-auto aspect-[4/3] rounded-xl overflow-hidden shadow-lg">
                    <div class="w-full h-full bg-center bg-no-repeat bg-cover bg-cover-placeholder" 
                         style='background-image: url("<?php echo $uploaded_image_path; ?>");'>
                    </div>
                </div>
                
                <div class="flex flex-col justify-center gap-6">
                    <div>
                        <p class="text-sm font-bold uppercase tracking-wider text-primary">Diagnosis Result</p>
                        <h2 class="text-4xl font-black tracking-tighter text-text-light dark:text-text-dark mt-1"><?php echo htmlspecialchars($disease_name); ?></h2>
                    </div>
                    
                    <div class="flex items-center gap-4">
                        <div class="flex items-center gap-2">
                            <p class="font-semibold text-text-light dark:text-text-dark">Confidence:</p>
                            <span class="text-lg font-bold text-primary"><?php echo htmlspecialchars($confidence); ?></span>
                        </div>
                        <div class="w-px h-5 bg-border-light dark:bg-border-dark"></div>
                        <div class="flex items-center gap-2">
                            <p class="font-semibold text-text-light dark:text-text-dark">Severity:</p>
                            <span class="inline-flex items-center rounded-full <?php echo $severity_color; ?> px-3 py-1 text-sm font-medium text-background-dark">
                                <?php echo htmlspecialchars($severity); ?>
                            </span>
                        </div>
                    </div>
                    
                    <div class="p-4 rounded-lg bg-surface-dark/5 dark:bg-surface-dark border border-border-light dark:border-border-dark">
                        <h3 class="text-lg font-bold mb-3">Care Instructions (AI Generated)</h3>
                        <div class="text-sm text-subtle-light dark:text-subtle-dark leading-relaxed whitespace-pre-wrap max-h-56 overflow-y-auto">
                            <?php echo htmlspecialchars($result_data['Advice']); ?>
                        </div>
                    </div>

                    <div class="flex flex-col sm:flex-row items-center gap-3 pt-4">
                        <a href="index.php" class="flex w-full sm:w-auto items-center justify-center rounded-lg h-11 px-5 bg-transparent border border-border-light dark:border-border-dark text-text-light dark:text-text-dark hover:bg-black/5 dark:hover:bg-white/5 text-sm font-bold transition-all">
                            <span class="material-symbols-outlined mr-2 !text-lg">home</span>
                            Go to Dashboard
                        </a>
                        
                        <button onclick="window.location.href='predict_wheat.php';" class="flex w-full sm:w-auto items-center justify-center rounded-lg h-11 px-5 bg-primary text-background-dark text-sm font-bold shadow-sm hover:brightness-110 transition-all">
                            <span class="material-symbols-outlined mr-2 !text-lg">restart_alt</span>
                            Run New Diagnosis
                        </button>
                    </div>
                </div>
            </div>
        </section>

    </div>
</div>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
    // Logic to update the status text when a file is selected
    document.getElementById('image-upload-input').addEventListener('change', function(e) {
        const fileName = e.target.files[0] ? e.target.files[0].name : "Select or drop an image file";
        const uploadStatusText = document.getElementById('upload-status-text');
        if(uploadStatusText) {
            uploadStatusText.textContent = `File Ready: ${fileName}`;
        }
    });
</script>
</body>
</html>
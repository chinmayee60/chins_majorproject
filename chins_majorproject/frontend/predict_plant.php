<?php
session_start();
require_once 'config.php';
if (!isset($_SESSION['user_id'])) { header("Location: auth.php"); exit(); }
$username = $_SESSION['username'];

// Determine active mode from URL parameter (default: plant)
$active_mode = isset($_GET['mode']) && $_GET['mode'] === 'wheat' ? 'wheat' : 'plant';

// Check for successful prediction result passed via URL
$has_result = false;
$result_data = [];
$uploaded_image_path = isset($_GET['img']) ? htmlspecialchars($_GET['img']) : 'results/default_plant_placeholder.jpg'; 


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
        // Mock Severity logic for display purposes
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
<title>Plant Health - Diagnosis Tool</title>
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
        .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; }
        .text-primary { color: #0df259 !important; }
        
        /* Header Navigation */
        .header-navigation {
            position: sticky;
            top: 0;
            z-index: 1000;
            background-color: #f5f8f6;
            border-bottom: 1px solid #cee8d7;
        }
        .dark .header-navigation {
            background-color: #102216;
            border-bottom-color: #2a4f37;
        }

        /* Mobile Menu */
        #mobileMenu {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #f5f8f6;
            border-bottom: 1px solid #cee8d7;
            padding: 1rem;
            flex-direction: column;
            gap: 0.5rem;
        }
        #mobileMenu.open {
            display: flex;
        }
        .dark #mobileMenu {
            background-color: #102216;
            border-bottom-color: #2a4f37;
        }
        
        /* Toggle Button Styles - Professional Design */
        .toggle-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem 2rem;
            border-radius: 0.75rem;
            font-size: 0.9375rem;
            font-weight: 600;
            color: #6b7280;
            background-color: transparent;
            border: none;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        .dark .toggle-btn {
            color: #9ca3af;
        }
        .toggle-btn:hover:not(.active) {
            color: #0df259;
            background-color: rgba(13, 242, 89, 0.08);
        }
        .toggle-btn.active {
            background: linear-gradient(135deg, #0df259 0%, #0bc047 100%);
            color: #ffffff;
            font-weight: 700;
            box-shadow: 0 4px 12px rgba(13, 242, 89, 0.25);
        }
        .dark .toggle-btn.active {
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(13, 242, 89, 0.35);
        }
        
        /* Professional Upload Area */
        .upload-area {
            background: linear-gradient(to bottom, #ffffff, #f9fafb);
            border: 2px dashed #d1d5db;
            transition: all 0.3s ease;
        }
        .dark .upload-area {
            background: linear-gradient(to bottom, #1a2e21, #142218);
            border-color: #374151;
        }
        .upload-area:hover {
            border-color: #0df259;
            background: linear-gradient(to bottom, #f0fdf4, #f9fafb);
        }
        .dark .upload-area:hover {
            border-color: #0df259;
            background: linear-gradient(to bottom, #1f3a29, #1a2e21);
        }
        
        /* Icon Container Gradient */
        .icon-gradient {
            background: linear-gradient(135deg, #0df259 0%, #0bc047 100%);
        }
        
        /* Feature Badge */
        .feature-badge {
            background: linear-gradient(135deg, rgba(13, 242, 89, 0.1) 0%, rgba(11, 192, 71, 0.1) 100%);
            border: 1px solid rgba(13, 242, 89, 0.2);
            color: #0bc047;
            font-weight: 600;
        }
        .dark .feature-badge {
            background: linear-gradient(135deg, rgba(13, 242, 89, 0.15) 0%, rgba(11, 192, 71, 0.15) 100%);
            color: #0df259;
        }
        
        /* Results Section Enhancements */
        #results-section {
            animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Bento Card Hover Effects */
        .bento-card {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .bento-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        
        /* Prose Styling for AI Content */
        .prose p {
            margin-bottom: 0.75rem;
        }
        
        .prose p:last-child {
            margin-bottom: 0;
        }
    </style>
</head>
<!-- Updated: 2025-11-15 Header-only Navigation -->
<body class="bg-background-light dark:bg-background-dark font-display text-text-light dark:text-text-dark">

<!-- Header Navigation -->
<header class="header-navigation">
    <div class="flex items-center justify-between px-4 py-3 relative">
        <div class="flex items-center gap-3">
            <div class="size-6 text-primary">
                <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><g clip-path="url(#clip0_6_535)"><path clip-rule="evenodd" d="M47.2426 24L24 47.2426L0.757355 24L24 0.757355L47.2426 24ZM12.2426 21H35.7574L24 9.24264L12.2426 21Z" fill="currentColor" fill-rule="evenodd"></path></g></svg>
            </div>
            <h2 class="text-lg font-bold">Plant AI</h2>
        </div>
        
        <nav class="hidden md:flex items-center gap-6 text-sm font-medium">
            <a href="auth.php?landing=1" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Home</a>
            <a href="index.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Dashboard</a>
            <a href="predict_plant.php" class="text-text-light dark:text-text-dark font-bold hover:text-primary transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 rounded-lg bg-primary/20 text-text-light hover:bg-primary/30 transition-colors font-bold">Logout</a>
        </nav>
        
        <button id="mobileMenuToggle" class="md:hidden">
            <span class="material-symbols-outlined" style="font-size: 24px;">menu</span>
        </button>
        
        <div id="mobileMenu" class="md:hidden">
            <a href="auth.php?landing=1" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Home</a>
            <a href="index.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Dashboard</a>
            <a href="predict_plant.php" class="px-4 py-2 text-sm font-bold hover:bg-primary/10 rounded transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 bg-primary/20 rounded text-sm font-bold hover:bg-primary/30 transition-colors">Logout</a>
        </div>
    </div>
</header>

<div class="flex flex-1 justify-center py-8">
    <div class="w-full max-w-6xl px-4 sm:px-6 lg:px-8">
        
        <!-- Professional Header Section -->
        <div class="text-center mb-8">
            <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full feature-badge text-sm font-semibold mb-4">
                <span class="material-symbols-outlined" style="font-size: 18px;">neurology</span>
                AI-Powered Disease Detection
            </div>
            <h1 class="text-5xl md:text-6xl font-black tracking-tight text-text-light dark:text-text-dark mb-4">
                Plant Health Diagnosis
            </h1>
            <p class="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
                Advanced artificial intelligence to identify plant diseases instantly and accurately
            </p>
        </div>
        
        <!-- Toggle Switch - Professional Design -->
        <div class="flex justify-center mb-12">
            <div class="inline-flex rounded-xl bg-gray-100 dark:bg-gray-800 p-1.5 shadow-sm">
                <button id="plantModeBtn" class="toggle-btn <?php echo $active_mode === 'plant' ? 'active' : ''; ?>" onclick="switchMode('plant')">
                    <span class="material-symbols-outlined mr-2.5" style="font-size: 22px;">eco</span>
                    Leaf Disease Detection
                </button>
                <button id="wheatModeBtn" class="toggle-btn <?php echo $active_mode === 'wheat' ? 'active' : ''; ?>" onclick="switchMode('wheat')">
                    <span class="material-symbols-outlined mr-2.5" style="font-size: 22px;">agriculture</span>
                    Wheat Disease Detection
                </button>
            </div>
        </div>
        
        <!-- Plant Mode Content -->
        <div id="plantContent" class="mode-content" style="display: <?php echo $active_mode === 'plant' ? 'block' : 'none'; ?>;">
            <div class="text-center mb-8">
                <h2 class="text-2xl font-bold text-text-light dark:text-text-dark mb-2">Leaf Disease Analysis</h2>
                <p class="text-base text-gray-600 dark:text-gray-400">Upload a clear image of the affected plant leaf for instant diagnosis</p>
            </div>
        </div>
        
        <!-- Wheat Mode Content -->
        <div id="wheatContent" class="mode-content" style="display: <?php echo $active_mode === 'wheat' ? 'block' : 'none'; ?>;">
            <div class="text-center mb-8">
                <h2 class="text-2xl font-bold text-text-light dark:text-text-dark mb-2">Wheat Disease Analysis</h2>
                <p class="text-base text-gray-600 dark:text-gray-400">Upload an image of wheat stem or root to detect infections and diseases</p>
            </div>
        </div>

        <!-- Plant Mode Upload Form -->
        <section id="plant-upload-section" class="mode-content" style="display: <?php echo ($active_mode === 'plant' && !$has_result) ? 'block' : 'none'; ?>;">
            <form action="action_predict.php" method="POST" enctype="multipart/form-data" id="plant-diagnosis-form" class="max-w-3xl mx-auto">
                <input type="hidden" name="model_type" value="plant">
                
                <div class="mb-8">
                    <div class="upload-area flex flex-col items-center gap-8 rounded-2xl px-8 py-16">
                        
                        <div class="flex flex-col items-center justify-center rounded-2xl size-20 icon-gradient shadow-lg">
                            <span class="material-symbols-outlined !text-4xl text-white">cloud_upload</span>
                        </div>
                        
                        <div class="flex max-w-lg flex-col items-center gap-3">
                            <p class="text-xl font-bold text-text-light dark:text-text-dark text-center" id="plant-upload-status-text">
                                Drag and drop your image here
                            </p>
                            <p class="text-sm text-gray-500 dark:text-gray-400 text-center">
                                or click below to browse files
                            </p>
                            <div class="flex items-center gap-4 mt-2">
                                <div class="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
                                    <span class="material-symbols-outlined" style="font-size: 16px;">check_circle</span>
                                    JPG, PNG
                                </div>
                                <div class="w-px h-4 bg-gray-300 dark:bg-gray-600"></div>
                                <div class="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
                                    <span class="material-symbols-outlined" style="font-size: 16px;">data_usage</span>
                                    Max 5MB
                                </div>
                            </div>
                        </div>
                        
                        <label for="plant-image-upload-input" class="flex items-center justify-center gap-2 rounded-xl h-12 px-8 bg-gradient-to-r from-primary to-[#0bc047] text-white text-base font-bold shadow-lg hover:shadow-xl hover:scale-105 transition-all cursor-pointer">
                            <span class="material-symbols-outlined" style="font-size: 20px;">add_photo_alternate</span>
                            <span class="truncate">Choose Image</span>
                        </label>
                        
                        <input type="file" id="plant-image-upload-input" name="image" accept="image/jpeg, image/png" required style="display: none;">
                    </div>
                </div>

                <div class="flex justify-center">
                    <button type="submit" class="flex items-center justify-center gap-3 rounded-xl h-14 px-12 bg-gradient-to-r from-primary to-[#0bc047] text-white text-lg font-black shadow-xl hover:shadow-2xl hover:scale-105 transition-all">
                        <span class="material-symbols-outlined" style="font-size: 24px;">biotech</span>
                        <span class="truncate">Analyze Now</span>
                    </button>
                </div>
            </form>
        </section>
        
        <!-- Wheat Mode Upload Form -->
        <section id="wheat-upload-section" class="mode-content" style="display: <?php echo ($active_mode === 'wheat' && !$has_result) ? 'block' : 'none'; ?>;">
            <form action="action_predict.php" method="POST" enctype="multipart/form-data" id="wheat-diagnosis-form" class="max-w-3xl mx-auto">
                <input type="hidden" name="model_type" value="wheat">
                
                <div class="mb-8">
                    <div class="upload-area flex flex-col items-center gap-8 rounded-2xl px-8 py-16">
                        
                        <div class="flex flex-col items-center justify-center rounded-2xl size-20 icon-gradient shadow-lg">
                            <span class="material-symbols-outlined !text-4xl text-white">cloud_upload</span>
                        </div>
                        
                        <div class="flex max-w-lg flex-col items-center gap-3">
                            <p class="text-xl font-bold text-text-light dark:text-text-dark text-center" id="wheat-upload-status-text">
                                Drag and drop your image here
                            </p>
                            <p class="text-sm text-gray-500 dark:text-gray-400 text-center">
                                or click below to browse files
                            </p>
                            <div class="flex items-center gap-4 mt-2">
                                <div class="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
                                    <span class="material-symbols-outlined" style="font-size: 16px;">check_circle</span>
                                    JPG, PNG
                                </div>
                                <div class="w-px h-4 bg-gray-300 dark:bg-gray-600"></div>
                                <div class="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
                                    <span class="material-symbols-outlined" style="font-size: 16px;">data_usage</span>
                                    Max 5MB
                                </div>
                            </div>
                        </div>
                        
                        <label for="wheat-image-upload-input" class="flex items-center justify-center gap-2 rounded-xl h-12 px-8 bg-gradient-to-r from-primary to-[#0bc047] text-white text-base font-bold shadow-lg hover:shadow-xl hover:scale-105 transition-all cursor-pointer">
                            <span class="material-symbols-outlined" style="font-size: 20px;">add_photo_alternate</span>
                            <span class="truncate">Choose Image</span>
                        </label>
                        
                        <input type="file" id="wheat-image-upload-input" name="image" accept="image/jpeg, image/png" required style="display: none;">
                    </div>
                </div>

                <div class="flex justify-center">
                    <button type="submit" class="flex items-center justify-center gap-3 rounded-xl h-14 px-12 bg-gradient-to-r from-primary to-[#0bc047] text-white text-lg font-black shadow-xl hover:shadow-2xl hover:scale-105 transition-all">
                        <span class="material-symbols-outlined" style="font-size: 24px;">biotech</span>
                        <span class="truncate">Analyze Now</span>
                    </button>
                </div>
            </form>
        </section>

        <section id="upload-section" style="display:none;">
            <form action="action_predict.php" method="POST" enctype="multipart/form-data" id="old-plant-diagnosis-form">
                <input type="hidden" name="model_type" value="plant">
                
                <div class="flex flex-col p-4">
                    <div class="flex flex-col items-center gap-6 rounded-xl border-2 border-dashed border-border-light dark:border-border-dark px-6 py-14 bg-surface-light dark:bg-surface-dark">
                        
                        <div class="flex flex-col items-center justify-center rounded-full size-16 bg-primary/20 text-primary">
                            <span class="material-symbols-outlined !text-3xl">upload_file</span>
                        </div>
                        
                        <div class="flex max-w-md flex-col items-center gap-2">
                            <p class="text-lg font-bold tracking-tight text-text-light dark:text-text-dark text-center" id="upload-status-text">Drag & drop an image or click to upload</p>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark text-center">Model: `plant_disease_cnn.pth`</p>
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
        
        <!-- Results Section - Professional Bento Grid Layout -->
        <section id="results-section" class="max-w-6xl mx-auto" style="<?php echo $has_result ? '' : 'display: none;'; ?>">
            
            <!-- Success Header -->
            <div class="text-center mb-10">
                <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-sm font-semibold mb-4">
                    <span class="material-symbols-outlined" style="font-size: 18px;">check_circle</span>
                    Analysis Complete
                </div>
                <h2 class="text-3xl md:text-4xl font-black text-text-light dark:text-text-dark">
                    Diagnosis Results
                </h2>
            </div>

            <!-- Bento Grid Layout -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-8">
                
                <!-- Image Card - Larger Bento Box -->
                <div class="lg:col-span-5 rounded-2xl overflow-hidden shadow-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                    <div class="relative aspect-[4/3] bg-cover bg-center" style='background-image: url("<?php echo $uploaded_image_path; ?>");'>
                        <div class="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                        <div class="absolute bottom-4 left-4 right-4">
                            <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm">
                                <span class="material-symbols-outlined text-primary" style="font-size: 16px;">photo_camera</span>
                                <span class="text-xs font-semibold text-gray-700 dark:text-gray-300">Uploaded Image</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Disease Info Card - Bento Box -->
                <div class="lg:col-span-7 space-y-6">
                    
                    <!-- Disease Name Card -->
                    <div class="rounded-2xl bg-gradient-to-br from-green-50 to-emerald-50 dark:from-gray-800 dark:to-gray-700 p-6 border border-green-200 dark:border-green-900/30 shadow-lg">
                        <div class="flex items-start justify-between mb-4">
                            <div>
                                <p class="text-sm font-bold uppercase tracking-wider text-green-700 dark:text-green-400 mb-2">Detected Disease</p>
                                <h3 class="text-3xl font-black text-gray-900 dark:text-white leading-tight break-words">
                                    <?php echo htmlspecialchars(str_replace('___', ' - ', $disease_name)); ?>
                                </h3>
                            </div>
                            <div class="flex-shrink-0 w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center">
                                <span class="material-symbols-outlined text-primary" style="font-size: 28px;">coronavirus</span>
                            </div>
                        </div>
                    </div>

                    <!-- Metrics Grid - Bold Pop-Out Design -->
                    <div class="grid grid-cols-2 gap-5">
                        <!-- Confidence Card - Bold Design -->
                        <div class="group relative rounded-2xl bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 p-6 border-2 border-blue-200 dark:border-blue-800 shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
                            <div class="absolute top-4 right-4 w-12 h-12 rounded-xl bg-blue-500/10 dark:bg-blue-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined text-blue-600 dark:text-blue-400" style="font-size: 28px; font-weight: 600;">analytics</span>
                            </div>
                            <div class="mb-3">
                                <p class="text-sm font-bold uppercase tracking-wider text-blue-700 dark:text-blue-400 mb-1">Confidence Score</p>
                            </div>
                            <p class="text-5xl font-black text-blue-600 dark:text-blue-400 mb-4 tracking-tight"><?php echo htmlspecialchars($confidence); ?></p>
                            <div class="w-full bg-blue-200 dark:bg-blue-900/40 rounded-full h-3 overflow-hidden shadow-inner">
                                <div class="bg-gradient-to-r from-blue-500 to-cyan-400 h-3 rounded-full transition-all duration-700 ease-out shadow-md" style="width: <?php echo str_replace(['%', ' '], '', $confidence); ?>%;"></div>
                            </div>
                            <p class="mt-2 text-xs font-semibold text-blue-600/70 dark:text-blue-400/70">Model Accuracy</p>
                        </div>

                        <!-- Severity Card - Bold Design -->
                        <div class="group relative rounded-2xl bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 p-6 border-2 border-orange-200 dark:border-orange-800 shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
                            <div class="absolute top-4 right-4 w-12 h-12 rounded-xl bg-orange-500/10 dark:bg-orange-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined text-orange-600 dark:text-orange-400" style="font-size: 28px; font-weight: 600;">warning</span>
                            </div>
                            <div class="mb-3">
                                <p class="text-sm font-bold uppercase tracking-wider text-orange-700 dark:text-orange-400 mb-1">Severity Level</p>
                            </div>
                            <div class="mt-3">
                                <span class="inline-flex items-center gap-2 rounded-xl <?php echo $severity_color; ?> px-6 py-3 text-xl font-black text-white shadow-xl border-2 <?php echo $severity === 'High' ? 'border-red-700' : 'border-yellow-600'; ?>">
                                    <span class="material-symbols-outlined" style="font-size: 24px; font-weight: 700;">
                                        <?php echo $severity === 'High' ? 'priority_high' : 'info'; ?>
                                    </span>
                                    <?php echo htmlspecialchars($severity); ?>
                                </span>
                            </div>
                            <p class="mt-4 text-xs font-semibold text-orange-600/70 dark:text-orange-400/70">Risk Assessment</p>
                        </div>
                    </div>

                </div>
            </div>

            <!-- AI Care Instructions - Full Width Bento Card -->
            <div class="rounded-2xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-xl overflow-hidden mb-8">
                <div class="bg-gradient-to-r from-primary/10 to-green-500/10 dark:from-primary/20 dark:to-green-500/20 px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-green-500 flex items-center justify-center shadow-md">
                            <span class="material-symbols-outlined text-white" style="font-size: 24px;">psychology</span>
                        </div>
                        <div>
                            <h3 class="text-xl font-black text-gray-900 dark:text-white">Treatment Recommendations</h3>
                        </div>
                    </div>
                </div>
                
                <div class="p-6">
                    <div class="text-gray-700 dark:text-gray-300 leading-relaxed space-y-4 text-base">
                        <?php 
                        // Function to format text with bold and other markdown-like syntax
                        function formatAdviceText($text) {
                            // Replace **text** with bold formatting
                            $text = preg_replace('/\*\*(.*?)\*\*/', '<strong class="font-bold text-gray-900 dark:text-white">$1</strong>', $text);
                            
                            // Replace *text* with italic formatting
                            $text = preg_replace('/\*(.*?)\*/', '<em class="italic">$1</em>', $text);
                            
                            return $text;
                        }
                        
                        // Split advice into parts and format nicely
                        $advice_parts = explode('Part ', $result_data['Advice']);
                        $formatted_advice = '';
                        
                        foreach ($advice_parts as $index => $part) {
                            if (trim($part)) {
                                if ($index > 0) {
                                    // Extract the content after the number and colon
                                    $content = trim(substr($part, strpos($part, ':') + 1));
                                    $formatted_content = formatAdviceText($content);
                                    
                                    $formatted_advice .= '<div class="flex gap-3 p-4 rounded-xl bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 mb-3">';
                                    $formatted_advice .= '<div class="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-primary to-green-500 flex items-center justify-center text-white font-bold shadow-sm">' . $index . '</div>';
                                    $formatted_advice .= '<div class="flex-1"><p class="m-0">' . $formatted_content . '</p></div>';
                                    $formatted_advice .= '</div>';
                                } else {
                                    // First part (before any "Part X:")
                                    if (!empty(trim($part))) {
                                        $formatted_advice .= '<p class="text-gray-600 dark:text-gray-400 mb-4">' . formatAdviceText($part) . '</p>';
                                    }
                                }
                            }
                        }
                        
                        // If no parts were found, display the entire advice as-is with formatting
                        if (empty($formatted_advice)) {
                            echo '<div class="whitespace-pre-line">' . formatAdviceText($result_data['Advice']) . '</div>';
                        } else {
                            echo $formatted_advice;
                        }
                        ?>
                    </div>
                </div>
            </div>

            <!-- Action Button - Centered -->
            <div class="flex justify-center">
                <button onclick="window.location.href='predict_plant.php';" class="flex items-center justify-center gap-2 rounded-xl h-14 px-10 bg-gradient-to-r from-primary to-green-500 text-white text-base font-bold shadow-lg hover:shadow-xl hover:scale-105 transition-all">
                    <span class="material-symbols-outlined" style="font-size: 22px;">restart_alt</span>
                    <span>New Analysis</span>
                </button>
            </div>

        </section>

    </div>
</div>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
    // Mode switching function
    function switchMode(mode) {
        // Update URL without reload and clear result parameters
        const url = new URL(window.location);
        url.searchParams.set('mode', mode);
        // Remove result parameters when switching modes
        url.searchParams.delete('status');
        url.searchParams.delete('file');
        url.searchParams.delete('img');
        window.history.pushState({}, '', url);
        
        // Hide results section and show upload sections when toggling
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
        
        // Toggle content visibility
        if (mode === 'plant') {
            document.getElementById('plantContent').style.display = 'block';
            document.getElementById('wheatContent').style.display = 'none';
            document.getElementById('plant-upload-section').style.display = 'block';
            document.getElementById('wheat-upload-section').style.display = 'none';
            
            // Update toggle buttons
            document.getElementById('plantModeBtn').classList.add('active');
            document.getElementById('wheatModeBtn').classList.remove('active');
        } else {
            document.getElementById('plantContent').style.display = 'none';
            document.getElementById('wheatContent').style.display = 'block';
            document.getElementById('plant-upload-section').style.display = 'none';
            document.getElementById('wheat-upload-section').style.display = 'block';
            
            // Update toggle buttons
            document.getElementById('plantModeBtn').classList.remove('active');
            document.getElementById('wheatModeBtn').classList.add('active');
        }
    }

    // File upload handlers for plant mode with professional feedback
    const plantInput = document.getElementById('plant-image-upload-input');
    if (plantInput) {
        plantInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            const uploadStatusText = document.getElementById('plant-upload-status-text');
            if(file && uploadStatusText) {
                const fileSize = (file.size / 1024 / 1024).toFixed(2); // Convert to MB
                uploadStatusText.innerHTML = `<span class="text-primary">✓</span> ${file.name} <span class="text-gray-500">(${fileSize} MB)</span>`;
            }
        });
    }

    // File upload handlers for wheat mode with professional feedback
    const wheatInput = document.getElementById('wheat-image-upload-input');
    if (wheatInput) {
        wheatInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            const uploadStatusText = document.getElementById('wheat-upload-status-text');
            if(file && uploadStatusText) {
                const fileSize = (file.size / 1024 / 1024).toFixed(2); // Convert to MB
                uploadStatusText.innerHTML = `<span class="text-primary">✓</span> ${file.name} <span class="text-gray-500">(${fileSize} MB)</span>`;
            }
        });
    }

    // Legacy file input handler (for old upload section)
    const oldInput = document.getElementById('image-upload-input');
    if (oldInput) {
        oldInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0] ? e.target.files[0].name : "Select or drop an image file";
            const uploadStatusText = document.getElementById('upload-status-text');
            if(uploadStatusText) {
                uploadStatusText.textContent = `File Ready: ${fileName}`;
            }
        });
    }

    // Mobile menu toggle
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const mobileMenu = document.getElementById('mobileMenu');

    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', function(e) {
            e.stopPropagation();
            mobileMenu.classList.toggle('open');
        });
    }

    // Close mobile menu when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.header-navigation')) {
            if (mobileMenu) mobileMenu.classList.remove('open');
        }
    });
</script>

</body>
</html>
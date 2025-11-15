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
        
        .form-input-custom {
            width: 100%; border-radius: 0.5rem; border: 1px solid #cee8d7;
            background-color: #ffffff; padding: 0.5rem; font-size: 0.875rem;
            color: #0d1c12; transition: all 0.2s;
        }
        .form-input-custom:focus {
            outline: none; border-color: #0df259; box-shadow: 0 0 0 1px #0df259;
        }
        
        /* Custom Scrollbar */
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #0df259;
            border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: #0bc047;
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
            <a href="predict_plant.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="text-text-light dark:text-text-dark font-bold hover:text-primary transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 rounded-lg bg-primary/20 text-text-light hover:bg-primary/30 transition-colors font-bold">Logout</a>
        </nav>
        
        <button id="mobileMenuToggle" class="md:hidden">
            <span class="material-symbols-outlined" style="font-size: 24px;">menu</span>
        </button>
        
        <div id="mobileMenu" class="md:hidden">
            <a href="auth.php?landing=1" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Home</a>
            <a href="index.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Dashboard</a>
            <a href="predict_plant.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="px-4 py-2 text-sm font-bold hover:bg-primary/10 rounded transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 bg-primary/20 rounded text-sm font-bold hover:bg-primary/30 transition-colors">Logout</a>
        </div>
    </div>
</header>

<div class="flex flex-1 justify-center py-8">
    <div class="w-full max-w-6xl px-4 sm:px-6 lg:px-8">
        
        <!-- Professional Header -->
        <div class="flex flex-col items-center gap-3 mb-8 text-center">
            <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 text-green-700 dark:text-green-400 text-sm font-semibold mb-2">
                <span class="material-symbols-outlined" style="font-size: 18px;">agriculture</span>
                AI-Powered Crop Analysis
            </div>
            <h1 class="text-4xl md:text-5xl font-black tracking-tight text-text-light dark:text-text-dark">
                Crop Recommendation Tool
            </h1>
            <p class="max-w-2xl text-base md:text-lg text-subtle-light dark:text-subtle-dark">
                Enter soil and environmental parameters to get personalized crop recommendations based on advanced machine learning analysis
            </p>
        </div>

        <!-- Upload/Input Section -->
        <section id="upload-section" style="<?php echo $has_result ? 'display: none;' : ''; ?>">
            <form action="action_predict.php" method="POST" class="max-w-4xl mx-auto">
                <input type="hidden" name="model_type" value="crop">

                <!-- Soil Nutrients Card -->
                <div class="mb-6 p-6 md:p-8 rounded-2xl bg-gradient-to-br from-white to-green-50/30 dark:from-surface-dark dark:to-green-900/10 border border-border-light dark:border-border-dark shadow-lg">
                    <div class="flex items-center gap-3 mb-6">
                        <div class="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-green-400 to-emerald-500 text-white shadow-lg">
                            <span class="material-symbols-outlined" style="font-size: 28px;">science</span>
                        </div>
                        <div>
                            <h3 class="text-xl font-bold text-text-light dark:text-text-dark">Soil Nutrients</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Primary macronutrients (kg/ha)</p>
                        </div>
                    </div>
                    
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="relative">
                            <label class="block text-sm font-semibold text-text-light dark:text-text-dark mb-2" for="N">
                                Nitrogen (N)
                            </label>
                            <div class="relative">
                                <input type="number" step="any" 
                                    class="w-full h-12 pl-4 pr-16 rounded-xl border-2 border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-light dark:text-text-dark font-medium focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all" 
                                    name="N" required value="90" placeholder="0">
                                <span class="absolute right-4 top-1/2 -translate-y-1/2 text-sm font-medium text-subtle-light dark:text-subtle-dark">kg/ha</span>
                            </div>
                        </div>
                        
                        <div class="relative">
                            <label class="block text-sm font-semibold text-text-light dark:text-text-dark mb-2" for="P">
                                Phosphorus (P)
                            </label>
                            <div class="relative">
                                <input type="number" step="any" 
                                    class="w-full h-12 pl-4 pr-16 rounded-xl border-2 border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-light dark:text-text-dark font-medium focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all" 
                                    name="P" required value="42" placeholder="0">
                                <span class="absolute right-4 top-1/2 -translate-y-1/2 text-sm font-medium text-subtle-light dark:text-subtle-dark">kg/ha</span>
                            </div>
                        </div>
                        
                        <div class="relative">
                            <label class="block text-sm font-semibold text-text-light dark:text-text-dark mb-2" for="K">
                                Potassium (K)
                            </label>
                            <div class="relative">
                                <input type="number" step="any" 
                                    class="w-full h-12 pl-4 pr-16 rounded-xl border-2 border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-light dark:text-text-dark font-medium focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all" 
                                    name="K" required value="43" placeholder="0">
                                <span class="absolute right-4 top-1/2 -translate-y-1/2 text-sm font-medium text-subtle-light dark:text-subtle-dark">kg/ha</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Environmental Conditions Card -->
                <div class="mb-6 p-6 md:p-8 rounded-2xl bg-gradient-to-br from-white to-blue-50/30 dark:from-surface-dark dark:to-blue-900/10 border border-border-light dark:border-border-dark shadow-lg">
                    <div class="flex items-center gap-3 mb-6">
                        <div class="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-blue-400 to-cyan-500 text-white shadow-lg">
                            <span class="material-symbols-outlined" style="font-size: 28px;">thermostat</span>
                        </div>
                        <div>
                            <h3 class="text-xl font-bold text-text-light dark:text-text-dark">Environmental Conditions</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Climate and weather parameters</p>
                        </div>
                    </div>
                    
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div class="relative">
                            <label class="block text-sm font-semibold text-text-light dark:text-text-dark mb-2" for="temperature">
                                Temperature
                            </label>
                            <div class="relative">
                                <input type="number" step="any" 
                                    class="w-full h-12 pl-4 pr-12 rounded-xl border-2 border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-light dark:text-text-dark font-medium focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all" 
                                    name="temperature" required value="20.87" placeholder="0">
                                <span class="absolute right-4 top-1/2 -translate-y-1/2 text-sm font-medium text-subtle-light dark:text-subtle-dark">°C</span>
                            </div>
                        </div>
                        
                        <div class="relative">
                            <label class="block text-sm font-semibold text-text-light dark:text-text-dark mb-2" for="humidity">
                                Humidity
                            </label>
                            <div class="relative">
                                <input type="number" step="any" 
                                    class="w-full h-12 pl-4 pr-12 rounded-xl border-2 border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-light dark:text-text-dark font-medium focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all" 
                                    name="humidity" required value="82.0" placeholder="0">
                                <span class="absolute right-4 top-1/2 -translate-y-1/2 text-sm font-medium text-subtle-light dark:text-subtle-dark">%</span>
                            </div>
                        </div>
                        
                        <div class="relative">
                            <label class="block text-sm font-semibold text-text-light dark:text-text-dark mb-2" for="pH">
                                pH Value
                            </label>
                            <div class="relative">
                                <input type="number" step="any" 
                                    class="w-full h-12 pl-4 pr-4 rounded-xl border-2 border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-light dark:text-text-dark font-medium focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all" 
                                    name="pH" required value="6.5" placeholder="0.0">
                            </div>
                        </div>
                    </div>
                    
                    <div class="relative">
                        <label class="block text-sm font-semibold text-text-light dark:text-text-dark mb-2" for="rainfall">
                            Rainfall
                        </label>
                        <div class="relative">
                            <input type="number" step="any" 
                                class="w-full h-12 pl-4 pr-12 rounded-xl border-2 border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-light dark:text-text-dark font-medium focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all" 
                                name="rainfall" required value="202.93" placeholder="0">
                            <span class="absolute right-4 top-1/2 -translate-y-1/2 text-sm font-medium text-subtle-light dark:text-subtle-dark">mm</span>
                        </div>
                    </div>
                </div>

                <!-- Submit Button -->
                <div class="flex justify-center">
                    <button type="submit" class="group relative flex items-center justify-center gap-3 h-14 px-12 rounded-xl bg-gradient-to-r from-primary to-green-400 text-white text-base font-black shadow-xl hover:shadow-2xl hover:scale-105 transition-all duration-300">
                        <span class="material-symbols-outlined" style="font-size: 24px;">agriculture</span>
                        <span>Get Crop Recommendation</span>
                        <span class="material-symbols-outlined" style="font-size: 20px;">arrow_forward</span>
                    </button>
                </div>
            </form>
        </section>
        
        <!-- Results Section -->
        <section id="results-section" style="<?php echo $has_result ? '' : 'display: none;'; ?>">
            
            <!-- Success Header -->
            <div class="text-center mb-10">
                <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-sm font-semibold mb-4">
                    <span class="material-symbols-outlined" style="font-size: 18px;">check_circle</span>
                    Analysis Complete
                </div>
                <h2 class="text-3xl md:text-4xl font-black text-text-light dark:text-text-dark">
                    Crop Recommendation
                </h2>
            </div>

            <!-- Bento Grid Layout -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-8">
                
                <!-- Recommended Crop Card - Large Feature -->
                <div class="lg:col-span-12 p-10 md:p-12 rounded-3xl bg-gradient-to-br from-primary/10 to-green-100/50 dark:from-primary/5 dark:to-green-900/20 border-2 border-primary/30 shadow-2xl">
                    <div class="text-center">
                        <div class="inline-flex items-center justify-center w-24 h-24 rounded-3xl bg-gradient-to-br from-primary to-green-400 text-white shadow-xl mb-6">
                            <span class="material-symbols-outlined" style="font-size: 56px;">psychiatry</span>
                        </div>
                        <p class="text-sm font-bold uppercase tracking-widest text-primary mb-4">Recommended Crop</p>
                        <h3 class="text-6xl md:text-7xl font-black tracking-tight text-text-light dark:text-text-dark mb-6">
                            <?php echo htmlspecialchars($crop_name); ?>
                        </h3>
                        <p class="text-lg text-subtle-light dark:text-subtle-dark max-w-2xl mx-auto">
                            Based on your soil composition and environmental conditions, this crop is optimally suited for maximum yield
                        </p>
                    </div>
                </div>

                <!-- Input Parameters Card -->
                <div class="lg:col-span-5 p-8 rounded-3xl bg-gradient-to-br from-blue-50 to-white dark:from-blue-900/10 dark:to-surface-dark border-2 border-blue-200 dark:border-blue-800/50 shadow-xl hover:shadow-2xl transition-shadow duration-300">
                    <div class="flex items-center gap-4 mb-6 pb-4 border-b-2 border-blue-200 dark:border-blue-800/50">
                        <div class="flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 text-white shadow-lg">
                            <span class="material-symbols-outlined" style="font-size: 32px;">analytics</span>
                        </div>
                        <h3 class="text-2xl font-black text-text-light dark:text-text-dark">Input Parameters</h3>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-4">
                        <?php if (is_array($input_display)): ?>
                            <?php foreach ($input_display as $key => $value): ?>
                                <div class="p-4 rounded-xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-shadow">
                                    <p class="text-xs font-semibold uppercase tracking-wide text-subtle-light dark:text-subtle-dark mb-1">
                                        <?php echo htmlspecialchars($key); ?>
                                    </p>
                                    <p class="text-2xl font-black text-text-light dark:text-text-dark">
                                        <?php echo htmlspecialchars($value); ?>
                                    </p>
                                </div>
                            <?php endforeach; ?>
                        <?php else: ?>
                            <p class="col-span-2 text-sm text-subtle-light dark:text-subtle-dark"><?php echo htmlspecialchars($input_display); ?></p>
                        <?php endif; ?>
                    </div>
                </div>

                <!-- Cultivation Guide Card -->
                <div class="lg:col-span-7 p-8 rounded-3xl bg-gradient-to-br from-green-50 to-white dark:from-green-900/10 dark:to-surface-dark border-2 border-green-200 dark:border-green-800/50 shadow-xl hover:shadow-2xl transition-shadow duration-300">
                    <div class="flex items-center gap-4 mb-6 pb-4 border-b-2 border-green-200 dark:border-green-800/50">
                        <div class="flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-green-500 to-emerald-500 text-white shadow-lg">
                            <span class="material-symbols-outlined" style="font-size: 32px;">menu_book</span>
                        </div>
                        <h3 class="text-2xl font-black text-text-light dark:text-text-dark">Cultivation Guide</h3>
                    </div>
                    
                    <div class="text-base text-text-light dark:text-text-dark leading-relaxed max-h-[500px] overflow-y-auto custom-scrollbar pr-2">
                        <?php 
                        // Format the advice text with bold support
                        $formatted_advice = $result_data['Advice'];
                        // Convert **text** to <strong>text</strong>
                        $formatted_advice = preg_replace('/\*\*(.*?)\*\*/', '<strong class="font-bold text-primary">$1</strong>', $formatted_advice);
                        echo nl2br($formatted_advice); 
                        ?>
                    </div>
                </div>

            </div>

            <!-- Action Buttons -->
            <div class="flex justify-center">
                <button onclick="window.location.href='recommend_crop.php';" class="flex items-center justify-center gap-2 h-12 px-8 rounded-xl bg-gradient-to-r from-primary to-green-400 text-white text-sm font-black shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-300">
                    <span class="material-symbols-outlined" style="font-size: 20px;">refresh</span>
                    <span>New Analysis</span>
                </button>
            </div>

        </section>

    </div>
</div>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>

<script>
// Mobile Menu Toggle
const mobileMenuToggle = document.getElementById('mobileMenuToggle');
const mobileMenu = document.getElementById('mobileMenu');

if (mobileMenuToggle) {
    mobileMenuToggle.addEventListener('click', function() {
        mobileMenu.classList.toggle('open');
    });
}

// Close mobile menu when clicking outside
document.addEventListener('click', function(event) {
    if (mobileMenu && mobileMenuToggle && 
        !mobileMenu.contains(event.target) && 
        !mobileMenuToggle.contains(event.target)) {
        mobileMenu.classList.remove('open');
    }
});
</script>

</body>
</html>
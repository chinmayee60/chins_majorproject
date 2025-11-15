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
<html class="light" lang="en">
<?php include 'includes/global_head_scripts.php'; ?>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Plant AI</title>
    <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet"/>
    <script>
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    colors: {
                        "primary": "#0df259",
                        "background-light": "#f5f8f6",
                        "background-dark": "#102216",
                        "text-light": "#0d1c12",
                        "text-dark": "#e7f4eb",
                        "card-light": "#ffffff",
                        "card-dark": "#193222",
                        "border-light": "#cee8d7",
                        "border-dark": "#2a4f37"
                    },
                    fontFamily: {"display": ["Inter", "sans-serif"]},
                    borderRadius: {"DEFAULT": "0.5rem", "lg": "0.75rem", "xl": "1rem", "full": "9999px"},
                },
            },
        }
    </script>
    <style>
        .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; }
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
    </style>
</head>
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
            <a href="index.php" class="text-text-light dark:text-text-dark font-bold hover:text-primary transition-colors">Dashboard</a>
            <a href="predict_plant.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 rounded-lg bg-primary/20 text-text-light hover:bg-primary/30 transition-colors font-bold">Logout</a>
        </nav>
        
        <button id="mobileMenuToggle" class="md:hidden">
            <span class="material-symbols-outlined" style="font-size: 24px;">menu</span>
        </button>
        
        <div id="mobileMenu" class="md:hidden">
            <a href="auth.php?landing=1" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Home</a>
            <a href="index.php" class="px-4 py-2 text-sm font-bold hover:bg-primary/10 rounded transition-colors">Dashboard</a>
            <a href="predict_plant.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 bg-primary/20 rounded text-sm font-bold hover:bg-primary/30 transition-colors">Logout</a>
        </div>
    </div>
</header>

<div class="flex flex-1 justify-center py-8">
    <div class="w-full max-w-[1200px] px-4 sm:px-6 lg:px-8">
                
                <!-- Welcome Hero Section -->
                <div class="relative mb-10 p-8 md:p-10 rounded-3xl bg-gradient-to-br from-primary/10 via-green-50 to-emerald-50 dark:from-primary/5 dark:via-green-900/10 dark:to-emerald-900/10 border-2 border-primary/20 shadow-xl overflow-hidden">
                    <div class="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl"></div>
                    <div class="absolute bottom-0 left-0 w-48 h-48 bg-green-400/10 rounded-full blur-2xl"></div>
                    
                    <div class="relative">
                        <div class="flex items-start justify-between flex-wrap gap-6">
                            <div class="flex-1 min-w-[300px]">
                                <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white dark:bg-gray-800 shadow-sm mb-4">
                                    <span class="relative flex h-2 w-2">
                                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                                        <span class="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                                    </span>
                                    <span class="text-xs font-bold text-text-light dark:text-text-dark">Active Session</span>
                                </div>
                                <h1 class="text-4xl md:text-5xl font-black text-text-light dark:text-text-dark mb-3">
                                    👋 Welcome back, <span class="text-primary"><?php echo htmlspecialchars($username); ?></span>!
                                </h1>
                                <p class="text-lg text-subtle-light dark:text-subtle-dark">
                                    Access all your AI-powered plant analysis tools from your dashboard.
                                </p>
                            </div>
                            
                            <div class="flex gap-4">
                                <div class="p-4 rounded-2xl bg-white dark:bg-gray-800 shadow-lg border border-gray-200 dark:border-gray-700">
                                    <div class="flex items-center gap-3">
                                        <div class="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 text-white">
                                            <span class="material-symbols-outlined" style="font-size: 28px;">assessment</span>
                                        </div>
                                        <div>
                                            <p class="text-3xl font-black text-text-light dark:text-text-dark"><?php echo count($history); ?></p>
                                            <p class="text-xs font-semibold text-subtle-light dark:text-subtle-dark">Total Analyses</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <?php if ($status_message): ?>
                    <div class="mb-8 p-6 rounded-2xl bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border-2 border-primary/30 shadow-lg">
                        <div class="flex items-start gap-4">
                            <div class="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-green-400 text-white shadow-lg flex-shrink-0">
                                <span class="material-symbols-outlined" style="font-size: 28px;">check_circle</span>
                            </div>
                            <div class="flex-1">
                                <p class="font-black text-xl text-text-light dark:text-text-dark mb-3"><?php echo $status_message; ?></p>
                                <div class="rounded-xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
                                    <pre class="text-sm whitespace-pre-wrap overflow-x-auto text-text-light dark:text-text-dark"><?php echo htmlspecialchars($result_content); ?></pre>
                                </div>
                            </div>
                        </div>
                    </div>
                <?php endif; ?>

                <!-- Quick Actions Grid -->
                <div class="mb-12">
                    <h2 class="text-2xl font-black text-text-light dark:text-text-dark mb-6 flex items-center gap-2">
                        <span class="material-symbols-outlined" style="font-size: 28px;">rocket_launch</span>
                        Quick Actions
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    
                    <a href="predict_plant.php" class="group relative flex flex-col gap-5 rounded-2xl bg-gradient-to-br from-green-50 to-white dark:from-green-900/10 dark:to-surface-dark border-2 border-green-200 dark:border-green-800/30 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-300">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-green-500 to-emerald-500 text-white shadow-lg group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined" style="font-size: 32px;">eco</span>
                            </div>
                            <span class="material-symbols-outlined text-green-500 group-hover:translate-x-1 transition-transform" style="font-size: 24px;">arrow_forward</span>
                        </div>
                        <div>
                            <h3 class="text-xl font-black text-text-light dark:text-text-dark mb-2">Leaf Disease Detection</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Upload leaf images for AI-powered disease analysis</p>
                        </div>
                    </a>

                    <a href="predict_plant.php?mode=wheat" class="group relative flex flex-col gap-5 rounded-2xl bg-gradient-to-br from-orange-50 to-white dark:from-orange-900/10 dark:to-surface-dark border-2 border-orange-200 dark:border-orange-800/30 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-300">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-orange-500 to-amber-500 text-white shadow-lg group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined" style="font-size: 32px;">agriculture</span>
                            </div>
                            <span class="material-symbols-outlined text-orange-500 group-hover:translate-x-1 transition-transform" style="font-size: 24px;">arrow_forward</span>
                        </div>
                        <div>
                            <h3 class="text-xl font-black text-text-light dark:text-text-dark mb-2">Stem Disease Detection</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Diagnose wheat stem and root diseases</p>
                        </div>
                    </a>

                    <a href="recommend_crop.php" class="group relative flex flex-col gap-5 rounded-2xl bg-gradient-to-br from-blue-50 to-white dark:from-blue-900/10 dark:to-surface-dark border-2 border-blue-200 dark:border-blue-800/30 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-300">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 text-white shadow-lg group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined" style="font-size: 32px;">analytics</span>
                            </div>
                            <span class="material-symbols-outlined text-blue-500 group-hover:translate-x-1 transition-transform" style="font-size: 24px;">arrow_forward</span>
                        </div>
                        <div>
                            <h3 class="text-xl font-black text-text-light dark:text-text-dark mb-2">Crop Recommendation</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Get crop suggestions based on soil data</p>
                        </div>
                    </a>
                </div>
                </div>

                <!-- Recent Analysis History -->
                <div>
                    <div class="flex items-center justify-between mb-6">
                        <h2 class="text-2xl font-black text-text-light dark:text-text-dark flex items-center gap-2">
                            <span class="material-symbols-outlined" style="font-size: 28px;">history</span>
                            Recent Analysis History
                        </h2>
                        <?php if (!empty($history)): ?>
                        <span class="px-3 py-1.5 rounded-full bg-primary/20 text-primary text-sm font-bold">
                            <?php echo count($history); ?> Records
                        </span>
                        <?php endif; ?>
                    </div>
                
                <?php if (!empty($history)): ?>
                    <div class="grid grid-cols-1 gap-4">
                        <?php foreach ($history as $item): 
                            $color_map = [
                                'plant' => ['gradient' => 'from-green-500 to-emerald-500', 'bg' => 'from-green-50 to-emerald-50', 'border' => 'border-green-200', 'icon' => 'eco', 'text' => 'Leaf Disease Detection'],
                                'wheat' => ['gradient' => 'from-orange-500 to-amber-500', 'bg' => 'from-orange-50 to-amber-50', 'border' => 'border-orange-200', 'icon' => 'agriculture', 'text' => 'Stem Disease Detection'],
                                'crop' => ['gradient' => 'from-blue-500 to-cyan-500', 'bg' => 'from-blue-50 to-cyan-50', 'border' => 'border-blue-200', 'icon' => 'analytics', 'text' => 'Crop Recommendation']
                            ];
                            $colors = $color_map[$item['model_type']] ?? $color_map['plant'];
                        ?>
                            <div class="group relative rounded-2xl bg-gradient-to-br <?php echo $colors['bg']; ?> dark:from-gray-800/50 dark:to-gray-900/50 border-2 <?php echo $colors['border']; ?> dark:border-gray-700 p-5 hover:shadow-xl transition-all duration-300">
                                <div class="flex items-center justify-between gap-4">
                                    <div class="flex items-center gap-4 flex-1">
                                        <div class="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br <?php echo $colors['gradient']; ?> text-white shadow-lg group-hover:scale-110 transition-transform">
                                            <span class="material-symbols-outlined" style="font-size: 24px;"><?php echo $colors['icon']; ?></span>
                                        </div>
                                        <div class="flex-1">
                                            <p class="font-black text-lg text-text-light dark:text-text-dark"><?php echo $colors['text']; ?></p>
                                            <div class="flex items-center gap-2 mt-1">
                                                <span class="material-symbols-outlined text-subtle-light dark:text-subtle-dark" style="font-size: 14px;">schedule</span>
                                                <p class="text-sm text-subtle-light dark:text-subtle-dark font-medium"><?php echo date("M d, Y - H:i", strtotime($item['timestamp'])); ?></p>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="flex items-center gap-2">
                                        <button class="export-result-btn flex items-center gap-2 rounded-xl h-11 px-4 bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-700 text-text-light dark:text-text-dark hover:bg-gray-50 dark:hover:bg-gray-700 transition-all text-sm font-bold shadow-sm hover:shadow-md" data-file="<?php echo htmlspecialchars($item['result_file']); ?>" data-type="<?php echo $colors['text']; ?>" data-timestamp="<?php echo date('Y-m-d_H-i-s', strtotime($item['timestamp'])); ?>">
                                            <span class="material-symbols-outlined" style="font-size: 18px;">download</span>
                                            Export
                                        </button>
                                        <button class="view-result-btn flex items-center gap-2 rounded-xl h-11 px-5 bg-gradient-to-r <?php echo $colors['gradient']; ?> text-white hover:shadow-lg transition-all text-sm font-bold shadow-md hover:scale-105" data-file="<?php echo htmlspecialchars($item['result_file']); ?>">
                                            <span class="material-symbols-outlined" style="font-size: 18px;">visibility</span>
                                            View Report
                                        </button>
                                    </div>
                                </div>
                            </div>
                        <?php endforeach; ?>
                    </div>
                <?php else: ?>
                    <div class="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/30 p-16 text-center">
                        <div class="flex items-center justify-center w-20 h-20 rounded-full bg-gray-200 dark:bg-gray-800 mb-4">
                            <span class="material-symbols-outlined text-gray-400 dark:text-gray-600" style="font-size: 48px;">history</span>
                        </div>
                        <p class="text-lg font-bold text-text-light dark:text-text-dark mb-2">No Analysis History Yet</p>
                        <p class="text-subtle-light dark:text-subtle-dark">Start by running your first diagnosis above!</p>
                    </div>
                <?php endif; ?>
                </div>

            </div>
        </div>
    </div>
</div>

<!-- Modal for viewing results -->
<div id="resultModal" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 hidden items-center justify-center p-6 md:p-8 pt-24 md:pt-28">
    <div class="bg-white dark:bg-gray-900 rounded-2xl max-w-5xl w-full max-h-[70vh] flex flex-col shadow-2xl border-2 border-gray-200 dark:border-gray-700">
        <!-- Professional Header with Plant AI Branding -->
        <div id="modalHeader" class="flex items-center justify-between p-3 md:p-4 border-b-2 border-gray-200 dark:border-gray-700 bg-gradient-to-r from-primary/10 via-green-50 to-emerald-50 dark:from-primary/20 dark:via-green-900/10 dark:to-emerald-900/10 flex-shrink-0">
            <div class="flex items-center gap-2 md:gap-3">
                <div class="flex items-center justify-center w-9 h-9 md:w-10 md:h-10 rounded-xl bg-gradient-to-br from-primary to-green-400 text-white shadow-lg">
                    <span class="material-symbols-outlined" style="font-size: 20px;">eco</span>
                </div>
                <div>
                    <div class="flex items-center gap-2">
                        <h3 class="text-lg md:text-xl font-black text-text-light dark:text-text-dark">Plant AI</h3>
                        <span class="px-1.5 py-0.5 rounded bg-primary/20 text-primary text-[10px] font-bold">PRO</span>
                    </div>
                    <p class="text-[10px] md:text-xs font-semibold text-subtle-light dark:text-subtle-dark">Report Viewer</p>
                </div>
            </div>
            <div class="flex items-center gap-1.5 md:gap-2 export-buttons">
                <button id="exportModalBtn" class="flex items-center gap-1 rounded-lg h-8 md:h-9 px-2.5 md:px-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:shadow-lg transition-all text-[11px] md:text-xs font-bold shadow-md hover:scale-105">
                    <span class="material-symbols-outlined" style="font-size: 14px;">download</span>
                    <span class="hidden sm:inline">Export Image</span>
                </button>
                <button onclick="closeModal()" class="flex items-center justify-center rounded-lg w-8 h-8 md:w-9 md:h-9 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
                    <span class="material-symbols-outlined text-gray-600 dark:text-gray-400" style="font-size: 18px;">close</span>
                </button>
            </div>
        </div>
        
        <!-- Scrollable Content Area -->
        <div class="p-3 md:p-4 overflow-y-auto flex-1" id="modalResultContent">
            <!-- Dynamic content loaded here -->
        </div>
    </div>
</div>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
let currentResultData = '';
let currentResultFile = '';

function parseResultFile(data) {
    const lines = data.split('\n');
    const result = {
        modelType: '',
        timestamp: '',
        userId: '',
        disease: '',
        confidence: '',
        inputDetail: '',
        imagePath: '',
        advice: '',
        severity: 'Medium',
        crop: '',
        nitrogen: '',
        phosphorus: '',
        potassium: '',
        temperature: '',
        humidity: '',
        ph: '',
        rainfall: ''
    };
    
    let inAdviceSection = false;
    let adviceText = [];
    
    for (let line of lines) {
        line = line.trim();
        
        if (line.startsWith('Model Type:')) {
            result.modelType = line.split(':')[1].trim().toLowerCase();
        } else if (line.startsWith('Timestamp:')) {
            result.timestamp = line.split('Timestamp:')[1].trim();
        } else if (line.startsWith('User ID:')) {
            result.userId = line.split(':')[1].trim();
        } else if (line.startsWith('Detected Disease:')) {
            result.disease = line.split('Detected Disease:')[1].trim().replace(/___/g, ' - ');
        } else if (line.startsWith('Confidence:')) {
            result.confidence = line.split(':')[1].trim();
        } else if (line.startsWith('Input Detail:')) {
            result.inputDetail = line.split('Input Detail:')[1].trim();
        } else if (line.startsWith('Image Path:')) {
            result.imagePath = line.split('Image Path:')[1].trim();
        } else if (line.startsWith('Recommended Crop:')) {
            result.crop = line.split(':')[1].trim();
        } else if (line.startsWith('Input Conditions:')) {
            // Parse JSON input conditions for crop recommendations
            try {
                const jsonStr = line.split('Input Conditions:')[1].trim();
                const conditions = JSON.parse(jsonStr);
                result.nitrogen = conditions.N ? conditions.N.toString() : '';
                result.phosphorus = conditions.P ? conditions.P.toString() : '';
                result.potassium = conditions.K ? conditions.K.toString() : '';
                result.temperature = conditions.temperature ? conditions.temperature.toFixed(2) + '°C' : '';
                result.humidity = conditions.humidity ? conditions.humidity.toString() + '%' : '';
                result.ph = conditions.pH ? conditions.pH.toString() : '';
                result.rainfall = conditions.rainfall ? conditions.rainfall.toFixed(2) + ' mm' : '';
            } catch (e) {
                console.error('Error parsing Input Conditions JSON:', e);
            }
        } else if (line.includes('Nitrogen (N):')) {
            result.nitrogen = line.split(':')[1].trim();
        } else if (line.includes('Phosphorus (P):')) {
            result.phosphorus = line.split(':')[1].trim();
        } else if (line.includes('Potassium (K):')) {
            result.potassium = line.split(':')[1].trim();
        } else if (line.includes('Temperature:')) {
            result.temperature = line.split(':')[1].trim();
        } else if (line.includes('Humidity:')) {
            result.humidity = line.split(':')[1].trim();
        } else if (line.includes('pH Value:')) {
            result.ph = line.split(':')[1].trim();
        } else if (line.includes('Rainfall:')) {
            result.rainfall = line.split(':')[1].trim();
        } else if (line.includes('--- AI Generated Advice ---') || line.includes('--- Cultivation Guide ---')) {
            inAdviceSection = true;
            continue;
        } else if (inAdviceSection && line && !line.startsWith('--')) {
            adviceText.push(line);
        }
    }
    
    result.advice = adviceText.join('\n');
    
    // Determine severity
    if (result.confidence) {
        const conf = parseFloat(result.confidence);
        if (conf >= 90) result.severity = 'High';
        else if (conf >= 70) result.severity = 'Medium';
        else result.severity = 'Low';
    }
    
    return result;
}

function formatAdviceText(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-gray-900 dark:text-white">$1</strong>')
        .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
        .replace(/\n/g, '<br>');
}

function renderPlantWheatResults(data) {
    const severityColors = {
        'High': 'bg-gradient-to-r from-red-500 to-red-600',
        'Medium': 'bg-gradient-to-r from-yellow-500 to-orange-500',
        'Low': 'bg-gradient-to-r from-blue-500 to-cyan-500'
    };
    
    const severityBorder = {
        'High': 'border-red-700',
        'Medium': 'border-yellow-600',
        'Low': 'border-blue-600'
    };
    
    const iconMap = {
        'plant': 'eco',
        'wheat': 'agriculture'
    };
    
    const colorMap = {
        'plant': { gradient: 'from-green-500 to-emerald-500', light: 'from-green-50 to-emerald-50', border: 'border-green-200' },
        'wheat': { gradient: 'from-orange-500 to-amber-500', light: 'from-orange-50 to-amber-50', border: 'border-orange-200' }
    };
    
    const colors = colorMap[data.modelType] || colorMap['plant'];
    const confidenceNum = parseFloat(data.confidence);
    
    return `
        <!-- Success Badge -->
        <div class="text-center mb-8">
            <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-sm font-semibold mb-3">
                <span class="material-symbols-outlined" style="font-size: 18px;">check_circle</span>
                Analysis Complete
            </div>
            <h2 class="text-3xl md:text-4xl font-black text-text-light dark:text-text-dark">Diagnosis Results</h2>
        </div>

        <!-- Bento Grid Layout -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
            
            <!-- Image Card -->
            ${data.imagePath ? `
            <div class="lg:col-span-5 rounded-2xl overflow-hidden shadow-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                <div class="relative aspect-[4/3] bg-cover bg-center" style='background-image: url("${data.imagePath}");'>
                    <div class="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                    <div class="absolute bottom-4 left-4 right-4">
                        <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm">
                            <span class="material-symbols-outlined text-primary" style="font-size: 16px;">photo_camera</span>
                            <span class="text-xs font-semibold text-gray-700 dark:text-gray-300">Analyzed Image</span>
                        </div>
                    </div>
                </div>
            </div>
            ` : ''}

            <!-- Disease Info Card -->
            <div class="${data.imagePath ? 'lg:col-span-7' : 'lg:col-span-12'} space-y-6">
                
                <!-- Disease Name Card -->
                <div class="rounded-2xl bg-gradient-to-br ${colors.light} dark:from-gray-800 dark:to-gray-700 p-6 ${colors.border} dark:border-gray-600 shadow-lg border">
                    <div class="flex items-start justify-between mb-4">
                        <div class="flex-1">
                            <p class="text-sm font-bold uppercase tracking-wider text-primary dark:text-primary mb-2">Detected Disease</p>
                            <h3 class="text-2xl md:text-3xl font-black text-gray-900 dark:text-white leading-tight break-words">
                                ${data.disease || 'Unknown Disease'}
                            </h3>
                        </div>
                        <div class="flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-br ${colors.gradient} flex items-center justify-center shadow-lg">
                            <span class="material-symbols-outlined text-white" style="font-size: 28px;">${iconMap[data.modelType] || 'eco'}</span>
                        </div>
                    </div>
                    ${data.inputDetail ? `<p class="text-sm text-gray-600 dark:text-gray-400 mt-3"><strong>Source:</strong> ${data.inputDetail}</p>` : ''}
                </div>

                <!-- Metrics Grid -->
                <div class="grid grid-cols-2 gap-5">
                    <!-- Confidence Card -->
                    <div class="group relative rounded-2xl bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 p-6 border-2 border-blue-200 dark:border-blue-800 shadow-lg hover:shadow-2xl transition-all">
                        <div class="absolute top-4 right-4 w-12 h-12 rounded-xl bg-blue-500/10 dark:bg-blue-500/20 flex items-center justify-center">
                            <span class="material-symbols-outlined text-blue-600 dark:text-blue-400" style="font-size: 28px;">analytics</span>
                        </div>
                        <div class="mb-3">
                            <p class="text-sm font-bold uppercase tracking-wider text-blue-700 dark:text-blue-400 mb-1">Confidence Score</p>
                        </div>
                        <p class="text-5xl font-black text-blue-600 dark:text-blue-400 mb-4 tracking-tight">${data.confidence || 'N/A'}</p>
                        ${data.confidence ? `
                        <div class="w-full bg-blue-200 dark:bg-blue-900/40 rounded-full h-3 overflow-hidden shadow-inner">
                            <div class="bg-gradient-to-r from-blue-500 to-cyan-400 h-3 rounded-full transition-all duration-700" style="width: ${confidenceNum}%;"></div>
                        </div>
                        <p class="mt-2 text-xs font-semibold text-blue-600/70 dark:text-blue-400/70">Model Accuracy</p>
                        ` : ''}
                    </div>

                    <!-- Severity Card -->
                    <div class="group relative rounded-2xl bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 p-6 border-2 border-orange-200 dark:border-orange-800 shadow-lg hover:shadow-2xl transition-all">
                        <div class="absolute top-4 right-4 w-12 h-12 rounded-xl bg-orange-500/10 dark:bg-orange-500/20 flex items-center justify-center">
                            <span class="material-symbols-outlined text-orange-600 dark:text-orange-400" style="font-size: 28px;">warning</span>
                        </div>
                        <div class="mb-3">
                            <p class="text-sm font-bold uppercase tracking-wider text-orange-700 dark:text-orange-400 mb-1">Severity Level</p>
                        </div>
                        <div class="mt-3">
                            <span class="inline-flex items-center gap-2 rounded-xl ${severityColors[data.severity]} px-6 py-3 text-xl font-black text-white shadow-xl border-2 ${severityBorder[data.severity]}">
                                <span class="material-symbols-outlined" style="font-size: 24px;">
                                    ${data.severity === 'High' ? 'priority_high' : 'info'}
                                </span>
                                ${data.severity}
                            </span>
                        </div>
                        <p class="mt-4 text-xs font-semibold text-orange-600/70 dark:text-orange-400/70">Risk Assessment</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- AI Care Instructions -->
        ${data.advice ? `
        <div class="rounded-2xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-xl overflow-hidden">
            <div class="bg-gradient-to-r from-primary/10 to-green-500/10 dark:from-primary/20 dark:to-green-500/20 px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-green-500 flex items-center justify-center shadow-md">
                        <span class="material-symbols-outlined text-white" style="font-size: 24px;">psychology</span>
                    </div>
                    <div>
                        <h3 class="text-xl font-black text-gray-900 dark:text-white">Treatment Recommendations</h3>
                        <p class="text-xs text-gray-600 dark:text-gray-400">AI-Powered Guidance</p>
                    </div>
                </div>
            </div>
            <div class="p-6">
                <div class="text-gray-700 dark:text-gray-300 leading-relaxed space-y-4 text-base">
                    ${formatAdviceText(data.advice)}
                </div>
            </div>
        </div>
        ` : ''}

        <!-- Metadata Footer -->
        <div class="mt-6 p-4 rounded-xl bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700">
            <div class="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
                <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined" style="font-size: 16px;">schedule</span>
                    <span><strong>Analyzed:</strong> ${data.timestamp || 'Unknown'}</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined" style="font-size: 16px;">badge</span>
                    <span><strong>Analysis ID:</strong> ${data.userId || 'N/A'}</span>
                </div>
            </div>
        </div>
    `;
}

function renderCropResults(data) {
    return `
        <!-- Success Badge -->
        <div class="text-center mb-8">
            <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-sm font-semibold mb-3">
                <span class="material-symbols-outlined" style="font-size: 18px;">check_circle</span>
                Analysis Complete
            </div>
            <h2 class="text-3xl md:text-4xl font-black text-text-light dark:text-text-dark">Crop Recommendation</h2>
        </div>

        <!-- Bento Grid Layout -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
            
            <!-- Recommended Crop Card -->
            <div class="lg:col-span-4 rounded-2xl bg-gradient-to-br from-green-50 to-emerald-50 dark:from-gray-800 dark:to-gray-700 p-6 border-2 border-green-200 dark:border-green-900/30 shadow-xl">
                <div class="flex flex-col items-center text-center h-full justify-center">
                    <div class="w-20 h-20 rounded-2xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center shadow-xl mb-4">
                        <span class="material-symbols-outlined text-white" style="font-size: 48px;">agriculture</span>
                    </div>
                    <p class="text-sm font-bold uppercase tracking-wider text-green-700 dark:text-green-400 mb-3">Best Crop Match</p>
                    <h3 class="text-4xl font-black text-gray-900 dark:text-white leading-tight mb-4">
                        ${data.crop || 'Unknown Crop'}
                    </h3>
                    ${data.confidence ? `
                    <div class="mt-auto w-full p-4 rounded-xl bg-white dark:bg-gray-800 border border-green-200 dark:border-gray-700">
                        <p class="text-sm font-bold text-gray-600 dark:text-gray-400 mb-2">Suitability Score</p>
                        <p class="text-3xl font-black text-green-600 dark:text-green-400">${data.confidence}</p>
                    </div>
                    ` : ''}
                </div>
            </div>

            <!-- Input Parameters Card -->
            <div class="lg:col-span-8 rounded-2xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-xl p-6">
                <div class="flex items-center gap-3 mb-6 pb-4 border-b border-gray-200 dark:border-gray-700">
                    <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-md">
                        <span class="material-symbols-outlined text-white" style="font-size: 24px;">science</span>
                    </div>
                    <h3 class="text-xl font-black text-gray-900 dark:text-white">Soil & Environmental Data</h3>
                </div>
                
                <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    ${data.nitrogen ? `
                    <div class="p-4 rounded-xl bg-gradient-to-br from-green-50 to-white dark:from-green-900/20 dark:to-gray-800 border border-green-200 dark:border-green-800">
                        <p class="text-xs font-bold uppercase text-green-700 dark:text-green-400 mb-1">Nitrogen (N)</p>
                        <p class="text-2xl font-black text-gray-900 dark:text-white">${data.nitrogen}</p>
                    </div>
                    ` : ''}
                    ${data.phosphorus ? `
                    <div class="p-4 rounded-xl bg-gradient-to-br from-orange-50 to-white dark:from-orange-900/20 dark:to-gray-800 border border-orange-200 dark:border-orange-800">
                        <p class="text-xs font-bold uppercase text-orange-700 dark:text-orange-400 mb-1">Phosphorus (P)</p>
                        <p class="text-2xl font-black text-gray-900 dark:text-white">${data.phosphorus}</p>
                    </div>
                    ` : ''}
                    ${data.potassium ? `
                    <div class="p-4 rounded-xl bg-gradient-to-br from-purple-50 to-white dark:from-purple-900/20 dark:to-gray-800 border border-purple-200 dark:border-purple-800">
                        <p class="text-xs font-bold uppercase text-purple-700 dark:text-purple-400 mb-1">Potassium (K)</p>
                        <p class="text-2xl font-black text-gray-900 dark:text-white">${data.potassium}</p>
                    </div>
                    ` : ''}
                    ${data.temperature ? `
                    <div class="p-4 rounded-xl bg-gradient-to-br from-red-50 to-white dark:from-red-900/20 dark:to-gray-800 border border-red-200 dark:border-red-800">
                        <p class="text-xs font-bold uppercase text-red-700 dark:text-red-400 mb-1">Temperature</p>
                        <p class="text-2xl font-black text-gray-900 dark:text-white">${data.temperature}</p>
                    </div>
                    ` : ''}
                    ${data.humidity ? `
                    <div class="p-4 rounded-xl bg-gradient-to-br from-blue-50 to-white dark:from-blue-900/20 dark:to-gray-800 border border-blue-200 dark:border-blue-800">
                        <p class="text-xs font-bold uppercase text-blue-700 dark:text-blue-400 mb-1">Humidity</p>
                        <p class="text-2xl font-black text-gray-900 dark:text-white">${data.humidity}</p>
                    </div>
                    ` : ''}
                    ${data.ph ? `
                    <div class="p-4 rounded-xl bg-gradient-to-br from-yellow-50 to-white dark:from-yellow-900/20 dark:to-gray-800 border border-yellow-200 dark:border-yellow-800">
                        <p class="text-xs font-bold uppercase text-yellow-700 dark:text-yellow-400 mb-1">pH Value</p>
                        <p class="text-2xl font-black text-gray-900 dark:text-white">${data.ph}</p>
                    </div>
                    ` : ''}
                    ${data.rainfall ? `
                    <div class="p-4 rounded-xl bg-gradient-to-br from-cyan-50 to-white dark:from-cyan-900/20 dark:to-gray-800 border border-cyan-200 dark:border-cyan-800">
                        <p class="text-xs font-bold uppercase text-cyan-700 dark:text-cyan-400 mb-1">Rainfall</p>
                        <p class="text-2xl font-black text-gray-900 dark:text-white">${data.rainfall}</p>
                    </div>
                    ` : ''}
                </div>
                ${!data.nitrogen && !data.phosphorus && !data.potassium && !data.temperature && !data.humidity && !data.ph && !data.rainfall ? `
                <div class="text-center py-8">
                    <span class="material-symbols-outlined text-gray-400 dark:text-gray-600 mb-2" style="font-size: 40px;">warning</span>
                    <p class="text-sm text-gray-600 dark:text-gray-400">No environmental data available for this analysis</p>
                </div>
                ` : ''}
            </div>
        </div>

        <!-- Cultivation Guide -->
        ${data.advice ? `
        <div class="rounded-2xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-xl overflow-hidden">
            <div class="bg-gradient-to-r from-green-500 to-emerald-500 px-6 py-4">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-lg bg-white/20 backdrop-blur-sm flex items-center justify-center">
                        <span class="material-symbols-outlined text-white" style="font-size: 24px;">menu_book</span>
                    </div>
                    <div>
                        <h3 class="text-xl font-black text-white">Cultivation Guide</h3>
                        <p class="text-xs text-white/80">Expert Growing Tips</p>
                    </div>
                </div>
            </div>
            <div class="p-6">
                <div class="text-gray-700 dark:text-gray-300 leading-relaxed space-y-4 text-base">
                    ${formatAdviceText(data.advice)}
                </div>
            </div>
        </div>
        ` : ''}

        <!-- Metadata Footer -->
        <div class="mt-6 p-4 rounded-xl bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700">
            <div class="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
                <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined" style="font-size: 16px;">schedule</span>
                    <span><strong>Generated:</strong> ${data.timestamp || 'Unknown'}</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined" style="font-size: 16px;">badge</span>
                    <span><strong>Report ID:</strong> ${data.userId || 'N/A'}</span>
                </div>
            </div>
        </div>
    `;
}

$(document).ready(function() {
    $('.view-result-btn').on('click', function() {
        var filePath = $(this).data('file');
        currentResultFile = filePath;
        
        $.ajax({
            url: filePath,
            type: 'GET',
            dataType: 'text',
            success: function(data) {
                currentResultData = data;
                const parsedData = parseResultFile(data);
                
                let html = '';
                if (parsedData.modelType === 'crop') {
                    html = renderCropResults(parsedData);
                } else {
                    html = renderPlantWheatResults(parsedData);
                }
                
                $('#modalResultContent').html(html);
                $('#resultModal').removeClass('hidden').addClass('flex');
            },
            error: function() {
                $('#modalResultContent').html(`
                    <div class="text-center py-12">
                        <span class="material-symbols-outlined text-red-500 mb-4" style="font-size: 48px;">error</span>
                        <p class="text-lg font-bold text-red-600">Error loading result file</p>
                        <p class="text-sm text-gray-600 mt-2">Please check file permissions or try again.</p>
                    </div>
                `);
                $('#resultModal').removeClass('hidden').addClass('flex');
            }
        });
    });

    $('.export-result-btn').on('click', function() {
        var filePath = $(this).data('file');
        var analysisType = $(this).data('type');
        var timestamp = $(this).data('timestamp');
        
        $.ajax({
            url: filePath,
            type: 'GET',
            dataType: 'text',
            success: function(data) {
                const parsedData = parseResultFile(data);
                
                // Open modal first, then export after it's rendered
                let html = '';
                if (parsedData.modelType === 'crop') {
                    html = renderCropResults(parsedData);
                } else {
                    html = renderPlantWheatResults(parsedData);
                }
                
                $('#modalResultContent').html(html);
                $('#resultModal').removeClass('hidden').addClass('flex');
                
                // Wait for render, then export
                setTimeout(() => {
                    exportCurrentModal(analysisType, timestamp);
                }, 1000);
            },
            error: function() {
                alert('Error loading result file for export.');
            }
        });
    });

    $('#exportModalBtn').on('click', function() {
        if (currentResultData) {
            const parsedData = parseResultFile(currentResultData);
            const filename = currentResultFile.split('/').pop().split('.')[0];
            const analysisType = parsedData.modelType === 'crop' ? 'Crop_Recommendation' : 
                                parsedData.modelType === 'wheat' ? 'Wheat_Disease_Analysis' : 
                                'Plant_Disease_Analysis';
            
            exportCurrentModal(analysisType, filename);
        }
    });
});

function exportCurrentModal(analysisType, timestamp) {
    // Get the modal dialog
    const modal = document.getElementById('resultModal');
    const modalDialog = modal.querySelector('.bg-white');
    
    // Hide export buttons
    const exportButtons = modal.querySelector('.export-buttons');
    exportButtons.style.visibility = 'hidden';
    
    // Scroll content to top for complete capture
    const contentArea = document.getElementById('modalResultContent');
    const originalScroll = contentArea.scrollTop;
    contentArea.scrollTop = 0;
    
    // Get actual dimensions
    const scrollHeight = contentArea.scrollHeight;
    const headerHeight = document.getElementById('modalHeader').offsetHeight;
    const totalHeight = scrollHeight + headerHeight + 48; // 48 for padding
    
    html2canvas(modalDialog, {
        scale: 2,
        backgroundColor: '#ffffff',
        logging: true,
        useCORS: true,
        allowTaint: false,
        scrollY: -window.scrollY,
        scrollX: -window.scrollX,
        width: modalDialog.offsetWidth,
        height: totalHeight,
        windowHeight: totalHeight,
        onclone: function(clonedDoc) {
            const clonedModal = clonedDoc.querySelector('.bg-white');
            const clonedContent = clonedDoc.getElementById('modalResultContent');
            if (clonedContent) {
                clonedContent.style.maxHeight = 'none';
                clonedContent.style.overflow = 'visible';
                clonedContent.style.height = 'auto';
            }
            if (clonedModal) {
                clonedModal.style.maxHeight = 'none';
                clonedModal.style.height = 'auto';
            }
            // Hide buttons in clone
            const clonedButtons = clonedDoc.querySelector('.export-buttons');
            if (clonedButtons) {
                clonedButtons.style.display = 'none';
            }
        }
    }).then(canvas => {
        // Restore UI
        exportButtons.style.visibility = 'visible';
        contentArea.scrollTop = originalScroll;
        
        // Download
        canvas.toBlob(function(blob) {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = `${analysisType}_${timestamp}.png`;
            link.href = url;
            link.click();
            URL.revokeObjectURL(url);
        }, 'image/png', 1.0);
    }).catch(err => {
        exportButtons.style.visibility = 'visible';
        contentArea.scrollTop = originalScroll;
        console.error('Export failed:', err);
        alert('Failed to export image. Please try again.');
    });
}

function createExportImage(parsedData, analysisType, timestamp) {
    // Not used anymore - using exportCurrentModal instead
}

function downloadResult(data, analysisType, timestamp) {
    const filename = `${analysisType.replace(/\s+/g, '_')}_${timestamp}.txt`;
    const blob = new Blob([data], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function closeModal() {
    $('#resultModal').removeClass('flex').addClass('hidden');
    currentResultData = '';
    currentResultFile = '';
}

// Close modal on outside click
$('#resultModal').on('click', function(e) {
    if (e.target.id === 'resultModal') {
        closeModal();
    }
});
</script>

<script>
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
<?php
session_start();
require_once 'config.php';

// Check if user is already logged in (but allow viewing landing page)
if (isset($_SESSION['user_id']) && !isset($_GET['landing'])) {
    header("Location: index.php");
    exit();
}

$error = '';
$success = '';

// --- Authentication Logic ---
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    if (isset($_POST['login'])) {
        $username = trim($_POST['username']);
        $password = $_POST['password'];

        $stmt = $conn->prepare("SELECT id, username, password_hash FROM users WHERE username = ?");
        $stmt->bind_param("s", $username);
        $stmt->execute();
        $result = $stmt->get_result();

        if ($result->num_rows == 1) {
            $user = $result->fetch_assoc();
            if (password_verify($password, $user['password_hash'])) {
                $_SESSION['user_id'] = $user['id'];
                $_SESSION['username'] = $user['username'];
                header("Location: index.php");
                exit();
            } else {
                $error = "Invalid username or password.";
            }
        } else {
            $error = "Invalid username or password.";
        }
        $stmt->close();
    } elseif (isset($_POST['register'])) {
        $username = trim($_POST['username']);
        $email = trim($_POST['email']);
        $password = $_POST['password'];
        $confirm_password = $_POST['confirm_password'];

        if ($password !== $confirm_password) {
            $error = "Passwords do not match.";
        } else {
            $password_hash = password_hash($password, PASSWORD_DEFAULT);
            $stmt_check = $conn->prepare("SELECT id FROM users WHERE username = ? OR email = ?");
            $stmt_check->bind_param("ss", $username, $email);
            $stmt_check->execute();
            if ($stmt_check->get_result()->num_rows > 0) {
                $error = "Username or Email already exists.";
            } else {
                $stmt_insert = $conn->prepare("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)");
                $stmt_insert->bind_param("sss", $username, $email, $password_hash);
                if ($stmt_insert->execute()) {
                    $success = "Registration successful! You can now log in.";
                } else {
                    $error = "Registration failed. Please try again.";
                }
                $stmt_insert->close();
            }
            $stmt_check->close();
        }
    }
}
$conn->close();

// Logic to determine initial view state: Show login by default, unless explicitly requesting landing page
$show_forms = true;
if (isset($_GET['landing']) || isset($_GET['home'])) {
    $show_forms = false;
}
?>

<!DOCTYPE html>
<html class="light" lang="en">
<?php include 'includes/global_head_scripts.php'; ?>
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Plant AI - Smart Agro AI</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet"/>
<script id="tailwind-config">
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
          fontFamily: {
            "display": ["Inter", "sans-serif"]
          },
          borderRadius: {"DEFAULT": "0.5rem", "lg": "0.75rem", "xl": "1rem", "full": "9999px"},
        },
      },
    }
</script>
<style>
    body { min-height: 100dvh; }
    .form-container-custom { background-color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .bg-cover-placeholder { background-color: #A4BE7B; /* Fallback Green */ }
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
</style>
<script src="//unpkg.com/alpinejs" defer></script> 
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
        
        <?php if (isset($_SESSION['user_id'])): ?>
        <nav class="hidden md:flex items-center gap-6 text-sm font-medium">
            <a href="auth.php?landing=1" class="text-text-light dark:text-text-dark font-bold hover:text-primary transition-colors">Home</a>
            <a href="index.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Dashboard</a>
            <a href="predict_plant.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 rounded-lg bg-primary/20 text-text-light hover:bg-primary/30 transition-colors font-bold">Logout</a>
        </nav>
        
        <button id="mobileMenuToggle" class="md:hidden">
            <span class="material-symbols-outlined" style="font-size: 24px;">menu</span>
        </button>
        
        <div id="mobileMenu" class="md:hidden">
            <a href="auth.php?landing=1" class="px-4 py-2 text-sm font-bold hover:bg-primary/10 rounded transition-colors">Home</a>
            <a href="index.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Dashboard</a>
            <a href="predict_plant.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 bg-primary/20 rounded text-sm font-bold hover:bg-primary/30 transition-colors">Logout</a>
        </div>
        <?php else: ?>
        <nav class="hidden md:flex items-center gap-6 text-sm font-medium">
            <a href="auth.php?landing=1" class="text-text-light dark:text-text-dark font-bold hover:text-primary transition-colors">Home</a>
            <a href="#" onclick="showLoginForm(); return false;" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Diagnosis</a>
            <a href="#" onclick="showLoginForm(); return false;" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Crop Tool</a>
            <a href="#footer" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Contact</a>
            <button onclick="showLoginForm()" class="px-4 py-2 rounded-lg bg-primary/20 text-text-light hover:bg-primary/30 transition-colors font-bold">Login</button>
        </nav>
        
        <button id="mobileMenuToggle" class="md:hidden">
            <span class="material-symbols-outlined" style="font-size: 24px;">menu</span>
        </button>
        
        <div id="mobileMenu" class="md:hidden">
            <a href="auth.php?landing=1" class="px-4 py-2 text-sm font-bold hover:bg-primary/10 rounded transition-colors">Home</a>
            <a href="#" onclick="showLoginForm(); return false;" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Diagnosis</a>
            <a href="#" onclick="showLoginForm(); return false;" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Crop Tool</a>
            <a href="#footer" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Contact</a>
            <button onclick="showLoginForm()" class="px-4 py-2 bg-primary/20 rounded text-sm font-bold hover:bg-primary/30 transition-colors">Login</button>
        </div>
        <?php endif; ?>
    </div>
</header>

<main class="flex flex-1 justify-center py-8">
    <div class="w-full max-w-7xl px-4 sm:px-6 lg:px-8">
    <div class="flex flex-col gap-10 md:gap-12">

        <div id="landing-content" style="<?php echo $show_forms ? 'display: none;' : ''; ?>">
            
            <!-- Hero Section -->
            <section class="mb-12">
                <div class="relative rounded-3xl bg-gradient-to-br from-primary/10 via-green-50 to-emerald-50 dark:from-primary/5 dark:via-green-900/10 dark:to-emerald-900/10 border-2 border-primary/20 shadow-2xl overflow-hidden p-6 md:p-10 lg:p-12">
                    <div class="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl"></div>
                    <div class="absolute bottom-0 left-0 w-48 h-48 bg-green-400/10 rounded-full blur-2xl"></div>
                    
                    <div class="relative grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-10 items-center">
                        <div class="space-y-4">
                            <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white dark:bg-gray-800 shadow-md">
                                <span class="relative flex h-2 w-2">
                                    <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                                    <span class="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                                </span>
                                <span class="text-xs font-bold text-text-light dark:text-text-dark">AI-Powered Analysis</span>
                            </div>
                            
                            <h1 class="text-3xl md:text-4xl lg:text-5xl font-black text-text-light dark:text-text-dark leading-tight">
                                Instant Disease Prediction for Your Plants
                            </h1>
                            <p class="text-base md:text-lg text-subtle-light dark:text-subtle-dark leading-relaxed">
                                Upload a leaf image to get an accurate diagnosis, care instructions, and prevention tips in seconds.
                            </p>
                            
                            <?php if (isset($_SESSION['user_id'])): ?>
                                <a href="predict_plant.php" class="inline-flex items-center gap-2 rounded-xl h-12 px-6 bg-gradient-to-r from-primary to-green-400 text-white text-base font-black shadow-lg hover:shadow-xl hover:scale-105 transition-all">
                                    <span class="material-symbols-outlined" style="font-size: 24px;">upload_file</span>
                                    Upload Image & Diagnose Now
                                </a>
                            <?php else: ?>
                                <button id="trigger-auth-button" class="inline-flex items-center gap-2 rounded-xl h-12 px-6 bg-gradient-to-r from-primary to-green-400 text-white text-base font-black shadow-lg hover:shadow-xl hover:scale-105 transition-all">
                                    <span class="material-symbols-outlined" style="font-size: 24px;">upload_file</span>
                                    Upload Image & Diagnose Now
                                </button>
                            <?php endif; ?>
                        </div>
                        
                        <div class="relative rounded-2xl overflow-hidden shadow-2xl border-4 border-white dark:border-gray-800 h-64 lg:h-80">
                            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuC8EW-cAHjEZXUuKMgZOGKRuP1ecKjk7LoqApPNORW4-9hA0qclo0BoTLbiZ_lhVLUQoYKi0BOL6bLpbhY0MQrSfjuAPy4g3G1V4Pcyp9UZnbkTYMm_dfRZlMYp1Fk0spsv3A2a7v-G5Unt2-MbZsRzw6gte4BTN0WAGfsk6Pi7jNk2Dg5DhvKPWtqzZmawsMJYLcmcpFU_yDm4YoOO_ItKYnNrVI11frt7ff0eUNBu8oHQRZBeU5zQ4RkiSJU27xs1lAk7RDpPijk7" alt="Plant AI Analysis" class="w-full h-full object-cover">
                        </div>
                    </div>
                </div>
            </section>
            
            <!-- Common Diseases Section -->
            <section class="mb-12">
                <div class="text-center mb-8">
                    <span class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/20 text-primary text-sm font-bold mb-4">
                        <span class="material-symbols-outlined" style="font-size: 18px;">verified</span>
                        Common Diseases & Pests
                    </span>
                    <h2 class="text-2xl md:text-3xl font-black text-text-light dark:text-text-dark">What We Can Detect</h2>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-5">
                    <div class="group relative rounded-2xl bg-gradient-to-br from-green-50 to-white dark:from-green-900/10 dark:to-gray-800 border-2 border-green-200 dark:border-green-800/30 p-5 hover:shadow-2xl hover:scale-105 transition-all duration-300">
                        <div class="flex flex-col items-center text-center space-y-3">
                            <div class="w-14 h-14 rounded-2xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined text-white" style="font-size: 28px;">compost</span>
                            </div>
                            <h3 class="text-lg font-black text-text-light dark:text-text-dark">Fungal Infections</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Identify and treat common fungal issues.</p>
                        </div>
                    </div>
                    
                    <div class="group relative rounded-2xl bg-gradient-to-br from-orange-50 to-white dark:from-orange-900/10 dark:to-gray-800 border-2 border-orange-200 dark:border-orange-800/30 p-5 hover:shadow-2xl hover:scale-105 transition-all duration-300">
                        <div class="flex flex-col items-center text-center space-y-3">
                            <div class="w-14 h-14 rounded-2xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined text-white" style="font-size: 28px;">bug_report</span>
                            </div>
                            <h3 class="text-xl font-black text-text-light dark:text-text-dark">Pest Infestations</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Detect pests before they cause damage.</p>
                        </div>
                    </div>
                    
                    <div class="group relative rounded-2xl bg-gradient-to-br from-red-50 to-white dark:from-red-900/10 dark:to-gray-800 border-2 border-red-200 dark:border-red-800/30 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-300">
                        <div class="flex flex-col items-center text-center space-y-4">
                            <div class="w-16 h-16 rounded-2xl bg-gradient-to-br from-red-500 to-pink-500 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined text-white" style="font-size: 32px;">coronavirus</span>
                            </div>
                            <h3 class="text-xl font-black text-text-light dark:text-text-dark">Viral Diseases</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Learn about viral threats to your plants.</p>
                        </div>
                    </div>
                    
                    <div class="group relative rounded-2xl bg-gradient-to-br from-blue-50 to-white dark:from-blue-900/10 dark:to-gray-800 border-2 border-blue-200 dark:border-blue-800/30 p-6 hover:shadow-2xl hover:scale-105 transition-all duration-300">
                        <div class="flex flex-col items-center text-center space-y-4">
                            <div class="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                                <span class="material-symbols-outlined text-white" style="font-size: 32px;">science</span>
                            </div>
                            <h3 class="text-xl font-black text-text-light dark:text-text-dark">Nutrient Deficiencies</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Diagnose and correct nutritional imbalances.</p>
                        </div>
                    </div>
                </div>
            </section>
            
            <!-- Quote Section -->
            <section class="mb-12">
                <div class="rounded-2xl bg-gradient-to-r from-primary/5 to-green-50 dark:from-primary/10 dark:to-gray-800 border-l-4 border-primary p-8 md:p-12">
                    <div class="flex items-start gap-4">
                        <span class="material-symbols-outlined text-primary" style="font-size: 48px;">format_quote</span>
                        <p class="text-2xl md:text-3xl font-bold text-text-light dark:text-text-dark italic leading-relaxed">
                            Knowledge and early action are the best defense against crop failure and plant loss.
                        </p>
                    </div>
                </div>
            </section>

            <!-- Featured Solutions Section -->
            <section class="mb-12">
                <div class="mb-6">
                    <span class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-sm font-bold mb-4">
                        <span class="material-symbols-outlined" style="font-size: 18px;">lightbulb</span>
                        Featured Solutions
                    </span>
                    <h2 class="text-2xl md:text-3xl font-black text-text-light dark:text-text-dark">Identify Common Problems</h2>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
                    <div class="group rounded-2xl overflow-hidden bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-lg hover:shadow-2xl transition-all duration-300">
                        <div class="relative h-48 overflow-hidden">
                            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuAitbTVCzOrApMATo_2UHYvB1Cue3e9I2qPVbuPrcSayhd6kKHSuDBnThjYnA4Cfk7ekKkjeKfv4TMXNo-gCMsn1PfCMkmtElbHU3C5k0ibApmdzEEbO259WQqPrydQO462L3gLDVGmYsEDXnfacW66u_1Kkn0H8Axhqw_XodwOF8sqT10rADHewW9lddUjzKsc2pphZStA2gQvE1TdCkBdBAPdASJLMkaQ6eYOwf3BCSaC26NQmvmI1L0TJudz-5E5q_WBPBUl8_ZQ" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300" alt="Powdery Mildew">
                            <div class="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                        </div>
                        <div class="p-6">
                            <h3 class="text-xl font-black text-text-light dark:text-text-dark mb-2">Powdery Mildew</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">A common fungal disease affecting a wide variety of plants.</p>
                        </div>
                    </div>
                    
                    <div class="group rounded-2xl overflow-hidden bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-lg hover:shadow-2xl transition-all duration-300">
                        <div class="relative h-48 overflow-hidden">
                            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuCbsKbPYK_AmOgYWkjnBE3MVHGohDnjjbTVRYqeFJx--qXRoHig28SlaRwEQ-SsYl7Mqt8O_MwNi8J9ENtFQy-stYSALZUHpyBZAh1sTOMem28DUm4ENpm9g_M_RU24F54wQfjL08MGmM6jVm_VgRILdJ4RuDHPyTvJnn9UbBKamXGdGUVPffSzwEhuPF2BStvk1fbH1KNF7jSjhmH9AwnUhirqccjP7fMrWdQYKyVGw9TKiX4Fjb0yYfs_BO3KxCzagDuq6KRiPGzu" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300" alt="Tomato Blight">
                            <div class="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                        </div>
                        <div class="p-6">
                            <h3 class="text-xl font-black text-text-light dark:text-text-dark mb-2">Tomato Blight</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Learn to spot the early signs of late blight on tomato plants.</p>
                        </div>
                    </div>
                    
                    <div class="group rounded-2xl overflow-hidden bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-lg hover:shadow-2xl transition-all duration-300">
                        <div class="relative h-48 overflow-hidden">
                            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuAuW2sjvggc_1N-I0DowconngESi0PO4lcfv03l81rdoDKHAAWz41m8vTlUXpVDTUeuIlxK5lHlaRujli6D8cFl3YPVSrEFBKO1kAmZU1yZ9MxEp6VWrIsGJ4AJaRDO34JOseq3xsWPKlTrf8x9Q_ZVG3K0dYrEl3VPv_luQrVagnMAU6xRvRRT3qHCdIH__zTyLVgUpg1Y_kPmgMSc32IPvbxgYOkZzqd1ZOtXEMahNQne1mnHsyCSV011jt7BGN4N7DEkHRZ9JeaB" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300" alt="Spider Mites">
                            <div class="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                        </div>
                        <div class="p-6">
                            <h3 class="text-xl font-black text-text-light dark:text-text-dark mb-2">Spider Mites</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Tiny but destructive pests that can quickly damage your plants.</p>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Blog Section -->
            <section class="mb-10">
                <div class="mb-6">
                    <span class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 text-sm font-bold mb-4">
                        <span class="material-symbols-outlined" style="font-size: 18px;">article</span>
                        Learn More
                    </span>
                    <h2 class="text-2xl md:text-3xl font-black text-text-light dark:text-text-dark">From Our Blog</h2>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-5 md:gap-6">
                    <div class="group flex flex-col sm:flex-row gap-5 p-6 rounded-2xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:shadow-xl transition-all duration-300">
                        <div class="w-full sm:w-32 h-32 rounded-xl bg-cover bg-center flex-shrink-0 overflow-hidden">
                            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuC9if1youFqPuKbPEd9V8VInhR-uZpgArWbH8jvIZWu-bBXZno-3y-y8Gb3R6NGJeRTUZBXre0wSwfusEjNllk6KQrWIaapSfVQUXNippYqFZYoQF4pk48AQhjPkIJ7L8SfFlsTHgha8yUjua40ZzYEypkHW4bt5SCf_jkLePDOUH0I0pIVYAhVn3jiuiweNR7YhgW5WOEPdoUPJlQsqe0Pq1_PnsGq3S_sosXThsnqjv9O1BMfROBT7-9L_kBjwqyR1Ucsb3C8dUVJ" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300" alt="Preventing Fungal Growth">
                        </div>
                        <div class="flex flex-col gap-2 flex-1">
                            <h3 class="text-lg font-black text-text-light dark:text-text-dark group-hover:text-primary transition-colors">5 Tips for Preventing Fungal Growth</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">Proactive steps you can take to keep your garden healthy and fungus-free.</p>
                            <a href="#" class="inline-flex items-center gap-1 text-primary font-bold text-sm mt-auto group-hover:gap-2 transition-all">
                                Read More
                                <span class="material-symbols-outlined" style="font-size: 16px;">arrow_forward</span>
                            </a>
                        </div>
                    </div>
                    
                    <div class="group flex flex-col sm:flex-row gap-5 p-6 rounded-2xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:shadow-xl transition-all duration-300">
                        <div class="w-full sm:w-32 h-32 rounded-xl bg-cover bg-center flex-shrink-0 overflow-hidden">
                            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuDBrGw06MSwKkRB2HPGOYKRSnqimQx0x-bew5KRVzBg5EpWx7f2ayWXjZVUjPWmcC4iDQu4awrpX4BWNhPWhISZLKa3vKg2E6fGpxKamwA5AV75BmLNWWFJtLUwUrBXnad7avJ5UwStKumKaNnbNqIIdMOracxjwKekDeaYhb-hIkVJcxS-j6O1asqOU1IpVYNSaDuZqaJWq2rYkEJWBAP7TZIccXd86nibTdVBs8bXxvI5c6oMPh1WyfGYdas5azRleesKOX91lvRx" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300" alt="Identifying Garden Pests">
                        </div>
                        <div class="flex flex-col gap-2 flex-1">
                            <h3 class="text-lg font-black text-text-light dark:text-text-dark group-hover:text-primary transition-colors">Identifying Common Garden Pests</h3>
                            <p class="text-sm text-subtle-light dark:text-subtle-dark">A visual guide to identifying the most common pests in your garden.</p>
                            <a href="#" class="inline-flex items-center gap-1 text-primary font-bold text-sm mt-auto group-hover:gap-2 transition-all">
                                Read More
                                <span class="material-symbols-outlined" style="font-size: 16px;">arrow_forward</span>
                            </a>
                        </div>
                    </div>
                </div>
            </section>
        </div> 
        
        <section id="auth-forms-view" class="flex flex-col items-center justify-center p-4 py-10 flex-1" style="<?php echo $show_forms ? '' : 'display: none;'; ?>">
            <div class="w-full max-w-md form-container-custom rounded-xl p-6 shadow-xl border border-border-light dark:border-border-dark">
                <h2 class="text-2xl font-bold text-text-light dark:text-text-dark mb-4 text-center">Log In to Continue</h2>
                
                <?php if ($error): ?>
                    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4 w-full" role="alert"><?php echo $error; ?></div>
                <?php endif; ?>
                <?php if ($success): ?>
                    <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4 w-full" role="alert"><?php echo $success; ?></div>
                <?php endif; ?>

                <div x-data="{ currentTab: 'login' }" class="w-full">
                    <div class="flex justify-around mb-6 border-b border-border-light dark:border-border-dark">
                        <button @click="currentTab = 'login'" :class="{'border-b-2 border-primary text-primary': currentTab === 'login', 'text-text-secondary dark:text-text-secondary-dark': currentTab !== 'login'}" class="px-4 py-2 font-semibold transition duration-150">Login</button>
                        <button @click="currentTab = 'register'" :class="{'border-b-2 border-primary text-primary': currentTab === 'register', 'text-text-secondary dark:text-text-secondary-dark': currentTab !== 'register'}" class="px-4 py-2 font-semibold transition duration-150">Register</button>
                    </div>

                    <form method="POST" action="auth.php" x-show="currentTab === 'login'">
                        <input type="hidden" name="login" value="1">
                        <div class="mb-4">
                            <label for="login-username" class="block text-sm font-medium text-text-secondary dark:text-text-secondary-dark">Username</label>
                            <input type="text" id="login-username" name="username" required class="mt-1 block w-full rounded-lg border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-2 focus:border-primary focus:ring-primary">
                        </div>
                        <div class="mb-6">
                            <label for="login-password" class="block text-sm font-medium text-text-secondary dark:text-text-secondary-dark">Password</label>
                            <input type="password" id="login-password" name="password" required class="mt-1 block w-full rounded-lg border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-2 focus:border-primary focus:ring-primary">
                        </div>
                        <button type="submit" class="w-full h-12 rounded-lg bg-primary text-text-light text-base font-bold transition duration-150 hover:opacity-90">
                            Log In
                        </button>
                        <div class="mt-4 text-center text-sm text-text-secondary dark:text-text-secondary-dark">Use **testuser / password123** to try.</div>
                    </form>

                    <form method="POST" action="auth.php" x-show="currentTab === 'register'">
                        <input type="hidden" name="register" value="1">
                        <div class="mb-4">
                            <label for="reg-username" class="block text-sm font-medium text-text-secondary dark:text-text-secondary-dark">Username</label>
                            <input type="text" id="reg-username" name="username" required class="mt-1 block w-full rounded-lg border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-2 focus:border-primary focus:ring-primary">
                        </div>
                        <div class="mb-4">
                            <label for="reg-email" class="block text-sm font-medium text-text-secondary dark:text-text-secondary-dark">Email</label>
                            <input type="email" id="reg-email" name="email" required class="mt-1 block w-full rounded-lg border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-2 focus:border-primary focus:ring-primary">
                        </div>
                        <div class="mb-4">
                            <label for="reg-password" class="block text-sm font-medium text-text-secondary dark:text-text-secondary-dark">Password</label>
                            <input type="password" id="reg-password" name="password" required class="mt-1 block w-full rounded-lg border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-2 focus:border-primary focus:ring-primary">
                        </div>
                        <div class="mb-6">
                            <label for="reg-confirm-password" class="block text-sm font-medium text-text-secondary dark:text-text-secondary-dark">Confirm Password</label>
                            <input type="password" id="reg-confirm-password" name="confirm_password" required class="mt-1 block w-full rounded-lg border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-2 focus:border-primary focus:ring-primary">
                        </div>
                        <button type="submit" class="w-full h-12 rounded-lg bg-primary text-text-light text-base font-bold transition duration-150 hover:opacity-90">
                            Create Account
                        </button>
                    </form>
                    <button id="back-to-landing-button-auth" class="mt-6 flex max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 px-5 bg-transparent border border-gray-400 text-text-secondary dark:text-text-secondary-dark hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors gap-2 text-sm font-bold leading-normal tracking-[0.015em]">
                        <span class="truncate">← Back to Info</span>
                    </button>
                </div>
            </div>
        </section>
    
    <footer id="footer" class="mt-10 border-t border-border-light dark:border-border-dark py-6">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div class="col-span-2 md:col-span-1 flex flex-col gap-3">
                <div class="flex items-center gap-2">
                    <div class="size-5 text-primary"><svg fill="currentColor" viewbox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><g clip-path="url(#clip0_6_535)"><path clip-rule="evenodd" d="M47.2426 24L24 47.2426L0.757355 24L24 0.757355L47.2426 24ZM12.2426 21H35.7574L24 9.24264L12.2426 21Z" fill-rule="evenodd"></path></g></svg></div>
                    <h2 class="text-text-light dark:text-text-dark text-lg font-bold">Plant AI</h2>
                </div>
                <p class="text-sm text-text-light/70 dark:text-text-dark/70">Empowering growers with data-driven plant care.</p>
            </div>
            <div class="flex flex-col gap-3"><h4 class="font-bold text-sm text-text-light dark:text-text-dark">Quick Links</h4><a class="text-sm text-text-light/80 dark:text-text-dark/80 hover:text-primary transition-colors" href="index.php">Home</a><a class="text-sm text-text-light/80 dark:text-text-dark/80 hover:text-primary transition-colors" href="#" onclick="alert('Please log in to use the Diagnosis Tool'); return false;">Diagnosis Tool</a><a class="text-sm text-text-light/80 dark:text-text-dark/80 hover:text-primary transition-colors" href="#">Disease Database</a></div>
            <div class="flex flex-col gap-3"><h4 class="font-bold text-sm text-text-light dark:text-text-dark">Company</h4><a class="text-sm text-text-light/80 dark:text-text-dark/80 hover:text-primary transition-colors" href="#">About Us</a><a class="text-sm text-text-light/80 dark:text-text-dark/80 hover:text-primary transition-colors" href="#">Blog</a><a class="text-sm text-text-light/80 dark:text-text-dark/80 hover:text-primary transition-colors" href="auth.php#footer">Contact</a></div>
            <div class="flex flex-col gap-3"><h4 class="font-bold text-sm text-text-light dark:text-text-dark">Legal</h4><a class="text-sm text-text-light/80 dark:text-text-dark/80 hover:text-primary transition-colors" href="#">Privacy Policy</a><a class="text-sm text-text-light/80 dark:text-text-dark/80 hover:text-primary transition-colors" href="#">Terms of Service</a></div>
        </div>
        <div class="mt-10 pt-6 border-t border-border-light dark:border-border-dark text-center text-sm text-text-light/60 dark:text-text-dark/60">
            © 2024 Plant AI. All rights reserved.
        </div>
    </footer>
    </div>
    </div>
</main>

<script>
    const landingContent = document.getElementById('landing-content');
    const authFormsView = document.getElementById('auth-forms-view');
    const diagnoseButton = document.getElementById('trigger-auth-button');
    const backButton = document.getElementById('back-to-landing-button-auth');

    // Function to switch to the Auth Forms view
    function switchToAuthForms() {
        if (landingContent) landingContent.style.display = 'none';
        if (authFormsView) authFormsView.style.display = 'flex';
    }
    
    // Alias for header button
    function showLoginForm() {
        switchToAuthForms();
    }

    if (diagnoseButton) {
        diagnoseButton.addEventListener('click', switchToAuthForms);
    }

    if (backButton) {
        backButton.addEventListener('click', function() {
            if (landingContent) landingContent.style.display = 'block'; // Block display for the landing page
            if (authFormsView) authFormsView.style.display = 'none';
        });
    }
    
    // Initial State Check: If there's an error/success message, the auth forms should be visible on load.
    <?php if ($error || $success): ?>
        window.addEventListener('DOMContentLoaded', switchToAuthForms);
    <?php endif; ?>

    // Mobile Menu Toggle
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
        if (mobileMenu && !e.target.closest('.header-navigation')) {
            mobileMenu.classList.remove('open');
        }
    });
</script>
</body>
</html>
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
?>
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
            <a href="predict_wheat.php" class="text-text-light dark:text-text-dark font-bold hover:text-primary transition-colors">Wheat</a>
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
            <a href="predict_plant.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Diagnosis</a>
            <a href="predict_wheat.php" class="px-4 py-2 text-sm font-bold hover:bg-primary/10 rounded transition-colors">Wheat</a>
            <a href="recommend_crop.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 bg-primary/20 rounded text-sm font-bold hover:bg-primary/30 transition-colors">Logout</a>
        </div>
    </div>
</header>

<div class="flex flex-1 justify-center py-10 sm:py-16">
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
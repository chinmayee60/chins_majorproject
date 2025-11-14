<?php
session_start();
require_once 'config.php';

// Check if user is already logged in
if (isset($_SESSION['user_id'])) {
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

// Logic to determine initial view state: If error/success exists, show forms.
$show_forms = false;
if (isset($_POST['login']) || isset($_POST['register']) || $error || $success) {
    $show_forms = true;
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
</style>
<script src="//unpkg.com/alpinejs" defer></script> 
</head>
<body class="bg-background-light dark:bg-background-dark font-display text-text-light dark:text-text-dark">
<div class="relative flex min-h-screen w-full flex-col group/design-root overflow-x-hidden">
<div class="layout-container flex h-full grow flex-col">
<div class="flex flex-1 justify-center py-5 sm:px-10 md:px-20 lg:px-40">
<div class="layout-content-container flex flex-col w-full max-w-[960px] flex-1">
    
    <header class="flex items-center justify-between whitespace-nowrap border-b border-solid border-border-light dark:border-border-dark px-4 md:px-10 py-3">
        <div class="flex items-center gap-4">
            <div class="size-6 text-primary">
                <svg fill="none" viewbox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><g clip-path="url(#clip0_6_535)"><path clip-rule="evenodd" d="M47.2426 24L24 47.2426L0.757355 24L24 0.757355L47.2426 24ZM12.2426 21H35.7574L24 9.24264L12.2426 21Z" fill="currentColor" fill-rule="evenodd"></path></g></svg>
            </div>
            <h2 class="text-text-light dark:text-text-dark text-lg font-bold leading-tight tracking-[-0.015em]">Plant AI</h2>
        </div>
        <div class="hidden md:flex flex-1 justify-end gap-8">
            <div class="flex items-center gap-9">
                <a class="text-text-light dark:text-text-dark text-sm font-medium leading-normal hover:text-primary transition-colors" href="auth.php" id="nav-home">Home</a>
                <a class="text-text-light dark:text-text-dark text-sm font-medium leading-normal hover:text-primary transition-colors" href="#" id="nav-tool" onclick="alert('Please log in to access the Diagnosis Tool'); return false;">Diagnosis Tool</a>
                <a class="text-text-light dark:text-text-dark text-sm font-medium leading-normal hover:text-primary transition-colors" href="#" id="nav-database" onclick="alert('Please log in to access the Database'); return false;">Disease Database</a>
                <a class="text-text-light dark:text-text-dark text-sm font-medium leading-normal hover:text-primary transition-colors" href="#footer">Contact</a>
            </div>
        </div>
        <div class="md:hidden">
            <button class="flex max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-10 bg-primary/20 text-text-light dark:bg-primary/30 dark:text-text-dark hover:bg-primary/30 dark:hover:bg-primary/40 transition-colors gap-2 text-sm font-bold leading-normal tracking-[0.015em] min-w-0 px-2.5">
                <span class="material-symbols-outlined text-text-light dark:text-text-dark" style="font-size: 20px;">menu</span>
            </button>
        </div>
    </header>

    <main class="flex flex-col gap-16 md:gap-24 py-10 md:py-16 flex-1">

        <div id="landing-content" style="<?php echo $show_forms ? 'display: none;' : ''; ?>">
            
            <section id="hero-section" class="@container">
                <div class="flex flex-col gap-6 px-4 py-10 @[864px]:flex-row @[864px]:items-center">
                    <div class="flex flex-col gap-6 @[480px]:gap-8 @[864px]:w-1/2 @[864px]:justify-center">
                        <div class="flex flex-col gap-2 text-left">
                            <h1 class="text-text-light dark:text-text-dark text-4xl font-black leading-tight tracking-[-0.033em] @[480px]:text-5xl">Instant Disease Prediction for Your Plants</h1>
                            <h2 class="text-text-light/80 dark:text-text-dark/80 text-base font-normal leading-normal @[480px]:text-lg">Upload a leaf image to get an accurate diagnosis, care instructions, and prevention tips in seconds.</h2>
                        </div>
                        <button id="trigger-auth-button" class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-12 px-5 bg-primary text-text-light text-base font-bold leading-normal tracking-[0.015em] hover:opacity-90 transition-opacity">
                            <span class="truncate">Upload Image & Diagnose Now</span>
                        </button>
                    </div>
                    <div class="w-full @[864px]:w-1/2 h-64 @[480px]:h-80 @[864px]:h-96 bg-cover-placeholder bg-center bg-no-repeat bg-cover rounded-xl" data-alt="Abstract graphic of green digital plant leaves with data points" style='background-image: url("https://lh3.googleusercontent.com/aida-public/AB6AXuC8EW-cAHjEZXUuKMgZOGKRuP1ecKjk7LoqApPNORW4-9hA0qclo0BoTLbiZ_lhVLUQoYKi0BOL6bLpbhY0MQrSfjuAPy4g3G1V4Pcyp9UZnbkTYMm_dfRZlMYp1Fk0spsv3A2a7v-G5Unt2-MbZsRzw6gte4BTN0WAGfsk6Pi7jNk2Dg5DhvKPWtqzZmawsMJYLcmcpFU_yDm4YoOO_ItKYnNrVI11frt7ff0eUNBu8oHQRZBeU5zQ4RkiSJU27xs1lAk7RDpPijk7");'></div>
                </div>
            </section>
            
            <section class="flex flex-col gap-4 mt-10">
                <h4 class="text-primary text-sm font-bold leading-normal tracking-[0.015em] px-4 text-center">Common Diseases & Pests</h4>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 p-4">
                    <div class="flex flex-1 flex-col gap-3 rounded-xl border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-4 text-center items-center">
                        <span class="material-symbols-outlined text-primary" style="font-size: 28px;">compost</span>
                        <div class="flex flex-col gap-1"><h2 class="text-text-light dark:text-text-dark text-base font-bold leading-tight">Fungal Infections</h2></div>
                        <p class="text-text-light/70 dark:text-text-dark/70 text-sm font-normal leading-normal">Identify and treat common fungal issues.</p>
                    </div>
                    <div class="flex flex-1 flex-col gap-3 rounded-xl border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-4 text-center items-center">
                        <span class="material-symbols-outlined text-primary" style="font-size: 28px;">bug_report</span>
                        <div class="flex flex-col gap-1"><h2 class="text-text-light dark:text-text-dark text-base font-bold leading-tight">Pest Infestations</h2></div>
                        <p class="text-text-light/70 dark:text-text-dark/70 text-sm font-normal leading-normal">Detect pests before they cause damage.</p>
                    </div>
                    <div class="flex flex-1 flex-col gap-3 rounded-xl border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-4 text-center items-center">
                        <span class="material-symbols-outlined text-primary" style="font-size: 28px;">coronavirus</span>
                        <div class="flex flex-col gap-1"><h2 class="text-text-light dark:text-text-dark text-base font-bold leading-tight">Viral Diseases</h2></div>
                        <p class="text-text-light/70 dark:text-text-dark/70 text-sm font-normal leading-normal">Learn about viral threats to your plants.</p>
                    </div>
                    <div class="flex flex-1 flex-col gap-3 rounded-xl border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-4 text-center items-center">
                        <span class="material-symbols-outlined text-primary" style="font-size: 28px;">science</span>
                        <div class="flex flex-col gap-1"><h2 class="text-text-light dark:text-text-dark text-base font-bold leading-tight">Nutrient Deficiencies</h2></div>
                        <p class="text-text-light/70 dark:text-text-dark/70 text-sm font-normal leading-normal">Diagnose and correct nutritional imbalances.</p>
                    </div>
                </div>
            </section>
            
            <section><h2 class="text-text-light dark:text-text-dark tracking-tight text-2xl md:text-3xl font-bold leading-tight px-4 text-center pb-3 pt-5 max-w-3xl mx-auto">"Knowledge and early action are the best defense against crop failure and plant loss."</h2></section>

            <section class="flex flex-col gap-6 px-4">
                <div class="flex flex-col gap-1 text-left">
                    <h3 class="text-primary text-sm font-bold leading-normal tracking-[0.015em]">Featured Solutions</h3>
                    <h2 class="text-text-light dark:text-text-dark text-3xl font-bold leading-tight tracking-tight">Identify Common Problems</h2>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="flex flex-col overflow-hidden rounded-xl border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark group">
                        <div class="w-full h-48 bg-cover bg-center" data-alt="A close-up of a plant leaf with white powdery mildew." style="background-image: url('https://lh3.googleusercontent.com/aida-public/AB6AXuAitbTVCzOrApMATo_2UHYvB1Cue3e9I2qPVbuPrcSayhd6kKHSuDBnThjYnA4Cfk7ekKkjeKfv4TMXNo-gCMsn1PfCMkmtElbHU3C5k0ibApmdzEEbO259WQqPrydQO462L3gLDVGmYsEDXnfacW66u_1Kkn0H8Axhqw_XodwOF8sqT10rADHewW9lddUjzKsc2pphZStA2gQvE1TdCkBdBAPdASJLMkaQ6eYOwf3BCSaC26NQmvmI1L0TJudz-5E5q_WBPBUl8_ZQ')"></div>
                        <div class="p-5 flex flex-col gap-2">
                            <h4 class="text-text-light dark:text-text-dark text-lg font-bold">Powdery Mildew</h4>
                            <p class="text-text-light/70 dark:text-text-dark/70 text-sm">A common fungal disease affecting a wide variety of plants.</p>
                        </div>
                    </div>
                    <div class="flex flex-col overflow-hidden rounded-xl border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark group">
                        <div class="w-full h-48 bg-cover bg-center" data-alt="A tomato plant leaf showing signs of late blight." style="background-image: url('https://lh3.googleusercontent.com/aida-public/AB6AXuCbsKbPYK_AmOgYWkjnBE3MVHGohDnjjbTVRYqeFJx--qXRoHig28SlaRwEQ-SsYl7Mqt8O_MwNi8J9ENtFQy-stYSALZUHpyBZAh1sTOMem28DUm4ENpm9g_M_RU24F54wQfjL08MGmM6jVm_VgRILdJ4RuDHPyTvJnn9UbBKamXGdGUVPffSzwEhuPF2BStvk1fbH1KNF7jSjhmH9AwnUhirqccjP7fMrWdQYKyVGw9TKiX4Fjb0yYfs_BO3KxCzagDuq6KRiPGzu')"></div>
                        <div class="p-5 flex flex-col gap-2">
                            <h4 class="text-text-light dark:text-text-dark text-lg font-bold">Tomato Blight</h4>
                            <p class="text-text-light/70 dark:text-text-dark/70 text-sm">Learn to spot the early signs of late blight on tomato plants.</p>
                        </div>
                    </div>
                    <div class="flex flex-col overflow-hidden rounded-xl border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark group">
                        <div class="w-full h-48 bg-cover bg-center" data-alt="A zoomed-in image of spider mites on a green leaf." style="background-image: url('https://lh3.googleusercontent.com/aida-public/AB6AXuAuW2sjvggc_1N-I0DowconngESi0PO4lcfv03l81rdoDKHAAWz41m8vTlUXpVDTUeuIlxK5lHlaRujli6D8cFl3YPVSrEFBKO1kAmZU1yZ9MxEp6VWrIsGJ4AJaRDO34JOseq3xsWPKlTrf8x9Q_ZVG3K0dYrEl3VPv_luQrVagnMAU6xRvRRT3qHCdIH__zTyLVgUpg1Y_kPmgMSc32IPvbxgYOkZzqd1ZOtXEMahNQne1mnHsyCSV011jt7BGN4N7DEkHRZ9JeaB')"></div>
                        <div class="p-5 flex flex-col gap-2">
                            <h4 class="text-text-light dark:text-text-dark text-lg font-bold">Spider Mites</h4>
                            <p class="text-text-light/70 dark:text-text-dark/70 text-sm">Tiny but destructive pests that can quickly damage your plants.</p>
                        </div>
                    </div>
                </div>
            </section>

            <section class="flex flex-col gap-6 px-4">
                <div class="flex flex-col gap-1 text-left">
                    <h3 class="text-primary text-sm font-bold leading-normal tracking-[0.015em]">Learn More</h3>
                    <h2 class="text-text-light dark:text-text-dark text-3xl font-bold leading-tight tracking-tight">From Our Blog</h2>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div class="flex flex-col sm:flex-row gap-5 items-center group">
                        <div class="w-full sm:w-1/3 aspect-square rounded-xl bg-cover bg-center flex-shrink-0" data-alt="Healthy green plant leaves in a garden setting." style="background-image: url('https://lh3.googleusercontent.com/aida-public/AB6AXuC9if1youFqPuKbPEd9V8VInhR-uZpgArWbH8jvIZWu-bBXZno-3y-y8Gb3R6NGJeRTUZBXre0wSwfusEjNllk6KQrWIaapSfVQUXNippYqFZYoQF4pk48AQhjPkIJ7L8SfFlsTHgha8yUjua40ZzYEypkHW4bt5SCf_jkLePDOUH0I0pIVYAhVn3jiuiweNR7YhgW5WOEPdoUPJlQsqe0Pq1_PnsGq3S_sosXThsnqjv9O1BMfROBT7-9L_kBjwqyR1Ucsb3C8dUVJ')"></div>
                        <div class="flex flex-col gap-2">
                            <h4 class="text-text-light dark:text-text-dark text-lg font-bold group-hover:text-primary transition-colors">5 Tips for Preventing Fungal Growth</h4>
                            <p class="text-text-light/70 dark:text-text-dark/70 text-sm">Proactive steps you can take to keep your garden healthy and fungus-free.</p>
                            <a class="text-primary font-bold text-sm mt-1" href="#">Read More →</a>
                        </div>
                    </div>
                    <div class="flex flex-col sm:flex-row gap-5 items-center group">
                        <div class="w-full sm:w-1/3 aspect-square rounded-xl bg-cover bg-center flex-shrink-0" data-alt="A person inspecting a leaf for pests with a magnifying glass." style="background-image: url('https://lh3.googleusercontent.com/aida-public/AB6AXuDBrGw06MSwKkRB2HPGOYKRSnqimQx0x-bew5KRVzBg5EpWx7f2ayWXjZVUjPWmcC4iDQu4awrpX4BWNhPWhISZLKa3vKg2E6fGpxKamwA5AV75BmLNWWFJtLUwUrBXnad7avJ5UwStKumKaNnbNqIIdMOracxjwKekDeaYhb-hIkVJcxS-j6O1asqOU1IpVYNSaDuZqaJWq2rYkEJWBAP7TZIccXd86nibTdVBs8bXxvI5c6oMPh1WyfGYdas5azRleesKOX91lvRx')"></div>
                        <div class="flex flex-col gap-2">
                            <h4 class="text-text-light dark:text-text-dark text-lg font-bold group-hover:text-primary transition-colors">Identifying Common Garden Pests</h4>
                            <p class="text-text-light/70 dark:text-text-dark/70 text-sm">A visual guide to identifying the most common pests in your garden.</p>
                            <a class="text-primary font-bold text-sm mt-1" href="#">Read More →</a>
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
        </main>
    
    <footer id="footer" class="mt-16 border-t border-border-light dark:border-border-dark py-10 px-4">
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
</div>

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
</script>
</body>
</html>
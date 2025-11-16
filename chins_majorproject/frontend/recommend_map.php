<?php
session_start();
require_once 'config.php';
if (!isset($_SESSION['user_id'])) { header("Location: auth.php"); exit(); }
$username = $_SESSION['username'];
?>

<!DOCTYPE html>
<html lang="en">
<?php include 'includes/global_head_scripts.php'; ?>
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Map - Crop Recommendation</title>
    
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet"/>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
    <script>
        // NOTE: Including the tailwind config here for completeness, though it's assumed to be loaded globally.
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

        /* Map-Specific Styles */
        .map-container {
            width: 100%;
            height: calc(100vh - 280px); 
            min-height: 500px;
            border-radius: 1.5rem;
            overflow: hidden;
            box-shadow: 0 20px 50px rgba(0,0,0,0.15);
            border: 2px solid #cee8d7;
        }
        .dark .map-container {
            border-color: #2a4a35;
            box-shadow: 0 20px 50px rgba(0,0,0,0.4);
        }
        #map { 
            height: 100%; 
            width: 100%; 
        }
        .info-card { 
            padding: 2rem;
            background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
            border: 2px solid #cee8d7;
            border-radius: 1.5rem;
            margin-top: 1.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .dark .info-card {
            background: linear-gradient(135deg, #1a2e21 0%, #102216 100%);
            border-color: #2a4a35;
        }
        
        /* Custom Leaflet Popup Styling */
        .leaflet-popup-content-wrapper {
            border-radius: 1rem;
            padding: 0;
        }
        .leaflet-popup-content {
            margin: 1rem;
            font-family: Inter, sans-serif;
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
            <h2 class="text-lg font-bold">AgriVision AI</h2>
        </div>
        
        <nav class="hidden md:flex items-center gap-6 text-sm font-medium">
            <a href="auth.php?landing=1" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Home</a>
            <a href="index.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Dashboard</a>
            <a href="predict_plant.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="text-text-light dark:text-text-dark hover:text-primary transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="text-text-light dark:text-text-dark font-bold hover:text-primary transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 rounded-lg bg-primary/20 text-text-light hover:bg-primary/30 transition-colors font-bold">Logout</a>
        </nav>
        
        <button id="mobileMenuToggle" class="md:hidden">
            <span class="material-symbols-outlined" style="font-size: 24px;">menu</span>
        </button>
        
        <div id="mobileMenu" class="md:hidden">
            <a href="auth.php?landing=1" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Home</a>
            <a href="index.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Dashboard</a>
            <a href="predict_plant.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Diagnosis</a>
            <a href="recommend_crop.php" class="px-4 py-2 text-sm font-medium hover:bg-primary/10 rounded transition-colors">Crop Tool</a>
            <a href="recommend_map.php" class="px-4 py-2 text-sm font-bold hover:bg-primary/10 rounded transition-colors">Map</a>
            <a href="index.php?logout=true" class="px-4 py-2 bg-primary/20 rounded text-sm font-bold hover:bg-primary/30 transition-colors">Logout</a>
        </div>
    </div>
</header>

<div class="flex flex-1 justify-center py-8">
    <div class="w-full max-w-7xl px-4 sm:px-6 lg:px-8">
        
        <!-- Professional Header -->
        <div class="flex flex-col items-center gap-3 mb-6 text-center">
            <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-blue-100 to-cyan-100 dark:from-blue-900/30 dark:to-cyan-900/30 text-blue-700 dark:text-blue-400 text-sm font-semibold mb-2">
                <span class="material-symbols-outlined" style="font-size: 18px;">public</span>
                Real-Time Weather Analysis
            </div>
            <h1 class="text-4xl md:text-5xl font-black tracking-tight text-text-light dark:text-text-dark">
                Interactive Crop Map
            </h1>
            <p class="max-w-2xl text-base md:text-lg text-subtle-light dark:text-subtle-dark">
                Click anywhere on the map to get live weather data and personalized crop recommendations for that location
            </p>
        </div>
        
        <!-- Map Container with Professional Styling -->
        <div class="map-container">
            <div id="map"></div>
        </div>
        
        <!-- Info Card with Modern Design -->
        <div id="info" class="info-card text-text-light dark:text-text-dark" style="display: none;"></div>
    </div>
</div>

<script>
    // --- MAP INITIALIZATION AND LOGIC ---
    
    // Initialize map
    const map = L.map('map').setView([20.5937, 78.9629], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>',
    }).addTo(map);

    // Call invalidateSize after a slight delay to ensure the map container is fully rendered
    // This is the CRITICAL fix for gray tiles in dynamic layouts.
    setTimeout(() => {
        map.invalidateSize();
    }, 200);


    // Crop suitability ranges (simplified for temperature & humidity only)
    const cropRanges = {
      Wheat: { temp: [15, 25], hum: [40, 70] },
      Rice: { temp: [20, 35], hum: [70, 90] },
      Maize: { temp: [18, 27], hum: [50, 80] },
      Soybean: { temp: [20, 30], hum: [50, 75] },
      Cotton: { temp: [20, 30], hum: [50, 80] },
      Chickpea: { temp: [15, 25], hum: [30, 50] },
      Sugarcane: { temp: [21, 27], hum: [50, 70] },
      Groundnut: { temp: [20, 30], hum: [40, 60] },
      Mustard: { temp: [10, 25], hum: [40, 60] },
      Sorghum: { temp: [25, 35], hum: [40, 60] },
      PearlMillet: { temp: [25, 35], hum: [40, 60] },
      PigeonPea: { temp: [20, 30], hum: [40, 70] },
      Lentil: { temp: [20, 30], hum: [50, 70] },
      MungBean: { temp: [25, 35], hum: [60, 80] },
      BlackGram: { temp: [25, 35], hum: [60, 80] },
      Jute: { temp: [22, 27], hum: [70, 90] },
      Potato: { temp: [15, 20], hum: [70, 80] },
      Onion: { temp: [13, 25], hum: [50, 70] },
      Tomato: { temp: [18, 27], hum: [50, 80] },
      Mango: { temp: [20, 35], hum: [50, 70] },
      Banana: { temp: [20, 35], hum: [70, 85] },
      Coconut: { temp: [20, 35], hum: [70, 90] },
    };

    // Crop scoring (based only on temp & humidity)
    function scoreCrop(crop, temp, hum) {
      const ranges = cropRanges[crop];
      let score = 0;

      // Score based on proximity to temperature range center
      const tempCenter = (ranges.temp[0] + ranges.temp[1]) / 2;
      const tempRange = ranges.temp[1] - ranges.temp[0];
      const tempDeviation = Math.abs(temp - tempCenter);
      score += 50 * Math.max(0, 1 - (tempDeviation / (tempRange / 2)) * 0.5); // Max 50 points

      // Score based on proximity to humidity range center
      const humCenter = (ranges.hum[0] + ranges.hum[1]) / 2;
      const humRange = ranges.hum[1] - ranges.hum[0];
      const humDeviation = Math.abs(hum - humCenter);
      score += 50 * Math.max(0, 1 - (humDeviation / (humRange / 2)) * 0.5); // Max 50 points

      return Math.round(Math.min(100, score));
    }

    // Check if the clicked location is on land
    async function isLand(lat, lon) {
      try {
        const url = `https://is-on-water.balbona.me/api/v1/get/${lat}/${lon}`;
        const response = await fetch(url);
        const data = await response.json();
        return data.feature === "LAND";
      } catch {
        return true; // default to land if API fails
      }
    }

    // Fetch live weather data
    async function fetchWeather(lat, lon) {
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,precipitation_probability&timezone=auto`;
      const response = await fetch(url);
      const data = await response.json();
      return {
        temp: data.current.temperature_2m,
        hum: data.current.relative_humidity_2m,
        rainProb: data.current.precipitation_probability, 
      };
    }

    // On map click
    let loadingMarker = null;
    map.on("click", async (e) => {
      const lat = e.latlng.lat;
      const lon = e.latlng.lng;

      if (loadingMarker) map.removeLayer(loadingMarker);
      loadingMarker = L.marker([lat, lon]).addTo(map);
      loadingMarker.bindPopup("Fetching live weather data...").openPopup();

      try {
        const land = await isLand(lat, lon);
        if (!land) {
          const msg = "❌ This location is over water — not suitable for crops.";
          loadingMarker.setPopupContent(msg);
          document.getElementById("info").innerHTML = msg;
          document.getElementById("info").style.display = "block";
          return;
        }

        const { temp, hum, rainProb } = await fetchWeather(lat, lon);

        // Score all crops
        const recommendations = Object.keys(cropRanges)
          .map((crop) => ({
            crop,
            score: scoreCrop(crop, temp, hum),
          }))
          .sort((a, b) => b.score - a.score)
          .slice(0, 5);

        let popupContent = `
          <div style="font-family: Inter, sans-serif; min-width: 280px;">
            <div style="background: linear-gradient(135deg, #0df259 0%, #0bc047 100%); color: white; padding: 12px; margin: -16px -16px 12px -16px; border-radius: 12px 12px 0 0;">
              <div style="font-size: 14px; font-weight: 700; opacity: 0.9; margin-bottom: 4px;">📍 LOCATION</div>
              <div style="font-size: 12px; opacity: 0.8;">${lat.toFixed(4)}, ${lon.toFixed(4)}</div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 12px;">
              <div style="background: #f0fdf4; padding: 8px; border-radius: 8px; text-align: center;">
                <div style="font-size: 20px;">🌡️</div>
                <div style="font-size: 18px; font-weight: 900; color: #0d1c12;">${temp}°C</div>
                <div style="font-size: 10px; color: #499c65; font-weight: 600;">TEMP</div>
              </div>
              <div style="background: #f0fdf4; padding: 8px; border-radius: 8px; text-align: center;">
                <div style="font-size: 20px;">💧</div>
                <div style="font-size: 18px; font-weight: 900; color: #0d1c12;">${hum}%</div>
                <div style="font-size: 10px; color: #499c65; font-weight: 600;">HUMIDITY</div>
              </div>
              <div style="background: #f0fdf4; padding: 8px; border-radius: 8px; text-align: center;">
                <div style="font-size: 20px;">🌧️</div>
                <div style="font-size: 18px; font-weight: 900; color: #0d1c12;">${rainProb}%</div>
                <div style="font-size: 10px; color: #499c65; font-weight: 600;">RAIN</div>
              </div>
            </div>
            
            <div style="border-top: 2px solid #cee8d7; padding-top: 12px;">
              <div style="font-weight: 800; color: #0df259; font-size: 13px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 18px;">🌾</span> TOP CROP RECOMMENDATIONS
              </div>
              <div style="display: flex; flex-direction: column; gap: 6px;">`;
        recommendations.forEach((rec, index) => {
          const barColor = rec.score > 80 ? '#0df259' : rec.score > 60 ? '#fbbf24' : '#f97316';
          popupContent += `
            <div style="display: flex; align-items: center; gap: 8px;">
              <div style="min-width: 20px; font-weight: 900; color: #0d1c12; font-size: 14px;">${index + 1}.</div>
              <div style="flex: 1;">
                <div style="font-weight: 700; color: #0d1c12; font-size: 13px; margin-bottom: 2px;">${rec.crop}</div>
                <div style="background: #e5e7eb; border-radius: 4px; height: 6px; overflow: hidden;">
                  <div style="background: ${barColor}; height: 100%; width: ${rec.score}%; transition: width 0.3s;"></div>
                </div>
              </div>
              <div style="min-width: 45px; text-align: right; font-weight: 900; color: ${barColor}; font-size: 13px;">${rec.score}%</div>
            </div>`;
        });
        popupContent += `
              </div>
            </div>
          </div>`;

        loadingMarker.setPopupContent(popupContent);
        
        // Update info card with professional styling
        const infoHTML = `
          <div class="space-y-6">
            <div>
              <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/20 text-primary text-xs font-bold mb-3">
                <span class="material-symbols-outlined" style="font-size: 16px;">location_on</span>
                ${lat.toFixed(4)}, ${lon.toFixed(4)}
              </div>
              <h3 class="text-2xl font-black text-text-light dark:text-text-dark">Location Analysis</h3>
            </div>
            
            <div class="grid grid-cols-3 gap-4">
              <div class="p-4 rounded-xl bg-gradient-to-br from-orange-50 to-white dark:from-orange-900/10 dark:to-transparent border-2 border-orange-200 dark:border-orange-800/30">
                <div class="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-orange-400 to-red-500 text-white mb-3 mx-auto">
                  <span class="material-symbols-outlined" style="font-size: 28px;">thermostat</span>
                </div>
                <div class="text-3xl font-black text-center text-text-light dark:text-text-dark">${temp}°C</div>
                <div class="text-xs font-semibold text-center text-subtle-light dark:text-subtle-dark mt-1">Temperature</div>
              </div>
              
              <div class="p-4 rounded-xl bg-gradient-to-br from-blue-50 to-white dark:from-blue-900/10 dark:to-transparent border-2 border-blue-200 dark:border-blue-800/30">
                <div class="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-blue-400 to-cyan-500 text-white mb-3 mx-auto">
                  <span class="material-symbols-outlined" style="font-size: 28px;">water_drop</span>
                </div>
                <div class="text-3xl font-black text-center text-text-light dark:text-text-dark">${hum}%</div>
                <div class="text-xs font-semibold text-center text-subtle-light dark:text-subtle-dark mt-1">Humidity</div>
              </div>
              
              <div class="p-4 rounded-xl bg-gradient-to-br from-sky-50 to-white dark:from-sky-900/10 dark:to-transparent border-2 border-sky-200 dark:border-sky-800/30">
                <div class="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-sky-400 to-blue-500 text-white mb-3 mx-auto">
                  <span class="material-symbols-outlined" style="font-size: 28px;">rainy</span>
                </div>
                <div class="text-3xl font-black text-center text-text-light dark:text-text-dark">${rainProb}%</div>
                <div class="text-xs font-semibold text-center text-subtle-light dark:text-subtle-dark mt-1">Rain Chance</div>
              </div>
            </div>
            
            <div class="pt-4 border-t-2 border-border-light dark:border-border-dark">
              <div class="flex items-center gap-2 mb-4">
                <div class="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-green-400 text-white">
                  <span class="material-symbols-outlined" style="font-size: 24px;">agriculture</span>
                </div>
                <h4 class="text-xl font-black text-text-light dark:text-text-dark">Recommended Crops</h4>
              </div>
              <div class="space-y-3">
                ${recommendations.map((rec, index) => {
                  const barColor = rec.score > 80 ? 'bg-primary' : rec.score > 60 ? 'bg-yellow-400' : 'bg-orange-500';
                  const textColor = rec.score > 80 ? 'text-primary' : rec.score > 60 ? 'text-yellow-600' : 'text-orange-600';
                  return `
                    <div class="p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700">
                      <div class="flex items-center justify-between mb-2">
                        <span class="font-bold text-text-light dark:text-text-dark">${index + 1}. ${rec.crop}</span>
                        <span class="font-black ${textColor}">${rec.score}%</span>
                      </div>
                      <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                        <div class="${barColor} h-full rounded-full transition-all duration-500" style="width: ${rec.score}%"></div>
                      </div>
                    </div>
                  `;
                }).join('')}
              </div>
            </div>
          </div>
        `;
        
        document.getElementById("info").innerHTML = infoHTML;
        document.getElementById("info").style.display = "block";
      } catch (error) {
        const msg = `⚠️ Error fetching data: ${error.message}`;
        loadingMarker.setPopupContent(msg);
        document.getElementById("info").innerHTML = msg;
        document.getElementById("info").style.display = "block";
      }
    });

    // Auto trigger once for demo on center of India
    map.fire("click", { latlng: { lat: 20.5937, lng: 78.9629 } });
</script>
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
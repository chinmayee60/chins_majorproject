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
        /* Fixed Sidebar Implementation */
        .fixed-sidebar-menu {
            position: fixed; top: 0; left: 0; width: 280px; height: 100vh;
            background-color: #f5f8f6; color: #0d1c12; z-index: 1030; padding-top: 20px; border-right: 1px solid #cee8d7;
            display: flex; flex-direction: column;
        }
        .main-content-area { 
            margin-left: 280px; 
            padding-top: 30px; 
            /* Enforce vertical space for the scrollable content */
            min-height: 100vh;
        }
        .fixed-sidebar-menu .nav-link { padding: 15px 1.5rem; color: #0d1c12 !important; font-weight: 500; }
        .fixed-sidebar-menu .nav-link.active {
            background-color: rgba(13, 242, 89, 0.2); color: #0d1c12 !important;
            border-left: 3px solid #0df259; font-weight: 700;
        }

        /* Map-Specific Styles (Ensuring 100% parent height) */
        .map-container {
            width: 100%;
            /* Using a calculated height to ensure it fills the remaining view space */
            height: calc(100vh - 200px); 
            min-height: 600px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        #map { 
            height: 100%; 
            width: 100%; 
            border-radius: 10px; 
        }
        .info { 
            padding: 15px; background: #fff; border: 1px solid #cee8d7;
            border-radius: 6px; margin-top: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
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
      <a class="nav-link" href="recommend_crop.php">📊 Crop Recommendation</a>
      <a class="nav-link active" href="recommend_map.php">🗺️ Map Recommendation</a>
    </nav>
    <div class="p-4 border-top border-border-light dark:border-border-dark">
        <p class="text-sm text-text-light/70 dark:text-text-dark/70 mb-1">
            Logged in as: <strong><?php echo htmlspecialchars($username); ?></strong>
        </p>
        <a href="index.php?logout=true" class="flex w-full items-center justify-center rounded-lg h-10 bg-primary/20 text-text-light dark:bg-primary/30 dark:text-text-dark hover:bg-primary/30 dark:hover:bg-primary/40 transition-colors text-sm font-bold">Logout</a>
    </div>
</div>

<div class="main-content-area flex flex-1 justify-center px-4 sm:px-6 lg:px-8">
    <div class="w-full max-w-7xl">
        <h1 class="text-4xl font-black tracking-tighter text-text-light dark:text-text-dark mb-4 mt-5">🗺️ Weather-Based Crop Recommendation</h1>
        <p class="text-base text-subtle-light dark:text-subtle-dark mb-4">Click anywhere on the map to get real-time weather and best crops for that region.</p>
        
        <div class="map-container">
            <div id="map"></div>
        </div>
        
        <div id="info" class="info text-text-light dark:text-text-dark" style="display: none;"></div>
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
          <div style="font-family: Inter, sans-serif;">
            <b>📍 Location:</b> (${lat.toFixed(4)}, ${lon.toFixed(4)})<br>
            <b>🌡️ Weather:</b> Temp ${temp}°C, Hum ${hum}%<br>
            <b>🌧️ Rain Chance:</b> ${rainProb}%<br><hr>
            <b style="color: #009472;">🌾 Top Crop Recommendations:</b><ul style="list-style-type: none; padding: 0;">`;
        recommendations.forEach((rec) => {
          popupContent += `<li style="font-weight: bold;">${rec.crop}: <span style="color: ${rec.score > 70 ? 'green' : 'orange'};">${rec.score}% match</span></li>`;
        });
        popupContent += "</ul></div>";

        loadingMarker.setPopupContent(popupContent);
        document.getElementById("info").innerHTML = popupContent;
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
</body>
</html>
<?php
// Database Configuration
define('DB_SERVER', 'localhost');
define('DB_USERNAME', 'root');
define('DB_PASSWORD', '');
define('DB_NAME', 'chin_major');

// Flask API Endpoints
define('FLASK_PLANT_URL', 'http://127.0.0.1:5000/predict_plant');
define('FLASK_WHEAT_URL', 'http://127.0.0.1:5001/predict_wheat');
define('FLASK_CROP_URL', 'http://127.0.0.1:5002/recommend_crop');

// --- GEMINI API Key Configuration ---
define('GEMINI_API_KEY', 'AIzaSyCdeyO8wmyeq31AS8URc6koM7sGQ_08DFo');

// --- Connect to Database ---
$conn = new mysqli(DB_SERVER, DB_USERNAME, DB_PASSWORD, DB_NAME);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Global function to get AI Advice using Gemini API (Final Robust Version)
function get_gemini_advice($prompt) {
    $apiKey = GEMINI_API_KEY;
    if (empty($apiKey) || $apiKey === 'YOUR_GEMINI_API_KEY_HERE') {
        return "AI Advice: Gemini API key is missing.";
    }

    $url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" . $apiKey;
    
    $data = [
        'contents' => [
            [
                'role' => 'user',
                'parts' => [
                    ['text' => $prompt]
                ]
            ]
        ],
        'generationConfig' => [ 
            'temperature' => 0.5,
            'maxOutputTokens' => 1500, // Ample budget for stable output
        ]
    ];

    $ch = curl_init($url);
    
    $headers = [
        'Content-Type: application/json',
    ];

    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    curl_setopt($ch, CURLOPT_POST, 1);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false); 

    $response = curl_exec($ch);
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curl_error = curl_error($ch);
    curl_close($ch);

    if ($curl_error) {
        return "AI Error (cURL): Could not connect to Gemini API. Error: " . $curl_error;
    }

    $response_data = json_decode($response, true);

    // 1. Check for hard HTTP errors
    if ($http_code !== 200 || isset($response_data['error'])) {
        $error_message = $response_data['error']['message'] ?? "Unknown API Error.";
        $error_code = $response_data['error']['code'] ?? $http_code;
        return "AI Error (Code: {$error_code}): {$error_message}";
    }

    // 2. Robust Text Extraction Logic
    $advice = null;

    if (isset($response_data['candidates'][0]['content']['parts'][0]['text'])) {
        $advice = $response_data['candidates'][0]['content']['parts'][0]['text'];
    } elseif (isset($response_data['text'])) {
        $advice = $response_data['text'];
    }

    // 3. Check Finish Reason (Safety/Incomplete)
    if (isset($response_data['candidates'][0]['finishReason'])) {
        $finishReason = $response_data['candidates'][0]['finishReason'];
        
        if (($finishReason === 'SAFETY' || $finishReason === 'RECITATION')) {
            return "AI Error: Response blocked by safety filters. (Reason: {$finishReason})";
        }
    }
    
    // 4. Final Check: If text extraction failed
    if (is_null($advice)) {
        return "AI Error: Model returned an UNUSABLE structure. (Final parsing attempt failed.)";
    }
    
    return $advice;
}
<?php
// Enable error reporting for debugging
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Start session to get user_id (assuming user is logged in)
session_start(); 
require 'config.php';

// --- Security and Initialization ---
$user_id = $_SESSION['user_id'] ?? 1; 
$model_type = $_POST['model_type'] ?? '';

if (empty($model_type) || !in_array($model_type, ['plant', 'wheat', 'crop'])) {
    die("Error: Invalid model type specified.");
}

$api_url = '';
$result_key = '';
$file_key = '';
$input_detail = '';
$post_fields = [];
$image_redirect_param = '';

// --- FIX: Initialize result path here, ensuring it's always defined ---
$result_filename = 'pred_' . time() . '_' . $user_id . '.txt';
$result_filepath = 'results/' . $result_filename;


// --- 1. Configure API Call Parameters ---
switch ($model_type) {
    case 'plant':
        $api_url = FLASK_PLANT_URL;
        $result_key = 'disease';
        $file_key = 'image';
        break;
    case 'wheat':
        $api_url = FLASK_WHEAT_URL;
        $result_key = 'disease';
        $file_key = 'image';
        break;
    case 'crop':
        $api_url = FLASK_CROP_URL;
        $result_key = 'crop_recommendation';
        $file_key = 'data'; // Placeholder key for data input
        break;
}

// --- 2. Prepare Data (Image or JSON) ---
$ch = curl_init($api_url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, 1);
curl_setopt($ch, CURLOPT_TIMEOUT, 30); // Set a timeout for the ML models

if ($model_type === 'plant' || $model_type === 'wheat') {
    // --- Image Prediction Setup (With Image Saving) ---
    if (!isset($_FILES[$file_key]) || $_FILES[$file_key]['error'] != UPLOAD_ERR_OK) {
        die("Error: File upload failed, or no file was selected.");
    }
    
    $filepath = $_FILES[$file_key]['tmp_name'];
    $filename = $_FILES[$file_key]['name'];

    // Generate unique filename and SAVE the uploaded file temporarily/permanently for result display
    $image_extension = pathinfo($filename, PATHINFO_EXTENSION);
    $saved_image_name = 'img_' . time() . '_' . $user_id . '.' . $image_extension;
    $saved_image_path = 'results/' . $saved_image_name;
    
    if (!move_uploaded_file($filepath, $saved_image_path)) {
        die("Error: Failed to save uploaded image file. Check permissions on the /results/ folder.");
    }

    // Create a CURL file object for multipart/form-data using the SAVED file path
    $cfile = curl_file_create($saved_image_path, $_FILES[$file_key]['type'], $saved_image_name);
    $post_fields = ['image' => $cfile];
    
    curl_setopt($ch, CURLOPT_POSTFIELDS, $post_fields);

    $input_detail = "Uploaded file: " . $filename;
    
    // Pass the saved image path for redirect (needed for display)
    $image_redirect_param = '&img=' . urlencode($saved_image_path);

} else {
    // --- Crop Recommendation (JSON) Setup ---
    $input_data = [
        'N' => (float)$_POST['N'],
        'P' => (float)$_POST['P'],
        'K' => (float)$_POST['K'],
        'temperature' => (float)$_POST['temperature'],
        'humidity' => (float)$_POST['humidity'],
        'pH' => (float)$_POST['pH'],
        'rainfall' => (float)$_POST['rainfall']
    ];
    
    // Set headers for JSON input
    curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($input_data));

    $input_detail = json_encode($input_data);
    $image_redirect_param = '';
}

// --- 3. Execute Prediction API Call ---
$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$curl_error = curl_error($ch);
curl_close($ch);

if ($curl_error) {
    die("Error connecting to Flask API: " . $curl_error);
}

$prediction_result = json_decode($response, true);

if ($http_code !== 200 || !isset($prediction_result[$result_key])) {
    $error_message = $prediction_result['error'] ?? "Unknown Flask API error. Check if Python servers are running (Ports 5000-5002).";
    
    // Clean up saved image if prediction failed
    if (isset($saved_image_path) && file_exists($saved_image_path)) {
        unlink($saved_image_path);
    }
    
    die("API Call Failed (Code: $http_code): " . $error_message);
}

// --- 4. Successful Prediction and AI Integration ---
$main_result = $prediction_result[$result_key];
$final_output = "Model Type: " . ucfirst($model_type) . "\n";
$final_output .= "Timestamp: " . date('Y-m-d H:i:s') . "\n";
$final_output .= "User ID: " . $user_id . "\n";
$final_output .= "------------------------------------------\n";

// Generate the specific prompt for Gemini
if ($model_type === 'crop') {
    $final_output .= "Recommended Crop: " . $main_result . "\n";
    $final_output .= "Input Conditions: " . $input_detail . "\n";
    
    // Structured prompt for crop recommendation
    $advice_prompt = "For the crop recommendation: {$main_result}, provide a single, concise sentence recommending the best cultivation focus (e.g., irrigation, pest management, soil preparation).";
} else {
    // Plant/Wheat Disease
    $confidence = $prediction_result['confidence'] ?? 'N/A';
    $final_output .= "Detected Disease: " . $main_result . "\n";
    $final_output .= "Confidence: " . $confidence . "\n";
    $final_output .= "Input Detail: " . $input_detail . "\n";
    $final_output .= "Image Path: " . $saved_image_path . "\n"; // Log saved image path
    
    // Structured prompt for cure and fertilizer output
    $advice_prompt = "For the plant disease: {$main_result}, provide a two-part answer. Part 1: Give the best chemical or organic cure strategy (1 sentence). Part 2: Recommend one specific type of fertilizer or nutrient (e.g., high nitrogen, potash, boron) needed to support recovery.";
}

// --- CALL GEMINI API (FUNCTIONALITY IS RESTORED) ---
$ai_advice = get_gemini_advice($advice_prompt); 
$final_output .= "\n--- AI Generated Advice ---\n" . $ai_advice;


// --- 5. Save Final Results and History ---

// 5.1 Save to .txt file
file_put_contents($result_filepath, $final_output);

// 5.2 Record in MySQL (We also save the input details for history)
$stmt = $conn->prepare("INSERT INTO predictions (user_id, model_type, result_file, input_data) VALUES (?, ?, ?, ?)");
$stmt->bind_param("isss", $user_id, $model_type, $result_filepath, $input_detail);
$stmt->execute();
$stmt->close();
$conn->close();

// --- 6. Redirect to Display Results on the appropriate prediction page ---

$redirect_page = "predict_" . $model_type . ".php";

if ($model_type === 'crop') {
    $redirect_page = "recommend_crop.php";
}

header("Location: " . $redirect_page . "?status=success&file=" . urlencode($result_filepath) . $image_redirect_param);
exit();
?>
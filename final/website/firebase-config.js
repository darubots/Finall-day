/**
 * =========================================================================
 * FIREBASE CONFIGURATION
 * =========================================================================
 * Konfigurasi resmi Firebase Realtime Database untuk Smart City & PLTS Monitoring.
 * =========================================================================
 */

window.FIREBASE_CONFIG = {
  apiKey: "AIzaSyBiVhNjZ7mBlZUI3_ModDujSxXKPPqm_js",
  authDomain: "skatelplus-smart-city.firebaseapp.com",
  databaseURL: "https://skatelplus-smart-city-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "skatelplus-smart-city",
  storageBucket: "skatelplus-smart-city.firebasestorage.app",
  messagingSenderId: "610042872625",
  appId: "1:610042872625:web:dd16f546b8e9bc6c532610",
  measurementId: "G-SWC4XYEH4R"
};

/**
 * Mendapatkan konfigurasi Firebase yang aktif
 * Mengutamakan konfigurasi yang disimpan di LocalStorage browser (jika ada),
 * lalu fallback ke konfigurasi default di atas.
 */
function getActiveFirebaseConfig() {
  try {
    const saved = localStorage.getItem('smart_city_firebase_config');
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed.apiKey && !parsed.apiKey.includes('YOUR_API_KEY')) {
        return parsed;
      }
    }
  } catch (e) {
    console.warn('Gagal membaca Firebase config dari localStorage:', e);
  }
  return window.FIREBASE_CONFIG;
}

/**
 * Cek apakah konfigurasi Firebase valid (bukan placeholder)
 */
function isFirebaseConfigured(config = getActiveFirebaseConfig()) {
  return (
    config &&
    config.apiKey &&
    !config.apiKey.includes('YOUR_API_KEY') &&
    config.databaseURL &&
    !config.databaseURL.includes('YOUR_PROJECT_ID')
  );
}

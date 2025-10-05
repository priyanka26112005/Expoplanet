import React, { useState, useEffect } from 'react';

export default function ExoplanetExplorer() {
  const [currentPage, setCurrentPage] = useState(1);
  const [formData, setFormData] = useState({
    koi_period: '',
    koi_duration: '',
    koi_depth: '',
    koi_prad: '',
    koi_teq: '',
    koi_steff: '',
    koi_srad: '',
    koi_smass: ''
  });
  const [results, setResults] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState('');

  const API_URL = 'http://127.0.0.1:5000';

  useEffect(() => {
    const starsContainer = document.getElementById('starsContainer');
    if (!starsContainer) return;

    const starSizes = ['small', 'medium', 'large'];
    
    for (let i = 0; i < 200; i++) {
      const star = document.createElement('div');
      star.className = `star ${starSizes[Math.floor(Math.random() * starSizes.length)]}`;
      star.style.top = Math.random() * 100 + '%';
      star.style.left = Math.random() * 100 + '%';
      star.style.animationDuration = (Math.random() * 3 + 2) + 's';
      star.style.animationDelay = Math.random() * 3 + 's';
      starsContainer.appendChild(star);
    }

    for (let i = 0; i < 30; i++) {
      const particle = document.createElement('div');
      particle.className = 'particle';
      particle.style.top = Math.random() * 100 + '%';
      particle.style.left = Math.random() * 100 + '%';
      particle.style.animationDuration = (Math.random() * 4 + 4) + 's';
      particle.style.animationDelay = Math.random() * 4 + 's';
      starsContainer.appendChild(particle);
    }

    const shootingStarInterval = setInterval(() => {
      const shootingStar = document.createElement('div');
      shootingStar.className = 'shooting-star';
      shootingStar.style.top = Math.random() * 50 + '%';
      shootingStar.style.left = Math.random() * 50 + '%';
      starsContainer.appendChild(shootingStar);
      setTimeout(() => shootingStar.remove(), 2000);
    }, 3000);

    return () => {
      clearInterval(shootingStarInterval);
      starsContainer.innerHTML = '';
    };
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      const mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
      const mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
      const stars = document.querySelectorAll('.star');
      
      stars.forEach((star, index) => {
        const speed = (index % 3 + 1) * 5;
        star.style.transform = `translate(${mouseX * speed}px, ${mouseY * speed}px)`;
      });
    };

    document.addEventListener('mousemove', handleMouseMove);
    return () => document.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useEffect(() => {
    if (currentPage === 2) {
      const messages = [
        'Analyzing orbital mechanics...',
        'Processing transit signals...',
        'Calculating confidence scores...',
        'Finalizing predictions...'
      ];
      
      let index = 0;
      setLoadingMessage(messages[0]);
      
      const interval = setInterval(() => {
        index++;
        if (index < messages.length) {
          setLoadingMessage(messages[index]);
        } else {
          clearInterval(interval);
        }
      }, 750);

      return () => clearInterval(interval);
    }
  }, [currentPage]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.key === 'e' || e.key === 'E') && currentPage === 3) {
        e.preventDefault();
        fillExampleData();
      }
      if (e.key === 'Escape') {
        setCurrentPage(1);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [currentPage]);

  const goToPage = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.id]: e.target.value
    });
  };

  const fillExampleData = () => {
    setFormData({
      koi_period: '10.5',
      koi_duration: '3.2',
      koi_depth: '1500',
      koi_prad: '2.3',
      koi_teq: '850',
      koi_steff: '5500',
      koi_srad: '1.1',
      koi_smass: '1.0'
    });
  };

  const submitPrediction = async () => {
    const invalidFields = [];
    for (const [key, value] of Object.entries(formData)) {
      if (!value || isNaN(parseFloat(value))) {
        invalidFields.push(key.replace('koi_', ''));
      }
    }

    if (invalidFields.length > 0) {
      alert('Please fill in all fields with valid numbers:\n\n' + invalidFields.join(', '));
      return;
    }

    const numericData = {};
    for (const [key, value] of Object.entries(formData)) {
      numericData[key] = parseFloat(value);
    }

    goToPage(2);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(numericData)
      });

      if (!response.ok) {
        throw new Error('Prediction failed: ' + response.statusText);
      }

      const result = await response.json();

      setTimeout(() => {
        setResults(result);
        goToPage(4);
      }, 3000);

    } catch (error) {
      console.error('Error:', error);
      setTimeout(() => {
        alert(`Error connecting to prediction server.\n\nMake sure Flask backend is running:\npython app.py\n\nServer should be at: ${API_URL}`);
        goToPage(3);
      }, 2000);
    }
  };

  const resetForm = () => {
    setFormData({
      koi_period: '',
      koi_duration: '',
      koi_depth: '',
      koi_prad: '',
      koi_teq: '',
      koi_steff: '',
      koi_srad: '',
      koi_smass: ''
    });
    setResults(null);
    goToPage(3);
  };

  return (
    <>
      <style>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          overflow: hidden;
          background: linear-gradient(135deg, #0a1929 0%, #1e3a5f 50%, #4a90e2 100%);
          color: white;
          height: 100vh;
        }

        #starsContainer {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: 0;
          pointer-events: none;
        }

        .star {
          position: absolute;
          background: white;
          border-radius: 50%;
          animation: twinkle 3s infinite;
        }

        .star.small { width: 2px; height: 2px; }
        .star.medium { width: 3px; height: 3px; }
        .star.large { width: 4px; height: 4px; }

        @keyframes twinkle {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }

        .particle {
          position: absolute;
          width: 6px;
          height: 6px;
          background: rgba(74, 144, 226, 0.4);
          border-radius: 50%;
          animation: float 8s infinite ease-in-out;
        }

        @keyframes float {
          0%, 100% { transform: translateY(0) translateX(0); }
          50% { transform: translateY(-50px) translateX(30px); }
        }

        .shooting-star {
          position: absolute;
          width: 2px;
          height: 2px;
          background: white;
          box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8);
          animation: shoot 2s linear;
        }

        @keyframes shoot {
          0% { transform: translateX(0) translateY(0); opacity: 1; }
          100% { transform: translateX(300px) translateY(300px); opacity: 0; }
        }

        .page {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          display: flex;
          justify-content: center;
          align-items: center;
          opacity: 0;
          visibility: hidden;
          transition: opacity 0.5s ease, visibility 0.5s ease;
          z-index: 1;
        }

        .page.active {
          opacity: 1;
          visibility: visible;
        }

        .content {
          max-width: 1200px;
          width: 90%;
          text-align: center;
          padding: 40px;
          background: rgba(10, 25, 41, 0.85);
          backdrop-filter: blur(10px);
          border-radius: 20px;
          border: 1px solid rgba(74, 144, 226, 0.3);
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
          z-index: 10;
          position: relative;
          max-height: 90vh;
          overflow-y: auto;
        }

        .logo-container {
          width: 200px;
          height: 200px;
          margin: 0 auto 30px;
          background: rgba(74, 144, 226, 0.1);
          border: 3px solid rgba(74, 144, 226, 0.5);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 0 30px rgba(74, 144, 226, 0.3);
        }

        .logo-placeholder {
          font-size: 5rem;
        }

        .title {
          font-size: 4rem;
          font-weight: 700;
          letter-spacing: 4px;
          margin-bottom: 20px;
          background: linear-gradient(to right, #4a90e2, #90caf9);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .subtitle {
          font-size: 1.5rem;
          margin-bottom: 20px;
          color: rgba(144, 202, 249, 0.9);
        }

        .description {
          font-size: 1.1rem;
          line-height: 1.8;
          margin-bottom: 40px;
          color: rgba(255, 255, 255, 0.8);
          max-width: 800px;
          margin-left: auto;
          margin-right: auto;
        }

        .btn {
          padding: 18px 50px;
          font-size: 1.2rem;
          font-weight: 600;
          background: linear-gradient(135deg, #1e3a5f, #4a90e2);
          border: none;
          border-radius: 50px;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 10px 30px rgba(74, 144, 226, 0.4);
          margin: 10px;
        }

        .btn:hover {
          transform: translateY(-3px);
          box-shadow: 0 15px 40px rgba(74, 144, 226, 0.6);
          background: linear-gradient(135deg, #2e4a6f, #5aa0f2);
        }

        .btn-secondary {
          padding: 15px 40px;
          font-size: 1rem;
          font-weight: 600;
          background: rgba(74, 144, 226, 0.2);
          border: 2px solid rgba(74, 144, 226, 0.4);
          border-radius: 50px;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          margin: 10px;
        }

        .btn-secondary:hover {
          background: rgba(74, 144, 226, 0.3);
          transform: translateY(-2px);
        }

        .features-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 30px;
          margin-top: 60px;
          display: none;
        }

        .feature-card {
          background: rgba(74, 144, 226, 0.1);
          padding: 30px;
          border-radius: 15px;
          border: 1px solid rgba(74, 144, 226, 0.3);
          transition: transform 0.3s ease;
        }

        .feature-card:hover {
          transform: translateY(-5px);
          background: rgba(74, 144, 226, 0.15);
        }

        .feature-icon {
          font-size: 3rem;
          margin-bottom: 15px;
        }

        .orbital-loader {
          position: relative;
          width: 150px;
          height: 150px;
          margin: 0 auto 40px;
        }

        .planet {
          position: absolute;
          width: 40px;
          height: 40px;
          background: linear-gradient(135deg, #1e3a5f, #4a90e2);
          border-radius: 50%;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          box-shadow: 0 0 30px rgba(74, 144, 226, 0.8);
        }

        .orbit {
          position: absolute;
          width: 150px;
          height: 150px;
          border: 3px solid rgba(74, 144, 226, 0.3);
          border-radius: 50%;
          border-top-color: #4a90e2;
          animation: spin 2s linear infinite;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .loading-title {
          font-size: 2rem;
          margin-bottom: 20px;
        }

        .loading-message {
          font-size: 1.2rem;
          color: rgba(144, 202, 249, 0.8);
        }

        .page-title {
          font-size: 2.5rem;
          margin-bottom: 15px;
        }

        .page-subtitle {
          font-size: 1.2rem;
          color: rgba(144, 202, 249, 0.8);
          margin-bottom: 40px;
        }

        .form-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 20px;
          margin-bottom: 40px;
          max-width: 900px;
          margin-left: auto;
          margin-right: auto;
        }

        .input-field {
          text-align: left;
        }

        .input-field label {
          display: block;
          font-size: 0.95rem;
          margin-bottom: 8px;
          color: rgba(144, 202, 249, 0.9);
          font-weight: 500;
        }

        .input-field input {
          width: 100%;
          padding: 12px 15px;
          font-size: 1rem;
          background: rgba(30, 58, 95, 0.5);
          border: 2px solid rgba(74, 144, 226, 0.3);
          border-radius: 10px;
          color: white;
          transition: all 0.3s ease;
        }

        .input-field input::placeholder {
          color: rgba(144, 202, 249, 0.5);
        }

        .input-field input:focus {
          outline: none;
          border-color: rgba(74, 144, 226, 0.6);
          background: rgba(30, 58, 95, 0.7);
        }

        .button-group {
          display: flex;
          justify-content: center;
          flex-wrap: wrap;
          gap: 15px;
        }

        .hint {
          margin-top: 20px;
          font-size: 0.95rem;
          color: rgba(144, 202, 249, 0.7);
        }

        .result-card {
          background: rgba(30, 58, 95, 0.5);
          padding: 30px;
          border-radius: 15px;
          margin-bottom: 30px;
          border: 1px solid rgba(74, 144, 226, 0.3);
          text-align: left;
        }

        .result-card h2 {
          font-size: 1.8rem;
          margin-bottom: 20px;
          text-align: center;
        }

        .status-badge {
          display: inline-block;
          padding: 20px 40px;
          font-size: 1.8rem;
          font-weight: 700;
          border-radius: 50px;
          margin: 0 auto;
          display: block;
          width: fit-content;
        }

        .status-badge.confirmed {
          background: linear-gradient(135deg, #48bb78, #38a169);
          box-shadow: 0 10px 30px rgba(72, 187, 120, 0.4);
        }

        .status-badge.candidate {
          background: linear-gradient(135deg, #f6ad55, #ed8936);
          box-shadow: 0 10px 30px rgba(246, 173, 85, 0.4);
        }

        .status-badge.false-positive {
          background: linear-gradient(135deg, #fc8181, #f56565);
          box-shadow: 0 10px 30px rgba(252, 129, 129, 0.4);
        }

        .confidence-item {
          margin-bottom: 25px;
        }

        .confidence-label {
          display: flex;
          justify-content: space-between;
          margin-bottom: 10px;
          font-size: 1.1rem;
        }

        .confidence-value {
          font-weight: 700;
        }

        .confidence-bar-container {
          width: 100%;
          height: 30px;
          background: rgba(30, 58, 95, 0.5);
          border-radius: 15px;
          overflow: hidden;
        }

        .confidence-bar {
          height: 100%;
          border-radius: 15px;
          transition: width 1s ease;
        }

        .confirmed-bar {
          background: linear-gradient(90deg, #48bb78, #38a169);
        }

        .candidate-bar {
          background: linear-gradient(90deg, #f6ad55, #ed8936);
        }

        .false-positive-bar {
          background: linear-gradient(90deg, #fc8181, #f56565);
        }

        .interpretation-text {
          font-size: 1.1rem;
          line-height: 1.8;
          color: rgba(255, 255, 255, 0.9);
        }

        @media (max-width: 768px) {
          .title { font-size: 2.5rem; }
          .form-grid { 
            grid-template-columns: 1fr;
            gap: 15px;
          }
          .features-grid { grid-template-columns: 1fr; }
          .content { padding: 20px; width: 95%; }
          .logo-container {
            width: 150px;
            height: 150px;
          }
          .logo-placeholder {
            font-size: 4rem;
          }
        }
      `}</style>

      <div id="starsContainer"></div>

      <div className={`page ${currentPage === 1 ? 'active' : ''}`}>
        <div className="content">
          <div className="logo-container">
            <img 
              src="/logo.ico"
              alt="Exoplanet Explorer Logo" 
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain'
              }}
            />
          </div>
          <h1 className="title">EXOPLANET EXPLORER</h1>
          <p className="subtitle">Discover Worlds Beyond Our Solar System</p>
          <p className="description">
            Harness the power of artificial intelligence to detect and classify exoplanets 
            from NASA's open-source datasets. Join us in the search for new worlds.
          </p>
          <button className="btn" onClick={() => goToPage(3)}>Begin Exploration</button>
        </div>
      </div>

      <div className={`page ${currentPage === 2 ? 'active' : ''}`}>
        <div className="content">
          <div className="orbital-loader">
            <div className="planet"></div>
            <div className="orbit"></div>
          </div>
          <h2 className="loading-title">Analyzing Data...</h2>
          <p className="loading-message">{loadingMessage}</p>
        </div>
      </div>

      <div className={`page ${currentPage === 3 ? 'active' : ''}`}>
        <div className="content">
          <h1 className="page-title">Enter Exoplanet Parameters</h1>
          <p className="page-subtitle">Input the orbital and stellar characteristics</p>

          <div className="form-grid">
            <div className="input-field">
              <label htmlFor="koi_period">Orbital Period (days)</label>
              <input 
                type="number" 
                id="koi_period" 
                step="0.01" 
                placeholder="e.g., 10.5"
                value={formData.koi_period}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-field">
              <label htmlFor="koi_duration">Transit Duration (hours)</label>
              <input 
                type="number" 
                id="koi_duration" 
                step="0.01" 
                placeholder="e.g., 3.2"
                value={formData.koi_duration}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-field">
              <label htmlFor="koi_depth">Transit Depth (ppm)</label>
              <input 
                type="number" 
                id="koi_depth" 
                step="0.01" 
                placeholder="e.g., 1500"
                value={formData.koi_depth}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-field">
              <label htmlFor="koi_prad">Planet Radius (Earth radii)</label>
              <input 
                type="number" 
                id="koi_prad" 
                step="0.01" 
                placeholder="e.g., 2.3"
                value={formData.koi_prad}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-field">
              <label htmlFor="koi_teq">Equilibrium Temperature (K)</label>
              <input 
                type="number" 
                id="koi_teq" 
                step="0.01" 
                placeholder="e.g., 850"
                value={formData.koi_teq}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-field">
              <label htmlFor="koi_steff">Stellar Temp (K)</label>
              <input 
                type="number" 
                id="koi_steff" 
                step="0.01" 
                placeholder="e.g., 5500"
                value={formData.koi_steff}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-field">
              <label htmlFor="koi_srad">Stellar Radius (Solar radii)</label>
              <input 
                type="number" 
                id="koi_srad" 
                step="0.01" 
                placeholder="e.g., 1.1"
                value={formData.koi_srad}
                onChange={handleInputChange}
              />
            </div>
            <div className="input-field">
              <label htmlFor="koi_smass">Stellar Mass (Solar masses)</label>
              <input 
                type="number" 
                id="koi_smass" 
                step="0.01" 
                placeholder="e.g., 1.0"
                value={formData.koi_smass}
                onChange={handleInputChange}
              />
            </div>
          </div>

          <div className="button-group">
            <button className="btn" onClick={submitPrediction}>Analyze Data</button>
            <button className="btn-secondary" onClick={fillExampleData}>
              Load Example
            </button>
            <button className="btn-secondary" onClick={() => goToPage(1)}>
              Back to Home
            </button>
          </div>

          <p className="hint">Tip: Press 'E' to auto-fill example data</p>
        </div>
      </div>

      <div className={`page ${currentPage === 4 ? 'active' : ''}`}>
        <div className="content">
          <h1 className="page-title">Prediction Results</h1>
          
          {results && (
            <>
              <div className="result-card">
                <h2>Classification</h2>
                <div className={`status-badge ${results.prediction_label.toLowerCase().replace(' ', '-')}`}>
                  {results.prediction_label === 'EXOPLANET CONFIRMED' && '✅ '}
                  {results.prediction_label === 'CANDIDATE' && '⚠️ '}
                  {results.prediction_label === 'FALSE POSITIVE' && '❌ '}
                  {results.prediction_label}
                </div>
              </div>

              <div className="result-card">
                <h2>Confidence Levels</h2>
                
                <div className="confidence-item">
                  <div className="confidence-label">
                    <span>Confirmed</span>
                    <span className="confidence-value">
                      {(results.confidence.confirmed * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="confidence-bar-container">
                    <div 
                      className="confidence-bar confirmed-bar"
                      style={{ width: `${results.confidence.confirmed * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="confidence-item">
                  <div className="confidence-label">
                    <span>Candidate</span>
                    <span className="confidence-value">
                      {(results.confidence.candidate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="confidence-bar-container">
                    <div 
                      className="confidence-bar candidate-bar"
                      style={{ width: `${results.confidence.candidate * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="confidence-item">
                  <div className="confidence-label">
                    <span>False Positive</span>
                    <span className="confidence-value">
                      {(results.confidence.false_positive * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="confidence-bar-container">
                    <div 
                      className="confidence-bar false-positive-bar"
                      style={{ width: `${results.confidence.false_positive * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              <div className="result-card">
                <h2>Interpretation</h2>
                <p className="interpretation-text">{results.interpretation}</p>
              </div>
            </>
          )}

          <div className="button-group">
            <button className="btn" onClick={resetForm}>New Prediction</button>
            <button className="btn-secondary" onClick={() => goToPage(1)}>Back to Home</button>
          </div>
        </div>
      </div>
    </>
  );
}
import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider } from './theme/ThemeProvider';
import { ToastProvider } from './components/Toast';
import { Router } from './router';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider>
      <ToastProvider>
        <Router>
          <App />
        </Router>
      </ToastProvider>
    </ThemeProvider>
  </React.StrictMode>
);

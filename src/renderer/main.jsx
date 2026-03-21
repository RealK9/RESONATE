import React from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider } from './theme/ThemeProvider';
import { ToastProvider } from './components/Toast';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider>
      <ToastProvider>
        <App />
      </ToastProvider>
    </ThemeProvider>
  </React.StrictMode>
);

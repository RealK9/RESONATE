const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

// Prevent GPU process crash loop on macOS
app.disableHardwareAcceleration();
app.commandLine.appendSwitch('disable-gpu');
app.commandLine.appendSwitch('disable-software-rasterizer');

let mainWindow;
let backendProcess = null;

// ── Backend lifecycle ──────────────────────────────────────────────────────
function getBackendDir() {
  if (app.isPackaged) {
    // In packaged app, backend is alongside the app resources
    return path.join(process.resourcesPath, 'backend');
  }
  return path.join(__dirname, '../../backend');
}

function findPython() {
  const backendDir = getBackendDir();
  // Try venv python first
  const venvPython = path.join(backendDir, 'venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) return venvPython;
  // Try venv python3
  const venvPython3 = path.join(backendDir, 'venv', 'bin', 'python3');
  if (fs.existsSync(venvPython3)) return venvPython3;
  // Fallback to system python
  return 'python3';
}

function startBackend() {
  if (process.env.NODE_ENV === 'development') {
    console.log('[RESONATE] Dev mode — backend should be started separately');
    return;
  }

  const backendDir = getBackendDir();
  const serverScript = path.join(backendDir, 'server.py');

  if (!fs.existsSync(serverScript)) {
    console.error('[RESONATE] Backend server.py not found at:', serverScript);
    return;
  }

  const pythonPath = findPython();
  console.log('[RESONATE] Starting backend:', pythonPath, serverScript);

  backendProcess = spawn(pythonPath, [serverScript], {
    cwd: backendDir,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PYTHONPATH: backendDir,
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  backendProcess.stdout.on('data', (data) => {
    console.log('[Backend]', data.toString().trim());
  });

  backendProcess.stderr.on('data', (data) => {
    console.error('[Backend ERR]', data.toString().trim());
  });

  backendProcess.on('error', (err) => {
    console.error('[RESONATE] Failed to start backend:', err.message);
  });

  backendProcess.on('exit', (code) => {
    console.log('[RESONATE] Backend exited with code:', code);
    backendProcess = null;
  });
}

function stopBackend() {
  if (backendProcess) {
    console.log('[RESONATE] Stopping backend...');
    backendProcess.kill('SIGTERM');
    // Force kill after 3 seconds if still running
    setTimeout(() => {
      if (backendProcess) {
        backendProcess.kill('SIGKILL');
        backendProcess = null;
      }
    }, 3000);
  }
}

// ── Window ─────────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 20, y: 18 },
    backgroundColor: '#08080F',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
    show: false,
  });

  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile(path.join(__dirname, '../../dist/index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ── IPC Handlers ───────────────────────────────────────────────────────────
ipcMain.handle('dialog:openFile', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select your mixdown',
    properties: ['openFile'],
    filters: [
      { name: 'Audio Files', extensions: ['wav', 'mp3', 'flac', 'aiff', 'aif', 'm4a', 'ogg'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  if (result.canceled) return null;
  return result.filePaths[0];
});

ipcMain.handle('file:readBuffer', async (event, filePath) => {
  try {
    const buffer = fs.readFileSync(filePath);
    return { data: buffer.toString('base64'), name: path.basename(filePath) };
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('drag:start', (event, filePath) => {
  try {
    if (!fs.existsSync(filePath)) return { error: 'File not found' };
    event.sender.startDrag({
      file: filePath,
      icon: path.join(__dirname, '../../public/drag-icon.png'),
    });
    return { success: true };
  } catch (err) {
    try {
      event.sender.startDrag({ file: filePath });
      return { success: true };
    } catch (e2) {
      return { error: e2.message };
    }
  }
});

// ── App lifecycle ──────────────────────────────────────────────────────────
app.whenReady().then(() => {
  startBackend();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    stopBackend();
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackend();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFile: () => ipcRenderer.invoke('dialog:openFile'),
  readFileAsBuffer: (filePath) => ipcRenderer.invoke('file:readBuffer', filePath),
  startDrag: (filePath) => ipcRenderer.invoke('drag:start', filePath),
  platform: process.platform,
});

module.exports = {
  packagerConfig: {
    name: 'RESONATE',
    executableName: 'RESONATE',
    icon: './public/icon',
    asar: true,
    appBundleId: 'com.resonate.app',
    appCategoryType: 'public.app-category.music',
  },
  makers: [
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin'],
    },
    {
      name: '@electron-forge/maker-dmg',
      config: {
        format: 'ULFO',
        name: 'RESONATE',
      },
    },
  ],
  plugins: [
    {
      name: '@electron-forge/plugin-auto-unpack-natives',
      config: {},
    },
  ],
};

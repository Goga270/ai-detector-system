export default defineNuxtConfig({
  modules: ['@nuxtjs/tailwindcss', '@nuxt/eslint'],
  devtools: { enabled: true },
  app: {
    head: {
      title: 'AI Detector',
      htmlAttrs: { lang: 'ru' },
      charset: 'utf-8'
    }
  },
  css: ['~/assets/css/main.css'],
  compatibilityDate: '2025-07-15',
  eslint: {
    config: {
      stylistic: true // Включает встроенную поддержку стилистики
    }
  },
  tailwindcss: {
    cssPath: '~/assets/css/main.css',
    viewer: true
  }
});

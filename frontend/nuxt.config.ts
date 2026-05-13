export default defineNuxtConfig({
  modules: ['@nuxtjs/tailwindcss', '@nuxt/eslint', '@nuxt/icon', 'shadcn-nuxt'],
  devtools: { enabled: true },
  app: {
    head: {
      title: 'AI Detector',
      htmlAttrs: { lang: 'ru' },
      charset: 'utf-8'
    }
  },
  css: ['~/assets/css/main.css'],
  runtimeConfig: {
    public: {
      apiBaseUrl: '',
      useMocks: false
    }
  },
  compatibilityDate: '2025-07-15',
  eslint: {
    config: {
      stylistic: true
    }
  },
  shadcn: {
    prefix: 'Ui',
    componentDir: './app/components/ui'
  },
  tailwindcss: {
    cssPath: '~/assets/css/main.css',
    viewer: true
  }
});

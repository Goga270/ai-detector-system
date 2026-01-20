/** @type {import('tailwindcss').Config} */
module.exports = {
  // Указываем, где искать классы (для Nuxt 4 это папка app)
  content: ['./app/**/*.{vue,js,ts,jsx,tsx}', './app.vue'],
  theme: {
    extend: {
      colors: {
        primary: 'rgb(var(--primary-rgb) / <alpha-value>)',
        secondary: 'rgb(var(--secondary-rgb) / <alpha-value>)',
        accent: 'rgb(var(--accent-rgb) / <alpha-value>)',
        background: 'rgb(var(--background-rgb) / <alpha-value>)',
        foreground: 'rgb(var(--foreground-rgb) / <alpha-value>)',
        muted: 'var(--muted)',
        'muted-foreground': 'var(--muted-foreground)',
        'primary-foreground': 'var(--primary-foreground)',
        'secondary-foreground': 'var(--secondary-foreground)',
        border: 'var(--border)',
        card: 'var(--card)',
        input: 'var(--input)'
      }
    }
  },
  plugins: []
};
